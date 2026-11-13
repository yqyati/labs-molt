"""Supervise the mechanism test; never count ordinary training completion as proof."""
import ast
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys


def main():
    report_path = Path(sys.argv[1])
    command = sys.argv[sys.argv.index('--') + 1:]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, start_new_session=True)
    training_started = False
    pauses = set()
    verified = set()
    steps = []
    error = None
    weight_check_unavailable = False
    changed_weight_updates = set()
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
            training_started |= 'Train epoch' in clean or 'Global step ' in clean
            if '[check_weight_update] cannot verify' in clean:
                weight_check_unavailable = True
            if '[test_weight_values] {' in clean:
                try:
                    event = json.JSONDecoder().raw_decode(clean.split('[test_weight_values] ', 1)[1])[0]
                    if event['event'] == 'refit' and event['changed_parameters'] > 0:
                        changed_weight_updates.add(event['refit_index'])
                except (KeyError, ValueError):
                    error = 'invalid weight-value audit event'
                    break
            if 'Global step ' in clean and '{' in clean:
                try:
                    metrics = ast.literal_eval(clean[clean.index('{'):].strip())
                    if metrics.get('actor_grad_norm', 0) > 0 and metrics.get('timing/broadcast', 0) > 0:
                        steps.append(metrics['actor_grad_norm'])
                except (ValueError, SyntaxError):
                    pass
            if '[partial_routing] ' in clean:
                payload = clean.split('[partial_routing] ', 1)[1].strip()
                if payload.startswith(('FAIL', 'MISSING_ROUTING', 'UNVERIFIED')):
                    error = payload
                    break
                if payload.startswith('{'):
                    event = json.JSONDecoder().raw_decode(payload)[0]
                    boundary = event['boundary']
                    if event['event'] == 'paused' and training_started:
                        if not event['requests_with_tokens'] or not event['snapshots']:
                            error = f'pause {boundary} missed an active request with captured routing'
                            break
                        pauses.add(boundary)
                    if event['event'] == 'finished' and boundary in pauses:
                        if event['continued'] and event['prefix_preserved']:
                            verified.add(boundary)
            # Stop after evidence is collected; no need to drain 128 training groups.
            if len(verified) >= 3 and len(steps) >= 3:
                break
    except BaseException:
        error = 'supervisor interrupted or log parsing failed'
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        routing_verified = len(verified) >= 3 and len(steps) >= 3 and error is None
        all_verified = routing_verified and bool(verified & changed_weight_updates)
        report = dict(
            status='PASS' if all_verified else (
                'ROUTING_VERIFIED_WEIGHT_VALUES_UNVERIFIED' if routing_verified else 'FAIL'
            ),
            error=error, verified_boundaries=sorted(verified),
            nonzero_gradient_and_broadcast_steps=len(steps),
            changed_weight_updates=sorted(changed_weight_updates),
            verified_weight_and_routing_boundaries=sorted(verified & changed_weight_updates),
            weight_checker_unavailable=weight_check_unavailable,
            weight_values_verified=bool(verified & changed_weight_updates),
        )
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report), flush=True)
    # Broadcast timing proves the path ran, not that vLLM parameter values changed.
    return 0 if all_verified else (2 if routing_verified else 1)


if __name__ == '__main__':
    sys.exit(main())
