"""Start the normal trainer with explicitly propagated test-only instrumentation."""
import os
import runpy
import sys

import ray

_original_init = ray.init


def _init_with_test_env(*args, **kwargs):
    runtime_env = dict(kwargs.pop('runtime_env', {}) or {})
    env = dict(runtime_env.get('env_vars', {}))
    env.update({name: os.environ[name] for name in
                ('MOLT_TEST_WEIGHT_VALUE_AUDIT', 'PYTHONPATH', 'RAY_DEDUP_LOGS')})
    runtime_env['env_vars'] = env
    return _original_init(*args, runtime_env=runtime_env, **kwargs)


if __name__ == '__main__':
    if ray.is_initialized():
        raise RuntimeError('test requires its own driver runtime environment')
    # This fresh-run recipe has one pause and one refit check per training step.
    # Resuming checkpoints would insert an additional initial broadcast.
    if any('load_checkpoint' in arg or 'resume' in arg for arg in sys.argv[1:]):
        raise RuntimeError('weight audit requires a fresh run, without checkpoint resume')
    assert os.environ.get('MOLT_TEST_WEIGHT_VALUE_AUDIT') == '1'
    from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap
    assert WorkerWrap.reset_weight_update_check.__module__ == 'sitecustomize', 'test hook not loaded'
    ray.init = _init_with_test_env
    runpy.run_module('molt.cli.train_rl_ray', run_name='__main__')
