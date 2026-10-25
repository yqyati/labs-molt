"""Exercise the adapter retry loop without loading GPU or harness dependencies."""
import ast
import asyncio
import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class SandboxGatewayError(RuntimeError):
    pass


class SandboxStateError(SandboxGatewayError):
    pass


class JudgeThinkingFallbackTests(unittest.TestCase):
    def run_judge(self, responses, *, protocol=True, thinking="enabled", recovery=False, retries=(), direct_fallback=False):
        source = Path(__file__).resolve().parents[2] / "examples/python/agents/science_legacy_react_sandbox.py"
        tree = ast.parse(source.read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == ("sandbox_attempt" if recovery else "attempt"))
        wrapper = ast.AsyncFunctionDef(
            name="judge", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=ast.parse("judge_attempts = 0\njudge_thinking_fallbacks = 0\njudge_result = None\n"
                           "judge_error = ''\njudge_response_summary = ''\njudge_finish_reason = ''\n"
                           "judge_content_chars = 0\njudge_connection_failures = 0\njudge_run = initial_run\n"
                           "direct = False\njudge_backend = 'gateway_sandbox'\njudge_primary_error = ''\n"
                           "judge_model = 'test'\njudge_url = 'http://test'\n").body
                 + [loop, ast.Return(ast.Call(ast.Name("locals", ast.Load()), [], []))], decorator_list=[])
        namespace = dict(asyncio=asyncio, json=json, os=os, _text=lambda x: "" if x is None else str(x),
                         logger=logging.getLogger(__name__), judge_attempts=0, judge_thinking_fallbacks=0,
                         judge_result=None, judge_error="", judge_response_summary="", judge_finish_reason="",
                         judge_content_chars=0, retry_delays=retries, judge_model="test", judge_url="http://test",
                         judge_system_prompt="grade", judge_user_prompt="candidate", next_protocol=protocol,
                         sandbox_attempt=0, prediction="answer", prediction_missing=False, prediction_empty=False,
                         fallback_base="http://direct/v1" if direct_fallback else "", direct=False)
        calls = []
        sandbox_calls, closed = [], []
        instances = []
        direct_options = []

        class DirectClient:
            def __init__(self, **kwargs):
                direct_options.append(kwargs)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def post(self, url, *, json, headers):
                calls.append(dict(json))
                direct_options[-1].update(url=url, headers=headers)
                response = responses[len(calls) - 1]
                if isinstance(response, Exception):
                    raise response
                return SimpleNamespace(status_code=response["status_code"], text=response["text"])

        class Run:
            def __init__(self):
                self.index = len(instances)
                instances.append(self)

            def run_http(self, request, timeout):
                calls.append(json.loads(json.dumps(request["json"])))
                sandbox_calls.append(self.index)
                response = responses[len(calls) - 1]
                if isinstance(response, Exception):
                    raise response
                return response

            def close(self):
                closed.append(self.index)

        namespace["initial_run"] = Run()
        namespace.update(AgentSandboxRun=Run, SandboxGatewayError=SandboxGatewayError, SandboxStateError=SandboxStateError,
                         httpx=SimpleNamespace(AsyncClient=DirectClient))
        exec(compile(ast.fix_missing_locations(ast.Module([wrapper], type_ignores=[])), str(source), "exec"), namespace)

        async def run_inline(func, *args, **kwargs):
            return func(*args, **kwargs)

        with patch.dict(os.environ, SCIENCE_AGENT_JUDGE_THINKING=thinking, API_KEY="test"), \
                patch.object(asyncio, "to_thread", run_inline):
            result = asyncio.run(namespace["judge"]())
        result.update(sandbox_calls=sandbox_calls, closed=closed, direct_options=direct_options)
        return result, calls

    @staticmethod
    def response(content, *, status=200):
        return {"status_code": status, "text": json.dumps({"choices": [
            {"finish_reason": "stop", "message": {"content": content}}]})}

    def test_valid_false_is_not_a_fallback(self):
        result, calls = self.run_judge([self.response('{"correct": false}')])
        self.assertEqual(result["judge_result"]["score"], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["judge_thinking_fallbacks"], 0)

    def test_invalid_output_falls_back_once_in_both_protocols(self):
        for protocol in (True, False):
            for invalid in ('', 'not json', '[]', '{"correct": "false"}'):
                with self.subTest(protocol=protocol, invalid=invalid):
                    result, calls = self.run_judge([self.response(invalid),
                        self.response('{"correct": true, "score": 1}')], protocol=protocol)
                    self.assertEqual(result["judge_result"]["score"], 1)
                    self.assertEqual([r["thinking"]["type"] for r in calls], ["enabled", "disabled"])
                    self.assertEqual(result["judge_attempts"], 2)
                    self.assertEqual(result["judge_thinking_fallbacks"], 1)

    def test_failed_fallback_remains_ungraded(self):
        result, calls = self.run_judge([self.response(''), self.response('')])
        self.assertIsNone(result["judge_result"])
        self.assertEqual(len(calls), 2)
        self.assertIn('JSONDecodeError', result["judge_error"])

    def test_transport_error_does_not_trigger_thinking_fallback(self):
        result, calls = self.run_judge([self.response('', status=503)])
        self.assertIsNone(result["judge_result"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["judge_thinking_fallbacks"], 0)

    def test_disabled_thinking_has_no_second_attempt(self):
        result, calls = self.run_judge([self.response('')], thinking="disabled")
        self.assertIsNone(result["judge_result"])
        self.assertEqual(len(calls), 1)

    def test_pending_recovery_switches_sandbox_without_retrying_notice(self):
        result, calls = self.run_judge([SandboxStateError('pending recovery notice'),
            self.response('{"correct": true}')], recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 1])
        self.assertEqual(result['closed'], [1])
        self.assertEqual(result['judge_result']['score'], 1)
        self.assertEqual([r['messages'] for r in calls], [calls[0]['messages']] * 2)

    def test_two_connection_failures_switch_sandbox(self):
        error = SandboxGatewayError('proxy HTTP 502 (attempt 3/3): Sandbox upstream connection failed')
        result, _ = self.run_judge([error, error, self.response('{"correct": false}')],
                                   recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 0, 1])
        self.assertEqual(result['closed'], [1])
        self.assertEqual(result['judge_result']['score'], 0)

    def test_api_error_breaks_connection_failure_streak(self):
        error = SandboxGatewayError('proxy HTTP 502: Sandbox upstream connection failed')
        result, _ = self.run_judge([error, self.response('', status=503), error,
            self.response('{"correct": true}')], recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 0, 0, 0])
        self.assertEqual(result['judge_result']['score'], 1)

    def test_unusable_replacements_are_closed_and_remain_ungraded(self):
        result, _ = self.run_judge([SandboxStateError('pending notice')] * 2,
                                   recovery=True, retries=(0,) * 6)
        self.assertEqual(result['sandbox_calls'], [0, 1])
        self.assertEqual(result['closed'], [1])
        self.assertIsNone(result['judge_result'])

    def test_direct_fallback_runs_only_after_all_sandboxes_fail(self):
        for protocol, correct in ((True, False), (False, True)):
            with self.subTest(protocol=protocol):
                result, calls = self.run_judge([SandboxStateError('pending notice')] * 2 + [
                    self.response(json.dumps({'correct': correct, 'score': int(correct)}))],
                    protocol=protocol, recovery=True, direct_fallback=True)
                self.assertEqual(result['sandbox_calls'], [0, 1])
                self.assertEqual(result['closed'], [1])
                self.assertEqual(result['judge_backend'], 'direct_fallback')
                self.assertEqual(result['judge_result']['score'], int(correct))
                self.assertEqual(result['judge_error'], '')
                self.assertIn('pending notice', result['judge_primary_error'])
                self.assertEqual(calls[-1]['model'], 'DeepSeek-V4.1-Flash-1017')
                self.assertEqual(calls[-1]['chat_template_kwargs'], {'thinking': True, 'reasoning_effort': 20})
                self.assertNotIn('thinking', calls[-1])
                self.assertEqual([r['messages'] for r in calls], [calls[0]['messages']] * 3)
                options = result['direct_options'][0]
                self.assertFalse(options['trust_env'])
                self.assertEqual(options['headers'], {'Authorization': 'Bearer EMPTY'})
                self.assertEqual(options['url'], 'http://direct/v1/chat/completions')

    def test_valid_sandbox_result_never_calls_direct_fallback(self):
        result, calls = self.run_judge([self.response('{"correct": false}')], recovery=True, direct_fallback=True)
        self.assertEqual(result['judge_result']['score'], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(result['direct_options'], [])

    def test_direct_invalid_thinking_output_retries_without_thinking(self):
        result, calls = self.run_judge([SandboxStateError('pending notice')] * 2 + [
            self.response(''), self.response('{"correct": true}')], recovery=True, direct_fallback=True)
        self.assertEqual(result['judge_result']['score'], 1)
        self.assertEqual(calls[-1]['chat_template_kwargs'], {'thinking': False})

    def test_direct_failure_keeps_answer_ungraded(self):
        for response in (TimeoutError('direct timeout'), self.response('', status=503), self.response('')):
            with self.subTest(response=response):
                replies = [SandboxStateError('pending notice')] * 2 + [response, self.response('')]
                result, _ = self.run_judge(replies, recovery=True, direct_fallback=True)
                self.assertIsNone(result['judge_result'])
                self.assertTrue(result['judge_error'])
                self.assertEqual(result['closed'], [1])


if __name__ == "__main__":
    unittest.main()
