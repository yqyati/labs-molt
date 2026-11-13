"""CPU contract checks for the test-only producer, without Ray/vLLM imports."""
import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch


class ContinuousAgentTest(unittest.IsolatedAsyncioTestCase):
    async def test_refills_when_consumer_stops_and_truncates_aligned_fields(self):
        calls = []

        class FakeRunner:
            def __init__(self, env_cls):
                pass

            async def execute(self, **kwargs):
                calls.append(kwargs['sampling_params'].max_tokens)
                await asyncio.sleep(0.001)
                return SimpleNamespace(
                    observation_tokens=list(range(1200)), rollout_log_probs=[0.] * 1200,
                    routed_experts=list(range(1200)), action_ranges=[(50, 1200)],
                    off_policy_action_lens=[0], reward=0.3, truncated=False,
                )

        fake = ModuleType('molt.agents')
        fake.Env, fake.Result, fake.StepEnvRunner = object, SimpleNamespace, FakeRunner
        path = Path(__file__).with_name('partial_routing_continuous_agent.py')
        spec = importlib.util.spec_from_file_location('continuous_agent_under_test', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'molt.agents': fake}):
            spec.loader.exec_module(module)
        agent = module.AgentRunner()
        try:
            traj = await agent.execute('prompt', 'label', SimpleNamespace(max_tokens=1),
                                       32768, object(), object())
            self.assertEqual(len(traj.observation_tokens), 1024)
            self.assertEqual(len(traj.rollout_log_probs), 1024)
            self.assertEqual(traj.routed_experts, list(range(1024)))
            self.assertEqual(traj.action_ranges, [(50, 1024)])
            initial_calls = len(calls)
            await asyncio.sleep(0.05)
            self.assertGreater(len(calls), initial_calls + 4)
            self.assertLessEqual(agent.ready.qsize(), 4)
            self.assertEqual(set(calls), {2048, 8192, 16384, 24576})
        finally:
            for worker in agent.workers:
                worker.cancel()
            await asyncio.gather(*agent.workers, return_exceptions=True)


if __name__ == '__main__':
    unittest.main()
