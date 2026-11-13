"""Test-only continuous generation; retain bounded, short training trajectories."""
import asyncio
from collections import deque
from copy import deepcopy
import json
import random

from molt.agents import Env, Result, StepEnvRunner


class RandomRewardEnv(Env):
    async def step(self, state):
        return Result(reward=random.uniform(-1.0, 1.0), terminated=True)


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(RandomRewardEnv)
        self.seeds = deque(maxlen=128)
        self.ready = asyncio.Queue(maxsize=4)
        self.workers = []
        self.failure = None

    async def _produce(self, lane):
        try:
            while True:
                kwargs = dict(self.seeds[0])
                kwargs['sampling_params'] = deepcopy(kwargs['sampling_params'])
                self.seeds.rotate(-1)
                sp = kwargs['sampling_params']
                # Stagger lengths while maintaining continuous replacement requests.
                sp.max_tokens = (2048, 8192, 16384, 24576)[lane]
                sp.min_tokens = sp.max_tokens
                kwargs['max_length'] = 32768
                traj = await super().execute(**kwargs)
                original_length = len(traj.observation_tokens)
                assert traj.routed_experts is not None, 'routing missing'
                assert traj.rollout_log_probs is not None, 'logprobs missing'
                traj.observation_tokens = traj.observation_tokens[:1024]
                traj.rollout_log_probs = traj.rollout_log_probs[:1024]
                traj.routed_experts = traj.routed_experts[:1024]
                ranges = [(start, min(end, 1024)) for start, end in traj.action_ranges if start < 1024]
                assert ranges, 'prompt consumes the entire training window'
                traj.action_ranges = ranges
                traj.off_policy_action_lens = traj.off_policy_action_lens[:len(ranges)]
                traj.truncated = True
                print('[continuous_test] ' + json.dumps({
                    'generated_sequence_length': original_length,
                    'training_sequence_length': len(traj.observation_tokens),
                    'reward': traj.reward,
                }), flush=True)
                # Never stop refilling vLLM because the training queue is full.
                if self.ready.full():
                    self.ready.get_nowait()
                self.ready.put_nowait(traj)
        except Exception as exc:
            self.failure = exc
            if self.ready.full():
                self.ready.get_nowait()
            self.ready.put_nowait(None)

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer,
                      llm_engine, images=None, tools=None):
        assert not images, 'this mechanism test is text-only'
        # Keep clients/tokenizers by reference: neither is safe to deepcopy.
        self.seeds.append(dict(prompt=prompt, label=label, sampling_params=sampling_params,
                               max_length=max_length, hf_tokenizer=hf_tokenizer,
                               llm_engine=llm_engine, images=None, tools=tools))
        if not self.workers:
            self.workers = [asyncio.create_task(self._produce(lane)) for lane in range(4)]
        if self.failure is not None:
            raise RuntimeError('continuous generation failed') from self.failure
        traj = await self.ready.get()
        if traj is None or self.failure is not None:
            raise RuntimeError('continuous generation failed') from self.failure
        return traj
