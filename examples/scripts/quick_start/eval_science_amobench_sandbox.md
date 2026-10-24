AMO-Bench Sandbox evaluation
============================

Evaluates the local 50-question AMO set through the shared Molt ReAct agent.
Source: `jiangyichen/p1-shared-2/eval/math/amobench.jsonl` under the research mount.
Uses the original questions and all acceptable reference answers. References stay
in the label, separate from model-facing user prompts.

Run inside a Molt GPU runtime with a consolidated HF model:

```bash
MODEL_PATH=/path/to/consolidated-hf-model \
TP=2 MAX_WORKERS=16 \
bash examples/scripts/quick_start/rjob_entry_eval_science_amobench_sandbox.sh
```

The entry loads the existing `sandbox.local.env`; credentials are not copied
into the recipe. For RJob use the same runtime image and GPFS mounts as HLE,
allocate `TP` GPUs, and run this entry. Do not pass `--private-machine` when
using `ailab_ai4sgi_pool`. No job is submitted by the entry itself.

Defaults: 120k (122880) total trajectory tokens, temperature 0.6, top_p 1,
non-masking, search/visit/python/scholar tools. This measures agent performance;
the referenced pure-inference script disallows tools, so its scores use a
different evaluation setup.

The AMO branch keeps all boxed final answers, ignores text before `</think>`,
and judges mathematical equivalence against the acceptable reference list.
It follows the reference math judge's criteria, returning boolean JSON instead
of `VERDICT` text to reuse validation and retries. Judge defaults to
`bailian/deepseek-v4.1-flash`, thinking enabled, 6000 tokens per request, with
thinking-disabled fallback on an invalid grade in the same Sandbox.

Outputs: converted input, `eval.log` with `eval_amobench_pass1` and truncation
metrics, `rollout_dump` with answers and full messages, Sandbox trace and
preflight log. `OUTPUT_ROOT` overrides the timestamped output directory.

Prepare the dataset without GPU, credentials, or API calls:

```bash
PREPARE_ONLY=1 OUTPUT_ROOT=/tmp/amo-prepare \
bash examples/scripts/quick_start/eval_science_amobench_sandbox.sh
```

Only consolidated HF directories are accepted by this recipe. Convert a
sharded training checkpoint to HF before evaluation.
