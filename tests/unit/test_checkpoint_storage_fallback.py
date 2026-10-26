"""Exercise real checkpoint save/retry paths without Ray, GPUs or AutoModel."""

import ast
import errno
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch.distributed as dist
import torch.multiprocessing as mp


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("checkpoint_io", ROOT / "molt/trainer/fsdp/checkpoint.py")
checkpoint = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checkpoint)
tree = ast.parse((ROOT / "molt/trainer/workers/policy_actor.py").read_text())
actor = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PolicyModelActor")
methods = [n for n in actor.body if isinstance(n, ast.FunctionDef) and n.name in {
    "_save_with_storage_fallback", "save_checkpoint", "export_hf_model",
}]
namespace = {"errno": errno, "os": os, "torch_dist_barrier_and_cuda_sync": lambda: None}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
    name="Actor", bases=[], keywords=[], body=methods, decorator_list=[],
)], type_ignores=[])), "policy_actor.py", "exec"), namespace)


def distributed_writer(rank, rendezvous, directory):
    from datetime import timedelta

    dist.init_process_group("gloo", init_method="file://" + rendezvous,
                            rank=rank, world_size=2, timeout=timedelta(seconds=20))
    try:
        manager = checkpoint.CheckpointManager(SimpleNamespace())
        actor = namespace["Actor"]()
        actor.strategy = SimpleNamespace(print=lambda *a: None)
        calls = []

        def save(output_dir):
            calls.append(output_dir)
            with manager._checkpoint_write():
                if rank == 1 and output_dir == "/primary/checkpoint":
                    raise OSError(errno.EDQUOT, "injected rank-local quota failure")

        with patch.dict(os.environ, {"MOLT_CKPT_FALLBACK_ROOT": directory + "/fallback"}):
            saved = actor._save_with_storage_fallback(save, "output_dir", "/primary/checkpoint")
        Path(directory, f"rank{rank}.json").write_text(json.dumps({"calls": calls, "saved": saved}))
    finally:
        dist.destroy_process_group()


class CheckpointStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.fallback = self.root / "fallback"
        env = patch.dict(os.environ, {"MOLT_CKPT_FALLBACK_ROOT": str(self.fallback)})
        env.start()
        self.addCleanup(env.stop)
        self.actor = namespace["Actor"]()
        self.actor.strategy = SimpleNamespace(print=Mock())

    def test_quota_and_space_retry_then_future_saves_stay_on_fallback(self):
        for number in (errno.EDQUOT, errno.ENOSPC):
            with self.subTest(errno=number):
                self.actor._checkpoint_fallback_active = False
                save = Mock(side_effect=[OSError(number, "full"), None, None])
                first = self.actor._save_with_storage_fallback(save, "output_dir", "/primary/hf")
                second = self.actor._save_with_storage_fallback(save, "output_dir", "/primary/hf-next")
                self.assertEqual(first, str(self.fallback / "primary/hf"))
                self.assertEqual(second, str(self.fallback / "primary/hf-next"))
                self.assertEqual([c.kwargs["output_dir"] for c in save.call_args_list],
                                 ["/primary/hf", first, second])

    def test_success_and_unrelated_errors_do_not_use_fallback(self):
        save = Mock()
        self.assertEqual(self.actor._save_with_storage_fallback(save, "output_dir", "/primary"), "/primary")
        save.assert_called_once_with(output_dir="/primary")
        for error in (OSError(errno.EACCES, "denied"), RuntimeError("bad model")):
            save = Mock(side_effect=error)
            with self.assertRaises(type(error)):
                self.actor._save_with_storage_fallback(save, "output_dir", "/primary")
            self.assertEqual(save.call_count, 1)

    def test_fallback_failure_and_disabled_fallback_propagate(self):
        save = Mock(side_effect=OSError(errno.ENOSPC, "full"))
        with self.assertRaises(OSError):
            self.actor._save_with_storage_fallback(save, "output_dir", "/primary")
        self.assertEqual(save.call_count, 2)
        save.reset_mock()
        with patch.dict(os.environ, {"MOLT_CKPT_FALLBACK_ROOT": ""}), self.assertRaises(OSError):
            self.actor._save_with_storage_fallback(save, "output_dir", "/primary")
        self.assertEqual(save.call_count, 1)

    def test_real_dcp_save_retries_after_sidecar_quota_without_advancing_original_latest(self):
        manager = checkpoint.CheckpointManager(SimpleNamespace(_unwrap_model=lambda m: m))
        manager._build_checkpointer = Mock(return_value=SimpleNamespace(save_model=Mock()))
        original_write = manager._atomic_write_torch
        primary = self.root / "primary/state/_actor"
        primary.mkdir(parents=True)
        (primary / "latest").write_text("global_step40")

        def write(path, payload):
            if str(path).startswith(str(primary)):
                raise OSError(errno.EDQUOT, "sidecar quota")
            original_write(path, payload)

        manager._atomic_write_torch = write
        saved = self.actor._save_with_storage_fallback(
            manager.save_ckpt, "ckpt_path", str(primary), model=object(), tag="global_step50",
            client_states={"global_step": 50}, max_num=0,
        )
        self.assertEqual((primary / "latest").read_text(), "global_step40")
        self.assertEqual((Path(saved) / "latest").read_text(), "global_step50")
        self.assertTrue((Path(saved) / "global_step50/extra_state.pt").is_file())

    def test_dcp_wrapped_baseexception_is_recognized(self):
        from torch.distributed.checkpoint.api import CheckpointException

        manager = checkpoint.CheckpointManager(SimpleNamespace())
        try:
            raise OSError(errno.EDQUOT, "quota")
        except OSError as error:
            wrapped = CheckpointException("write", {0: (error, [])})
        with self.assertRaises(OSError) as caught:
            with manager._checkpoint_write():
                raise wrapped
        self.assertEqual(caught.exception.errno, errno.ENOSPC)

    def test_actor_rolling_hf_and_final_export_use_selected_storage(self):
        args = SimpleNamespace(ckpt=SimpleNamespace(
            path="/primary/state", output_dir="/primary/hf", dcp_max_num=3, max_mem=0,
            max_num=2, save_optimizer=False,
        ))
        strategy = self.actor.strategy
        strategy.args = args
        strategy.save_ckpt = Mock(side_effect=[OSError(errno.EDQUOT, "full"), None])
        strategy.save_model = Mock()
        strategy.prune_checkpoints = Mock()
        self.actor.actor = SimpleNamespace(model=object())
        self.actor.tokenizer = object()
        self.actor.save_hf_ckpt = True
        self.actor.save_checkpoint("global_step50", {"global_step": 50})
        self.actor.export_hf_model()
        self.assertEqual(strategy.save_model.call_args_list[0].kwargs["output_dir"],
                         str(self.fallback / "primary/state/_hf/global_step50"))
        self.assertEqual(strategy.save_model.call_args_list[1].kwargs["output_dir"],
                         str(self.fallback / "primary/hf"))
        strategy.prune_checkpoints.assert_called_once_with(
            str(self.fallback / "primary/state/_hf"), "global_step50", 2, 0,
        )

    def test_mixed_dcp_errors_do_not_trigger_storage_retry(self):
        from torch.distributed.checkpoint.api import CheckpointException

        manager = checkpoint.CheckpointManager(SimpleNamespace())
        wrapped = CheckpointException("write", {
            0: (OSError(errno.EDQUOT, "quota"), []),
            1: (OSError(errno.EACCES, "denied"), []),
        })
        with self.assertRaises(CheckpointException):
            with manager._checkpoint_write():
                raise wrapped

    def test_two_ranks_switch_together_when_only_one_rank_hits_quota(self):
        mp.spawn(distributed_writer, args=(str(self.root / "rendezvous"), str(self.root)),
                 nprocs=2, join=True)
        records = [json.loads((self.root / f"rank{rank}.json").read_text()) for rank in range(2)]
        self.assertEqual(records[0], records[1])
        self.assertEqual(len(records[0]["calls"]), 2)
        self.assertEqual(records[0]["saved"], str(self.fallback / "primary/checkpoint"))


if __name__ == "__main__":
    unittest.main()
