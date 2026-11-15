"""Synchronize NeMo metadata write failures before its internal save barriers."""

import ast
from pathlib import Path
import textwrap


METADATA_WRITE = '''
@contextlib.contextmanager
def _molt_metadata_write(process_group):
    error = None
    try:
        yield
    except Exception as exc:
        error = exc
    status = None
    if error is not None:
        full = (
            isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT)
        ) or bool(re.search(r"no space left on device|disk quota exceeded|\\[Errno (28|122)\\]", str(error), re.I))
        status = (full, str(error))
    statuses = [status]
    if torch.distributed.is_initialized():
        statuses = [None] * torch.distributed.get_world_size(group=process_group)
        torch.distributed.all_gather_object(statuses, status, group=process_group)
    failures = [item for item in statuses if item is not None]
    if failures:
        if all(full for full, _ in failures):
            raise OSError(errno.ENOSPC, failures[0][1]) from error
        raise RuntimeError("Checkpoint metadata write failed: " + failures[0][1]) from error
'''


def patch_source(source):
    if "def _molt_metadata_write(" in source:
        return source
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    replacements = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in {"ConsolidatedHFAddon", "PeftAddon"}:
            continue
        for method in node.body:
            if not isinstance(method, ast.FunctionDef):
                continue
            for first, last in zip(method.body, method.body[1:]):
                if (isinstance(first, ast.If) and ast.unparse(first.test) == "_is_group_rank_0(process_group)"
                        and isinstance(last, ast.Expr)
                        and ast.unparse(last.value) == "_group_barrier(process_group)"):
                    block = "".join(lines[first.lineno - 1:first.end_lineno])
                    replacements.append((first.lineno - 1, last.end_lineno,
                                         "        with _molt_metadata_write(process_group):\n"
                                         + textwrap.indent(block, "    ")))
    if len(replacements) != 3:
        raise RuntimeError(f"Unsupported NeMo checkpoint addons: expected 3 metadata hooks, found {len(replacements)}")
    for start, end, replacement in sorted(replacements, reverse=True):
        lines[start:end] = [replacement]
    patched = "".join(lines) + "\nimport contextlib\nimport errno\nimport re\n" + METADATA_WRITE
    compile(patched, "addons.py", "exec")
    return patched


if __name__ == "__main__":
    import nemo_automodel.components.checkpoint.addons as addons

    path = Path(addons.__file__)
    source = path.read_text()
    patched = patch_source(source)
    if patched != source:
        temporary = path.with_suffix(".py.molt-tmp")
        temporary.write_text(patched)
        temporary.replace(path)
    print(f"NeMo checkpoint metadata error synchronization ready: {path}")
