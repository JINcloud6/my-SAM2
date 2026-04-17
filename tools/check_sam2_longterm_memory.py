#!/usr/bin/env python
"""
Check whether the imported SAM2 package contains the local long-term memory hooks.

This script intentionally imports `sam2` after argument parsing so that you can test
two different situations:

1. Current environment resolution:
   python tools/check_sam2_longterm_memory.py

2. Force this repository to the front of sys.path:
   python tools/check_sam2_longterm_memory.py --force_repo_root

The hybrid vessel pipeline expects the video predictor to provide:
- promote_frame_output_to_cond(...)
- demote_frame_output_from_cond(...)
- prune_non_cond_memory(...)
"""

import argparse
import importlib
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


REQUIRED_METHODS = (
    "promote_frame_output_to_cond",
    "demote_frame_output_from_cond",
    "prune_non_cond_memory",
)


def path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def add_result(results: List[Tuple[str, bool, str]], name: str, ok: bool, detail: str) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {detail}")
    results.append((name, ok, detail))


def resolve_device(args) -> str:
    if str(args.device).lower() == "cpu":
        return "cpu"
    if str(args.device).startswith("cuda:"):
        return str(args.device)
    if str(args.device).startswith("cuda"):
        return f"cuda:{int(args.cuda_device)}"
    return str(args.device)


def import_sam2_modules() -> Tuple[Any, Any, Any]:
    sam2_pkg = importlib.import_module("sam2")
    video_module = importlib.import_module("sam2.sam2_video_predictor")
    build_module = importlib.import_module("sam2.build_sam")
    return sam2_pkg, video_module, build_module


def method_source_summary(cls, method_name: str) -> str:
    try:
        source = inspect.getsource(getattr(cls, method_name))
    except Exception as exc:
        return f"source unavailable: {exc}"
    compact = " ".join(line.strip() for line in source.strip().splitlines()[:4])
    return compact[:240]


def run_fake_state_longterm_test(predictor_like) -> None:
    """Exercise long-term hooks on a minimal inference_state-like dictionary."""
    sentinel_move = {"frame": 5, "obj": 1}
    sentinel_other = {"frame": 5, "obj": 2}
    state: Dict[str, Any] = {
        "obj_id_to_idx": {1: 0, 2: 1},
        "output_dict_per_obj": {
            0: {
                "cond_frame_outputs": {9: {"already_cond": True}},
                "non_cond_frame_outputs": {
                    3: {"old_non_cond": True},
                    5: sentinel_move,
                    8: {"keep_non_cond": True},
                },
            },
            1: {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {5: sentinel_other},
            },
        },
    }

    predictor_like.promote_frame_output_to_cond(state, frame_idx=5, obj_id=1)
    obj1 = state["output_dict_per_obj"][0]
    obj2 = state["output_dict_per_obj"][1]
    assert obj1["cond_frame_outputs"].get(5) is sentinel_move
    assert 5 not in obj1["non_cond_frame_outputs"]
    assert obj2["non_cond_frame_outputs"].get(5) is sentinel_other

    predictor_like.demote_frame_output_from_cond(state, frame_idx=5, obj_id=1)
    assert obj1["non_cond_frame_outputs"].get(5) is sentinel_move
    assert 5 not in obj1["cond_frame_outputs"]

    predictor_like.prune_non_cond_memory(state, min_keep_frame_idx=6, obj_id=1, keep_cond=True)
    assert 3 not in obj1["non_cond_frame_outputs"]
    assert 5 not in obj1["non_cond_frame_outputs"]
    assert 8 in obj1["non_cond_frame_outputs"]
    assert 9 in obj1["cond_frame_outputs"]
    assert obj2["non_cond_frame_outputs"].get(5) is sentinel_other


def maybe_build_predictor(args, build_module):
    if not args.sam2_model_cfg and not args.sam2_checkpoint:
        return None
    if not args.sam2_model_cfg or not args.sam2_checkpoint:
        raise ValueError("Both --sam2_model_cfg and --sam2_checkpoint are required for build test.")

    import hydra

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_module("use_sam2", version_base="1.2")
    return build_module.build_sam2_video_predictor(
        args.sam2_model_cfg,
        args.sam2_checkpoint,
        device=resolve_device(args),
    )


def main() -> int:
    inferred_repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Diagnose whether the active SAM2 import has the local long-term memory hooks."
    )
    parser.add_argument("--repo_root", default=str(inferred_repo_root))
    parser.add_argument(
        "--force_repo_root",
        action="store_true",
        help="Insert --repo_root at sys.path[0] before importing sam2.",
    )
    parser.add_argument(
        "--no_expect_repo",
        action="store_true",
        help="Do not fail when imported sam2 is outside --repo_root.",
    )
    parser.add_argument("--sam2_model_cfg", default=None)
    parser.add_argument("--sam2_checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if any check fails. Without this flag the script only reports diagnostics.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.force_repo_root:
        sys.path.insert(0, str(repo_root))

    print("=" * 80)
    print("SAM2 long-term memory import diagnostic")
    print(f"Python executable: {sys.executable}")
    print(f"Current working dir: {Path.cwd()}")
    print(f"Expected repo root: {repo_root}")
    print(f"force_repo_root: {args.force_repo_root}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', '')}")
    print("sys.path head:")
    for i, item in enumerate(sys.path[:8]):
        print(f"  [{i}] {item}")
    print("=" * 80)

    results: List[Tuple[str, bool, str]] = []

    try:
        sam2_pkg, video_module, build_module = import_sam2_modules()
    except Exception as exc:
        add_result(results, "import sam2", False, repr(exc))
        return 1 if args.strict else 0

    sam2_file = Path(getattr(sam2_pkg, "__file__", "")).resolve()
    video_file = Path(inspect.getfile(video_module)).resolve()
    build_file = Path(inspect.getfile(build_module)).resolve()

    print(f"Imported sam2.__file__: {sam2_file}")
    print(f"Imported sam2_video_predictor.py: {video_file}")
    print(f"Imported build_sam.py: {build_file}")

    expected_pkg_dir = repo_root / "sam2"
    imported_from_repo = path_is_relative_to(sam2_file, expected_pkg_dir)
    if args.no_expect_repo:
        add_result(results, "sam2 import path", True, "path check disabled by --no_expect_repo")
    else:
        add_result(
            results,
            "sam2 import path",
            imported_from_repo,
            "using local repository sam2" if imported_from_repo else "NOT using local repository sam2",
        )

    cls = getattr(video_module, "SAM2VideoPredictor", None)
    add_result(results, "SAM2VideoPredictor class", cls is not None, str(cls))
    if cls is None:
        return 1 if args.strict else 0

    for method_name in REQUIRED_METHODS:
        exists = hasattr(cls, method_name)
        detail = method_source_summary(cls, method_name) if exists else "missing"
        add_result(results, f"class has {method_name}", exists, detail)

    if all(hasattr(cls, method_name) for method_name in REQUIRED_METHODS):
        try:
            fake_predictor = cls.__new__(cls)
            run_fake_state_longterm_test(fake_predictor)
            add_result(results, "fake inference_state behavior", True, "promote/demote/prune mutated state correctly")
        except Exception as exc:
            add_result(results, "fake inference_state behavior", False, repr(exc))

    try:
        predictor = maybe_build_predictor(args, build_module)
    except Exception as exc:
        add_result(results, "build video predictor", False, repr(exc))
        predictor = None

    if predictor is not None:
        for method_name in REQUIRED_METHODS:
            add_result(
                results,
                f"built predictor has {method_name}",
                hasattr(predictor, method_name),
                f"type={type(predictor)}",
            )
        if all(hasattr(predictor, method_name) for method_name in REQUIRED_METHODS):
            try:
                run_fake_state_longterm_test(predictor)
                add_result(results, "built predictor fake-state behavior", True, "ok")
            except Exception as exc:
                add_result(results, "built predictor fake-state behavior", False, repr(exc))

    print("=" * 80)
    failed = [name for name, ok, _detail in results if not ok]
    if failed:
        print("Result: FAILED checks")
        for name in failed:
            print(f"  - {name}")
    else:
        print("Result: all checks passed")
    print("=" * 80)

    return 1 if args.strict and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
