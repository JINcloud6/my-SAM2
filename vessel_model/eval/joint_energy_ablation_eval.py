"""
Run ablations for the MSJI adjacent-frame energy design.

The script keeps the segmentation pipeline fixed and only changes the dynamic
programming energy used by joint initialization. The new "unweighted_terms"
mode sums selected terms directly, so the formula itself does not introduce
extra weights. By default, a fixed unary score term is kept for all unweighted
variants, and the ablation focuses on the adjacent-frame pairwise terms.
"""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from .sweep_and_eval import calculate_all_metrics, load_data, print_metrics, save_results_to_csv


CONTROLLED_FLAGS = {
    "--disable_joint_init",
    "--joint_energy_mode",
    "--joint_energy_terms",
    "--output_filename",
}


PAIRWISE_TERM_HELP = (
    "Available pairwise terms: iou, dice, centroid_radius, centroid_diag, "
    "area_log, radius_log, area_ratio, radius_ratio, containment, empty_pair. "
    "centroid_radius = centroid distance / mean equivalent mask radius."
)


@dataclass(frozen=True)
class EnergyVariant:
    name: str
    mode: str
    joint_energy_terms: Optional[str]
    pairwise_terms: str


def sanitize_name(text: str) -> str:
    safe = []
    for ch in text.lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in {"+", ",", "-", "_"}:
            safe.append("_")
    name = "".join(safe).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "none"


def parse_terms(text: str) -> List[str]:
    return [term.strip().lower() for term in text.replace(",", "+").split("+") if term.strip()]


def make_unweighted_variant(
    pairwise_terms: Sequence[str],
    include_score_unary: bool,
    name: Optional[str] = None,
) -> EnergyVariant:
    full_terms: List[str] = []
    if include_score_unary:
        full_terms.append("score")
    for term in pairwise_terms:
        if term not in full_terms:
            full_terms.append(term)

    term_text = ",".join(full_terms)
    pairwise_text = "+".join(pairwise_terms) if pairwise_terms else "none"
    variant_name = name or sanitize_name("+".join(full_terms) if full_terms else "no_terms")
    return EnergyVariant(
        name=variant_name,
        mode="unweighted_terms",
        joint_energy_terms=term_text,
        pairwise_terms=pairwise_text,
    )


def parse_custom_term_sets(text: str, include_score_unary: bool) -> List[EnergyVariant]:
    variants: List[EnergyVariant] = []
    for raw_item in text.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            name_text, terms_text = item.split("=", 1)
            name = sanitize_name(name_text)
        else:
            terms_text = item
            name = None
        pairwise_terms = parse_terms(terms_text)
        variants.append(make_unweighted_variant(pairwise_terms, include_score_unary, name=name))
    return variants


def default_pairwise_sets(suite: str) -> List[List[str]]:
    minimal = [
        [],
        ["iou"],
        ["centroid_radius"],
        ["area_log"],
        ["iou", "centroid_radius"],
        ["iou", "centroid_radius", "area_log"],
    ]
    if suite == "minimal":
        return minimal
    if suite != "default":
        raise ValueError(f"Unsupported energy_suite: {suite}")

    return [
        [],
        ["iou"],
        ["dice"],
        ["centroid_radius"],
        ["centroid_diag"],
        ["area_log"],
        ["radius_log"],
        ["area_ratio"],
        ["radius_ratio"],
        ["containment"],
        ["iou", "centroid_radius"],
        ["iou", "area_log"],
        ["centroid_radius", "area_log"],
        ["iou", "centroid_radius", "area_log"],
        ["dice", "centroid_radius", "area_log"],
        ["iou", "centroid_radius", "area_ratio"],
        ["iou", "centroid_radius", "area_log", "containment"],
    ]


def build_energy_variants(args) -> List[EnergyVariant]:
    variants: List[EnergyVariant] = []
    if not args.no_legacy_weighted:
        variants.append(
            EnergyVariant(
                name="legacy_weighted",
                mode="legacy_weighted",
                joint_energy_terms=None,
                pairwise_terms="legacy_weighted",
            )
        )

    include_score_unary = not args.omit_score_unary
    if args.energy_term_sets:
        variants.extend(parse_custom_term_sets(args.energy_term_sets, include_score_unary))
    else:
        for pairwise_terms in default_pairwise_sets(args.energy_suite):
            variants.append(make_unweighted_variant(pairwise_terms, include_score_unary))
    return variants


def make_result_name(prefix: str, index: int, variant: EnergyVariant) -> str:
    prefix_str = f"{prefix}_" if prefix else ""
    return f"{prefix_str}{index:02d}_joint_energy_{sanitize_name(variant.name)}.nii.gz"


def ensure_no_controlled_passthrough(passthrough_args: Sequence[str]) -> None:
    bad = []
    for arg in passthrough_args:
        for flag in CONTROLLED_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                bad.append(arg)
    if bad:
        raise ValueError(
            "These flags are controlled by joint_energy_ablation_eval.py and must not be passed manually: "
            + ", ".join(sorted(set(bad)))
        )


def has_passthrough_flag(passthrough_args: Sequence[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in passthrough_args)


def build_child_cmd(
    args,
    passthrough_args: Sequence[str],
    variant: EnergyVariant,
    result_name: str,
) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        args.segmentation_module,
        "--sam2_checkpoint",
        args.sam2_checkpoint,
        "--sam2_model_cfg",
        args.sam2_model_cfg,
        "--volume_path",
        args.volume_path,
        "--output_dir",
        args.output_dir,
        "--output_filename",
        result_name,
        "--dataset_key",
        args.dataset_key,
        "--joint_init_axis_mode",
        args.joint_init_axis_mode,
        "--joint_energy_mode",
        variant.mode,
    ]
    if variant.joint_energy_terms:
        cmd.extend(["--joint_energy_terms", variant.joint_energy_terms])
    if args.seed_file:
        cmd.extend(["--seed_file", args.seed_file])
    if args.init_seg_path:
        cmd.extend(["--init_seg_path", args.init_seg_path])
    cmd.extend(passthrough_args)
    return cmd


def empty_metric_row(
    variant: EnergyVariant,
    result_name: str,
    result_path: str,
    status: str,
) -> Dict[str, Any]:
    return {
        "variant_name": variant.name,
        "energy_mode": variant.mode,
        "joint_energy_terms": variant.joint_energy_terms,
        "pairwise_terms": variant.pairwise_terms,
        "result_name": result_name,
        "result_path": result_path,
        "status": status,
        "precision": None,
        "recall": None,
        "accuracy": None,
        "f1": None,
        "dice": None,
        "iou": None,
        "tp": None,
        "fp": None,
        "fn": None,
        "tn": None,
    }


def evaluate_result(
    variant: EnergyVariant,
    result_name: str,
    result_path: str,
    gt_data,
) -> Dict[str, Any]:
    pred_data = load_data(result_path)
    metrics = calculate_all_metrics(pred_data, gt_data)
    print_metrics(result_name, metrics)
    row = empty_metric_row(variant, result_name, result_path, "ok")
    row.update(metrics)
    return row


def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "Ablate MSJI adjacent-frame energy terms. All unknown arguments are forwarded "
            "to vessel_model.sam2_main4_hybrid unchanged."
        )
    )
    parser.add_argument("--sam2_checkpoint", required=True)
    parser.add_argument("--sam2_model_cfg", required=True)
    parser.add_argument("--volume_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gt_path", required=True, help="Ground truth path (.nii/.nii.gz/.h5)")
    parser.add_argument("--dataset_key", default="main")
    parser.add_argument("--seed_file", default=None)
    parser.add_argument("--init_seg_path", default=None)
    parser.add_argument("--segmentation_module", default="vessel_model.sam2_main4_hybrid")
    parser.add_argument("--joint_init_axis_mode", default="preselect", choices=["preselect", "joint_energy"])

    parser.add_argument("--energy_suite", default="default", choices=["default", "minimal"])
    parser.add_argument(
        "--energy_term_sets",
        default=None,
        help=(
            "Optional custom semicolon-separated pairwise term sets. Example: "
            "'iou;centroid_radius;iou+centroid_radius+area_log;shape=dice+area_ratio'. "
            + PAIRWISE_TERM_HELP
        ),
    )
    parser.add_argument(
        "--omit_score_unary",
        action="store_true",
        help=(
            "By default every unweighted variant includes the same unary score term so the "
            "ablation focuses on adjacent-frame energy. Enable this to test pure pairwise terms."
        ),
    )
    parser.add_argument("--no_legacy_weighted", action="store_true", help="Do not include the original weighted energy baseline.")

    parser.add_argument("--result_prefix", default="joint_energy_ablation")
    parser.add_argument("--csv_name", default="joint_energy_ablation_metrics.csv")
    parser.add_argument("--sort_by", default="dice", choices=["precision", "recall", "accuracy", "f1", "dice", "iou"])
    parser.add_argument("--skip_existing", action="store_true", help="If output exists, skip segmentation and only evaluate.")
    parser.add_argument("--eval_only", action="store_true", help="Do not run segmentation, only evaluate existing outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running segmentation or loading GT.")
    return parser.parse_known_args()


def main():
    args, passthrough_args = get_args()
    ensure_no_controlled_passthrough(passthrough_args)
    os.makedirs(args.output_dir, exist_ok=True)

    if has_passthrough_flag(passthrough_args, "--enable_seed_judge"):
        print(
            "[WARNING] --enable_seed_judge is enabled. Energy thresholds such as "
            "--thr_best_energy were tuned for the legacy weighted scale and may not be "
            "comparable across unweighted energy variants."
        )

    variants = build_energy_variants(args)
    gt_data = None if args.dry_run else load_data(args.gt_path)
    all_results: List[Dict[str, Any]] = []

    print(f"Total energy variants: {len(variants)}")
    print("Unweighted formula: selected terms are summed directly, with no extra term weights.")
    print(PAIRWISE_TERM_HELP)

    for index, variant in enumerate(variants, start=1):
        result_name = make_result_name(args.result_prefix, index, variant)
        result_path = os.path.join(args.output_dir, result_name)
        cmd = build_child_cmd(args, passthrough_args, variant, result_name)

        print(f"\n[{index}/{len(variants)}] {result_name}")
        print(
            f"  mode={variant.mode} terms={variant.joint_energy_terms} "
            f"pairwise={variant.pairwise_terms}"
        )
        print(shlex.join(cmd))

        if args.dry_run:
            all_results.append(empty_metric_row(variant, result_name, result_path, "dry_run"))
            continue

        if not args.eval_only:
            should_run = True
            if args.skip_existing and os.path.exists(result_path):
                print(f"Output exists, skip segmentation: {result_path}")
                should_run = False
            if should_run:
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as exc:
                    print(f"[ERROR] Segmentation failed for {result_name}: {exc}")
                    all_results.append(empty_metric_row(variant, result_name, result_path, "segmentation_failed"))
                    continue

        if not os.path.exists(result_path):
            print(f"[WARNING] Result file not found, skip evaluation: {result_path}")
            all_results.append(empty_metric_row(variant, result_name, result_path, "result_not_found"))
            continue

        try:
            all_results.append(evaluate_result(variant, result_name, result_path, gt_data))
        except Exception as exc:
            print(f"[ERROR] Evaluation failed for {result_name}: {exc}")
            all_results.append(empty_metric_row(variant, result_name, result_path, f"eval_failed: {exc}"))

    valid_results = [row for row in all_results if row["status"] == "ok"]
    invalid_results = [row for row in all_results if row["status"] != "ok"]
    valid_results = sorted(valid_results, key=lambda x: x[args.sort_by], reverse=True)
    all_results_sorted = valid_results + invalid_results

    csv_path = os.path.join(args.output_dir, args.csv_name)
    save_results_to_csv(all_results_sorted, csv_path)

    print("\n========== JOINT ENERGY ABLATION RESULTS ==========")
    for row in valid_results:
        print(
            f"{row['result_name']} | mode={row['energy_mode']} | "
            f"terms={row['joint_energy_terms']} | "
            f"dice={row['dice']:.4f} iou={row['iou']:.4f}"
        )
    if invalid_results:
        print("\n========== FAILED / SKIPPED ==========")
        for row in invalid_results:
            print(f"{row['result_name']} | status={row['status']}")


if __name__ == "__main__":
    main()
