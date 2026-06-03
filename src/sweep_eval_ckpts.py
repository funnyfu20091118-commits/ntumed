#!/usr/bin/env python3
"""Sweep stage-3 evaluation over multiple U-ViT checkpoints."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class EvalResult:
    ckpt: str
    fid: float | None
    auroc_avg: float | None
    json_path: str


def _epoch_key(path: str) -> int:
    m = re.search(r"epoch(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep checkpoints and rank by eval metrics.")
    p.add_argument("--ckpt-glob", default="checkpoints/uvit_epoch*.pt",
                   help="Glob pattern for checkpoint files")
    p.add_argument("--ckpt-list", nargs="*", default=None,
                   help="Explicit checkpoint paths (overrides --ckpt-glob)")
    p.add_argument("--metric", choices=["fid", "auroc", "all"], default="all")
    p.add_argument("--fid-num-samples", type=int, default=512,
                   help="Sample count per checkpoint (smaller is faster)")
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--guidance", type=float, default=None)
    p.add_argument("--out-dir", default="outputs/ckpt_sweep")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--python", default=sys.executable,
                   help="Python executable used for subprocess eval calls")
    return p.parse_args()


def collect_ckpts(args: argparse.Namespace) -> list[str]:
    if args.ckpt_list:
        ckpts = args.ckpt_list
    else:
        ckpts = glob.glob(args.ckpt_glob)

    ckpts = [c for c in ckpts if os.path.isfile(c)]
    ckpts = sorted(ckpts, key=_epoch_key)
    return ckpts


def run_eval(args: argparse.Namespace, ckpt_path: str, out_dir: str) -> EvalResult:
    ckpt_name = os.path.basename(ckpt_path).replace(".pt", "")
    result_json = os.path.join(out_dir, f"{ckpt_name}.json")

    cmd = [
        args.python,
        "src/evaluate.py",
        "--metric", args.metric,
        "--ckpt", os.path.basename(ckpt_path),
        "--fid-num-samples", str(args.fid_num_samples),
        "--eval-batch-size", str(args.eval_batch_size),
        "--disable-wandb",
        "--json-out", result_json,
    ]
    if args.guidance is not None:
        cmd.extend(["--guidance", str(args.guidance)])

    print(f"\n=== Evaluating {ckpt_path} ===")
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    with open(result_json, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {result_json}: {exc}") from exc

    return EvalResult(
        ckpt=ckpt_path,
        fid=payload.get("fid"),
        auroc_avg=payload.get("auroc_avg"),
        json_path=result_json,
    )


def write_summary(results: list[EvalResult], out_dir: str, top_k: int) -> str:
    summary_path = os.path.join(out_dir, "summary.json")

    rows = []
    for r in results:
        score = None
        if r.fid is not None and r.auroc_avg is not None:
            score = r.auroc_avg - 0.01 * r.fid
        elif r.auroc_avg is not None:
            score = r.auroc_avg
        elif r.fid is not None:
            score = -r.fid
        rows.append({
            "ckpt": r.ckpt,
            "fid": r.fid,
            "auroc_avg": r.auroc_avg,
            "score": score,
            "json_path": r.json_path,
        })

    ranked = sorted(rows, key=lambda x: (float("-inf") if x["score"] is None else x["score"]), reverse=True)
    payload = {
        "ranking_note": "score = auroc_avg - 0.01*fid when both exist; otherwise auroc_avg (higher better) or -fid (lower fid better)",
        "top_k": top_k,
        "top": ranked[:top_k],
        "all": rows,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n=== Top checkpoints ===")
    for i, row in enumerate(payload["top"], 1):
        print(
            f"{i:2d}. {os.path.basename(row['ckpt'])} | "
            f"FID={row['fid']} | AUROC(avg)={row['auroc_avg']} | score={row['score']}"
        )
    print(f"\nSaved sweep summary: {summary_path}")
    return summary_path


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = collect_ckpts(args)
    if not ckpts:
        print("No checkpoints found.")
        return 1

    print(f"Found {len(ckpts)} checkpoints to evaluate.")
    results = []
    for ckpt in ckpts:
        try:
            results.append(run_eval(args, ckpt, args.out_dir))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            print(f"Evaluation failed for {ckpt}: {exc}")

    if not results:
        print("All checkpoint evaluations failed.")
        return 2

    write_summary(results, args.out_dir, args.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
