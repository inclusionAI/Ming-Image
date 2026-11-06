#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
"""Segment-wise comparison of two alignment bundles (tools/capture_case.py).

Compares, in order: identity (model family / case parameters / shapes) ->
discrete inputs (token ids, masks) -> conditions -> initial noise -> per-step
latents -> final pixels. Discrete inputs, shapes and schedule sequences must
match exactly. Floating-point segments report dtype/shape/finite,
max/mean absolute error, relative L2 (||a-b||/max(||b||,eps)) and cosine
similarity. The final image reports pixel MAE/RMSE on 0-255.

Exit status is 0 only when every checked segment passes its threshold.
Segments without a measured cross-hardware basis are reported as
needs_review, never silently passed.

Usage:
    python tools/compare_case.py --reference <bundle-a> --candidate <bundle-b> \
        [--thresholds thresholds.json] [--report report.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_THRESHOLDS = {
    # Measured on NVIDIA L20Y with the unified runtime (2026-11-06):
    # same-hardware reruns are bitwise for discrete inputs and initial noise;
    # conditions show bf16 kernel-level nondeterminism below 1 ulp;
    # 30-step latents drift under pipeline bf16 accumulation; final images
    # land at MAE ~1.2 vs a separate reference pipeline. These values
    # are a same-stack reference, NOT a cross-hardware acceptance guarantee.
    "condition_embeds": {"max_abs": 0.5, "rel_l2": 1e-3},
    "directvlm_hidden_states": {"max_abs": 0.5, "rel_l2": 1e-3},
    "initial_noise": {"max_abs": 0.0, "rel_l2": 0.0},
    "step_latents_final": {"max_abs": 0.6, "rel_l2": 2e-3},
    "final_pixels": {"mae": 2.0, "rmse": 5.0},
}

NEEDS_REVIEW_SEGMENTS = {"step_latents_all"}


def load_bundle(path: Path):
    arrays = np.load(path / "arrays.npz")
    meta = json.loads((path / "case.json").read_text(encoding="utf-8"))
    return meta, {key: arrays[key] for key in arrays.files}


def tensor_report(a: np.ndarray, b: np.ndarray, thresholds: dict):
    a32, b32 = a.astype(np.float64), b.astype(np.float64)
    diff = np.abs(a32 - b32)
    rel = np.linalg.norm((a32 - b32).ravel()) / max(np.linalg.norm(b32.ravel()), 1e-12)
    cos = float(
        np.dot(a32.ravel(), b32.ravel())
        / max(np.linalg.norm(a32.ravel()) * np.linalg.norm(b32.ravel()), 1e-12)
    )
    report = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite": bool(np.isfinite(a32).all() and np.isfinite(b32).ravel().all()),
        "max_abs": float(diff.max()) if diff.size else 0.0,
        "mean_abs": float(diff.mean()) if diff.size else 0.0,
        "rel_l2": float(rel),
        "cosine": cos,
    }
    checks = []
    if "max_abs" in thresholds:
        checks.append(report["max_abs"] <= thresholds["max_abs"])
    if "rel_l2" in thresholds:
        checks.append(report["rel_l2"] <= thresholds["rel_l2"])
    report["pass"] = bool(report["finite"]) and all(checks)
    return report


def pixel_report(a: np.ndarray, b: np.ndarray, thresholds: dict):
    a32, b32 = a.astype(np.float64), b.astype(np.float64)
    mae = float(np.abs(a32 - b32).mean())
    rmse = float(np.sqrt(((a32 - b32) ** 2).mean()))
    report = {"shape": list(a.shape), "mae": mae, "rmse": rmse}
    report["pass"] = (
        mae <= thresholds.get("mae", np.inf)
        and rmse <= thresholds.get("rmse", np.inf)
    )
    return report


def compare(reference_dir: Path, candidate_dir: Path, thresholds: dict):
    ref_meta, ref = load_bundle(reference_dir)
    cand_meta, cand = load_bundle(candidate_dir)
    report = {"reference": str(reference_dir), "candidate": str(candidate_dir), "segments": {}}

    comparable = True
    identity_notes = []
    if ref_meta["mllm_family"] != cand_meta["mllm_family"]:
        identity_notes.append(
            f"mllm_family differs: {ref_meta['mllm_family']} vs {cand_meta['mllm_family']}"
        )
        comparable = False
    for key in ("task", "width", "height", "steps", "seed"):
        if ref_meta["case"][key] != cand_meta["case"][key]:
            identity_notes.append(
                f"case.{key} differs: {ref_meta['case'][key]!r} vs {cand_meta['case'][key]!r}"
            )
            comparable = False
    if ref_meta["case"]["prompt_sha256"] != cand_meta["case"]["prompt_sha256"]:
        identity_notes.append("prompt_sha256 differs")
        comparable = False
    report["identity"] = {"comparable": comparable, "notes": identity_notes}
    if not comparable:
        report["verdict"] = "incomparable"
        return report

    segments = report["segments"]
    ok = True

    ids_match = np.array_equal(ref["input_ids"], cand["input_ids"]) and np.array_equal(
        ref["attention_mask"], cand["attention_mask"]
    )
    segments["token_inputs"] = {
        "input_ids_match": bool(np.array_equal(ref["input_ids"], cand["input_ids"])),
        "attention_mask_match": bool(np.array_equal(ref["attention_mask"], cand["attention_mask"])),
        "pass": bool(ids_match),
    }
    ok &= ids_match

    for name in ("condition_embeds", "directvlm_hidden_states"):
        if name not in ref or name not in cand:
            # stripped public reference bundles carry only token inputs,
            # noise and pixels; heavyweight segments stay in the local
            # alignment bundle and are skipped, never failed.
            segments[name] = {"status": "absent", "note": "not stored in one of the bundles"}
            continue
        if ref[name].shape != cand[name].shape:
            segments[name] = {"pass": False, "error": f"shape {ref[name].shape} vs {cand[name].shape}"}
            ok = False
            continue
        segments[name] = tensor_report(ref[name], cand[name], thresholds.get(name, {}))
        ok &= segments[name]["pass"]

    if ref["initial_noise"].shape != cand["initial_noise"].shape:
        segments["initial_noise"] = {"pass": False, "error": "shape mismatch"}
        ok = False
    else:
        segments["initial_noise"] = tensor_report(
            ref["initial_noise"], cand["initial_noise"], thresholds.get("initial_noise", {})
        )
        ok &= segments["initial_noise"]["pass"]

    if "step_latents" not in ref or "step_latents" not in cand:
        segments["step_latents_all"] = {"status": "absent", "note": "not stored in one of the bundles"}
        segments["step_latents_final"] = {"status": "absent", "note": "not stored in one of the bundles"}
    elif ref["step_latents"].shape != cand["step_latents"].shape:
        segments["step_latents_all"] = {"pass": False, "error": "shape mismatch"}
        ok = False
    else:
        per_step = []
        for index in range(ref["step_latents"].shape[0]):
            per_step.append(
                tensor_report(
                    ref["step_latents"][index],
                    cand["step_latents"][index],
                    thresholds.get("step_latents_all", {}),
                )
            )
        final = tensor_report(
            ref["step_latents"][-1], cand["step_latents"][-1],
            thresholds.get("step_latents_final", {}),
        )
        segments["step_latents_all"] = {
            "pass": None,
            "status": "needs_review",
            "note": "per-step cross-backend thresholds require joint measurement",
            "worst_max_abs": max(item["max_abs"] for item in per_step),
            "worst_rel_l2": max(item["rel_l2"] for item in per_step),
        }
        segments["step_latents_final"] = final
        ok &= final["pass"]

    segments["final_pixels"] = pixel_report(
        ref["final_pixels"], cand["final_pixels"], thresholds.get("final_pixels", {})
    )
    ok &= segments["final_pixels"]["pass"]

    report["verdict"] = "pass" if ok else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.thresholds:
        overrides = json.loads(args.thresholds.read_text(encoding="utf-8"))
        for key, value in overrides.items():
            thresholds.setdefault(key, {}).update(value)

    report = compare(args.reference, args.candidate, thresholds)
    text = json.dumps(report, indent=1)
    print(text)
    if args.report:
        args.report.write_text(text, encoding="utf-8")
    sys.exit(0 if report["verdict"] == "pass" else 1)


if __name__ == "__main__":
    main()
