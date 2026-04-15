"""
Data preprocessing: build dataset manifest linking images ↔ reports.
Filters for AP/PA views, removes duplicates, and creates train/val/test splits.
"""
import os
import re
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from config import Config


def parse_report(report_path: str) -> str:
    """Extract FINDINGS + IMPRESSION from a MIMIC-CXR report file."""
    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    sections = {}
    current_section = None
    current_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # Section headers are all-caps followed by colon
        if re.match(r"^[A-Z][A-Z /\-]+:?\s*$", stripped):
            if current_section is not None:
                sections[current_section] = " ".join(current_lines).strip()
            current_section = stripped.rstrip(":").strip()
            current_lines = []
        else:
            current_lines.append(stripped)
    if current_section is not None:
        sections[current_section] = " ".join(current_lines).strip()

    # Prefer FINDINGS + IMPRESSION; fall back to whatever is available
    parts = []
    for key in ["FINDINGS", "IMPRESSION"]:
        if key in sections and sections[key]:
            parts.append(sections[key])
    if not parts:
        # Use the full text if no standard sections found
        parts = [" ".join(text.split()).strip()]

    report_text = " ".join(parts)
    # Clean up whitespace / de-id tokens
    report_text = re.sub(r"\s+", " ", report_text).strip()
    report_text = re.sub(r"_{2,}", "", report_text)  # remove ___ de-id blanks
    return report_text


def build_manifest(cfg: Config):
    """Build dataset manifest: list of (dicom_id, subject_id, study_id, split, image_path, report_text)."""
    print("Loading metadata...")
    meta_df = pd.read_csv(cfg.metadata_csv)
    split_df = pd.read_csv(cfg.split_csv)
    chexpert_df = pd.read_csv(cfg.chexpert_csv)

    # Merge metadata with split info
    df = meta_df.merge(split_df[["dicom_id", "split"]], on="dicom_id", how="inner")

    # Filter AP / PA views only
    df = df[df["ViewPosition"].isin(["AP", "PA"])].copy()
    print(f"After AP/PA filter: {len(df)} rows")

    # Build image paths: files/pXX/pXXXXXXXX/sXXXXXXXX/dicom_id.jpg
    def make_img_path(row):
        sid = str(row["subject_id"])
        prefix = f"p{sid[:2]}"
        patient = f"p{sid}"
        study = f"s{row['study_id']}"
        return os.path.join(cfg.image_root, prefix, patient, study, f"{row['dicom_id']}.jpg")

    df["image_path"] = df.apply(make_img_path, axis=1)

    # Build report paths
    def make_report_path(row):
        sid = str(row["subject_id"])
        prefix = f"p{sid[:2]}"
        patient = f"p{sid}"
        study = f"s{row['study_id']}"
        return os.path.join(cfg.report_root, prefix, patient, f"{study}.txt")

    df["report_path"] = df.apply(make_report_path, axis=1)

    # Filter rows where both image and report exist
    print("Building existing image index...")
    import subprocess
    # Pre-build image list to a temp file (handles large output better)
    img_list_file = os.path.join(cfg.cache_dir, "img_list.txt")
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.system(f"find {cfg.image_root} -name '*.jpg' -maxdepth 5 > {img_list_file} 2>/dev/null")
    with open(img_list_file) as f:
        existing_images = set(line.strip() for line in f if line.strip())
    print(f"  Found {len(existing_images)} images on disk")

    # Reports: we know they all exist (extracted from zip), just check study_id mapping
    # Use the study list CSV to know which reports exist
    study_df = pd.read_csv(cfg.study_csv)
    existing_studies = set(zip(study_df["subject_id"], study_df["study_id"]))
    print(f"  Found {len(existing_studies)} study records")

    # Filter: image must exist on disk, report study must exist
    exists_mask = (
        df["image_path"].isin(existing_images)
        & df.apply(lambda r: (r["subject_id"], r["study_id"]) in existing_studies, axis=1)
    )
    df = df[exists_mask].copy()
    print(f"After existence check: {len(df)} rows")

    # Remove duplicates: keep one image per (subject_id, study_id) — prefer PA over AP
    df["view_priority"] = df["ViewPosition"].map({"PA": 0, "AP": 1})
    df = df.sort_values("view_priority").drop_duplicates(
        subset=["subject_id", "study_id"], keep="first"
    )
    df = df.drop(columns=["view_priority"])
    print(f"After dedup: {len(df)} rows")

    # Parse reports
    print("Parsing reports...")
    report_cache = {}
    reports = []
    for rp in tqdm(df["report_path"].values, desc="Parsing reports"):
        if rp not in report_cache:
            report_cache[rp] = parse_report(rp)
        reports.append(report_cache[rp])
    df["report_text"] = reports

    # Drop rows with empty reports
    df = df[df["report_text"].str.len() > 10].copy()
    print(f"After empty report filter: {len(df)} rows")

    # Merge chexpert labels
    df = df.merge(chexpert_df, on=["subject_id", "study_id"], how="left")

    # Summary
    for split_name in ["train", "validate", "test"]:
        n = len(df[df["split"] == split_name])
        print(f"  {split_name}: {n}")

    # Save manifest
    os.makedirs(cfg.cache_dir, exist_ok=True)
    manifest_path = os.path.join(cfg.cache_dir, "manifest.csv")
    cols = ["dicom_id", "subject_id", "study_id", "split", "image_path",
            "report_path", "report_text", "ViewPosition"]
    # Add chexpert label columns
    label_cols = [c for c in chexpert_df.columns if c not in ["subject_id", "study_id"]]
    cols += label_cols
    df[cols].to_csv(manifest_path, index=False)
    print(f"Manifest saved to {manifest_path}")
    return df


if __name__ == "__main__":
    cfg = Config()
    build_manifest(cfg)
