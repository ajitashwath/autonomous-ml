

from __future__ import annotations

import argparse
import time
from typing import Literal

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

DriftLevel = Literal["none", "mild", "severe"]


def _apply_no_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    return df.copy()


def _apply_mild_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if "MonthlyCharges" in out.columns:
        out["MonthlyCharges"] = out["MonthlyCharges"] * 1.5
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = out["TotalCharges"] * 1.4
    if "Contract" in out.columns:
        flip_mask = rng.random(size=len(out)) < 0.3
        out.loc[flip_mask, "Contract"] = "Month-to-month"
    return out


def _apply_severe_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()
    if "MonthlyCharges" in out.columns:
        out["MonthlyCharges"] = (out["MonthlyCharges"] * 3.0).clip(upper=199.0)
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = out["TotalCharges"] * 3.0
    if "Contract" in out.columns:
        out["Contract"] = "Two year"
    if "tenure" in out.columns:
        out["tenure"] = out["tenure"].apply(lambda x: max(1, int(x * 0.1)))
    return out


_DRIFT_FN = {
    "none": _apply_no_drift,
    "mild": _apply_mild_drift,
    "severe": _apply_severe_drift,
}



def apply_drift(df: pd.DataFrame, level: DriftLevel, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fn = _DRIFT_FN.get(level, _apply_no_drift)
    return fn(df, rng)


def simulate_drift(
    num_requests: int = 500,
    drift_level: DriftLevel = "severe",
    reference_path: str = "data/reference/reference.parquet",
    seed: int = 42,
    dry_run: bool = False,
) -> None:
    settings = get_settings()
    api_url = f"http://{settings.inference_api_host}:{settings.inference_api_port}/api/v1/predict"

    logger.info(
        "loading_reference_data_for_simulation",
        path=reference_path,
        drift_level=drift_level,
    )

    try:
        df = pd.read_parquet(reference_path)
    except FileNotFoundError:
        logger.error("reference_data_not_found_run_training_first", path=reference_path)
        return

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(df), size=num_requests)
    sample_df = df.iloc[indices].copy().reset_index(drop=True)

    if "Churn" in sample_df.columns:
        sample_df = sample_df.drop(columns=["Churn"])

    perturbed_df = apply_drift(sample_df, level=drift_level, seed=seed)

    logger.info(
        "drift_injected",
        drift_level=drift_level,
        n_rows=len(perturbed_df),
    )

    if dry_run:
        logger.info("dry_run_mode_printing_first_3_rows")
        print("\n[DRY RUN] First 3 rows that would be sent:")
        print(perturbed_df.head(3).to_string())
        print(f"\nTarget API: {api_url}")
        print(f"Total rows: {len(perturbed_df)}")
        return

    records = perturbed_df.to_dict(orient="records")
    logger.info("sending_drifted_requests_to_inference_api", count=len(records), url=api_url)

    successes = 0
    failures = 0
    error_examples: list[str] = []

    with httpx.Client(timeout=10.0) as client:
        for row in tqdm(records, desc=f"Sending [{drift_level} drift]"):
            try:
                resp = client.post(api_url, json=row)
                if resp.status_code == 200:
                    successes += 1
                else:
                    failures += 1
                    if len(error_examples) < 3:
                        error_examples.append(f"HTTP {resp.status_code}: {resp.text[:100]}")
            except Exception as exc:
                failures += 1
                if len(error_examples) < 3:
                    error_examples.append(str(exc))

            time.sleep(0.01)

    logger.info(
        "simulation_complete",
        drift_level=drift_level,
        successes=successes,
        failures=failures,
        error_examples=error_examples,
        next_steps="Wait for drift_detector to read these logs and trigger Airflow.",
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate production data drift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="",
    )
    parser.add_argument(
        "--drift-level",
        choices=["none", "mild", "severe"],
        default="severe",
        help="Intensity of drift to inject (default: severe)",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=500,
        help="Number of requests to send (default: 500)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--reference-path",
        type=str,
        default="data/reference/reference.parquet",
        help="Path to reference parquet (default: data/reference/reference.parquet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be sent without hitting the API",
    )
    args = parser.parse_args()

    simulate_drift(
        num_requests=args.n_rows,
        drift_level=args.drift_level,
        reference_path=args.reference_path,
        seed=args.seed,
        dry_run=args.dry_run,
    )
