"""
drift_simulator/simulate.py

CLI tool to inject synthetic drift into the live system.

Process:
  1. Load reference data (healthy baseline).
  2. Perturb specific critical columns (e.g. triple the MonthlyCharges).
  3. Send requests to the Inference API (/predict).
  4. The data_logger writes these to Postgres.
  5. The drift_detector picks them up on its next run and fires an alert.

Usage:
  python -m src.drift_simulator.simulate --num-requests 500
"""

import argparse
import time

import httpx
import pandas as pd
from tqdm import tqdm

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)


def simulate_drift(num_requests: int = 500, reference_path: str = "data/reference/reference.parquet") -> None:
    settings = get_settings()
    api_url = f"http://{settings.inference_api_host}:{settings.inference_api_port}/api/v1/predict"

    logger.info("loading_reference_data_for_simulation", path=reference_path)
    try:
        df = pd.read_parquet(reference_path)
    except FileNotFoundError:
        logger.error("reference_data_not_found_run_training_first", path=reference_path)
        return

    # Sample rows
    sample_df = df.sample(n=min(num_requests, len(df)), replace=True).copy()

    # ── Inject Drift ───────────────────────────────────────────────────────────
    logger.info("injecting_synthetic_drift")

    # 1. Tripling numerical values (massive obvious drift)
    sample_df["MonthlyCharges"] = sample_df["MonthlyCharges"] * 3.0
    sample_df["TotalCharges"] = sample_df["TotalCharges"] * 3.0

    # 2. Flipping categorical distributions
    sample_df["Contract"] = "Two year"  # Everyone is now on a two-year contract

    records = sample_df.to_dict(orient="records")

    logger.info("sending_drifted_requests_to_inference_api", count=len(records), url=api_url)

    successes = 0
    failures = 0

    with httpx.Client(timeout=10.0) as client:
        # We don't use asyncio here because we want to mimic a steady stream of traffic
        for row in tqdm(records, desc="Sending requests"):
            try:
                # Remove target column if it exists in reference
                if "Churn" in row:
                    del row["Churn"]

                resp = client.post(api_url, json=row)
                if resp.status_code == 200:
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1

            # Sleep slightly so timestamps disperse
            time.sleep(0.01)

    logger.info(
        "simulation_complete",
        successes=successes,
        failures=failures,
        next_steps="Wait 5 minutes for drift_detector to read these logs and trigger Airflow.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate production data drift")
    parser.add_argument(
        "--num-requests",
        type=int,
        default=500,
        help="Number of drifted requests to send constraint (default: 500)",
    )
    args = parser.parse_args()
    simulate_drift(num_requests=args.num_requests)
