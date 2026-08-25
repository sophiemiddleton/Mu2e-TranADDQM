#!/usr/bin/env python3
"""
DQM CRV Anomaly Dataset Matrix Generator
Process isolation via subprocess prevents ROOT C++ segfaults and pickling errors.
"""

import sys
import csv
import subprocess
from collections import defaultdict

SOURCE_STR = "crvreco,kpp_crv,file,004-000"
OUTPUT_FILE = "dqm_crv_anomaly_dataset_004-000.csv"


def query_dqm_metric(source_str: str, metric_path: str) -> str:
    """Executes a single DQM query in an isolated Python subprocess."""
    py_code = f"""
import DQM
tool = DQM.DqmTool()
tool.init()
tool.printNumbers("numbers", False, "{source_str}", "{metric_path}", True)
print(tool.getResult() or "")
"""
    res = subprocess.run([sys.executable, "-c", py_code], capture_output=True, text=True)
    return res.stdout if res.returncode == 0 else ""


def get_all_metrics() -> list[tuple[str, str, str]]:
    """Retrieves all registered CRV metrics via a subprocess."""
    py_code = """
import DQM
tool = DQM.DqmTool()
tool.init()
tool.printValues()
print(tool.getResult() or "")
"""
    res = subprocess.run([sys.executable, "-c", py_code], capture_output=True, text=True)
    raw_values = res.stdout if res.returncode == 0 else ""

    metrics = []
    for line in raw_values.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4 and parts[1] == "CRV":
            v_id = parts[0]
            metric_path = f"{parts[1]},{parts[2]},{parts[3]}"
            feat_name = f"{parts[1]}_{parts[2]}_{parts[3]}"
            metrics.append((v_id, metric_path, feat_name))

    return metrics


def main():
    print("Step 1: Fetching CRV metrics...")
    metrics = get_all_metrics()
    if not metrics:
        print("Error: Could not retrieve metrics list.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(metrics)} CRV metrics.")

    matrix = defaultdict(dict)
    valid_features = []

    print(f"\nStep 2: Harvesting metrics for source '{SOURCE_STR}'...")

    for v_id, metric_path, feat_name in metrics:
        print(f"  [{v_id:>3}] Querying {feat_name:<45}", end="", flush=True)

        raw_output = query_dqm_metric(SOURCE_STR, metric_path)

        if not raw_output.strip():
            print(" → [0 rows]")
            continue

        count = 0
        lines = raw_output.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue

            parts = [p.strip() for p in line.split(",")]

            # Column 1 = Value, Column 15 = Run, Column 16 = Subrun, Column 19 = Timestamp
            if len(parts) < 17:
                continue

            try:
                val = parts[1]
                run = int(parts[15])
                subrun = int(parts[16])
                timestamp = parts[19] if len(parts) >= 20 else ""

                key = (run, subrun)
                if "run" not in matrix[key]:
                    matrix[key]["run"] = str(run)
                    matrix[key]["subrun"] = str(subrun)
                    matrix[key]["timestamp"] = timestamp

                matrix[key][feat_name] = val
                count += 1
            except ValueError:
                continue

        if count > 0:
            print(f" → ✓ {count} rows")
            valid_features.append(feat_name)
        else:
            print(" → [0 rows]")

    # Step 3: Write Matrix CSV
    if matrix and valid_features:
        print(f"\nStep 3: Saving matrix ({len(matrix)} states, {len(valid_features)} features)...")
        fieldnames = ["run", "subrun", "timestamp"] + valid_features

        with open(OUTPUT_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            for key in sorted(matrix.keys()):
                row_dict = matrix[key]
                for feat in valid_features:
                    row_dict.setdefault(feat, "")
                writer.writerow(row_dict)

        print(f"✓ Anomaly dataset saved to {OUTPUT_FILE}")
    else:
        print("\n✗ No matrix data collected.", file=sys.stderr)


if __name__ == "__main__":
    main()
