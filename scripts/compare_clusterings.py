"""Compare the saved k-means partitions of the 28 stations.

For each pair of clusterings, the labels of the second are aligned to the
first with the permutation that maximises their agreement (Hungarian
algorithm on the confusion matrix). The script reports the fraction of
stations with the same aligned label ("accuracy") and the Adjusted Rand
Index. It reads only the label files in clustering/output/ and writes
results/cluster_comparison.csv.

Run from the repository root:  python scripts/compare_clusterings.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "clustering" / "output"

# Label files: the 2020 partition is the same with and without scaling.
FILES = {
    ("2020", "scaled"): "cluster2020_output.csv",
    ("2020", "unscaled"): "cluster2020_output.csv",
}
for year in ("2035", "2055"):
    for ssp in ("ssp126", "ssp585"):
        FILES[(f"{year} {ssp}", "scaled")] = f"cluster{year}{ssp}_t_output.csv"
        FILES[(f"{year} {ssp}", "unscaled")] = f"cluster{year}{ssp}_output.csv"

PAIRS = [
    ("2020", "2035 ssp126"), ("2020", "2035 ssp585"),
    ("2020", "2055 ssp126"), ("2020", "2055 ssp585"),
    ("2035 ssp126", "2055 ssp126"), ("2035 ssp585", "2055 ssp585"),
    ("2035 ssp126", "2035 ssp585"), ("2055 ssp126", "2055 ssp585"),
]


def read_labels(name: str) -> pd.Series:
    df = pd.read_csv(OUT / name, sep=";", encoding="utf-8-sig", index_col=0)
    return df.iloc[:, 0]


def aligned_accuracy(a: np.ndarray, b: np.ndarray) -> float:
    cm = confusion_matrix(a, b)
    rows, cols = linear_sum_assignment(cm, maximize=True)
    mapping = {c: r for r, c in zip(rows, cols)}
    return float(np.mean(a == np.array([mapping[x] for x in b])))


def main() -> None:
    records = []
    for first, second in PAIRS:
        row = {"first": first, "second": second}
        for scaling in ("scaled", "unscaled"):
            a = read_labels(FILES[(first, scaling)])
            b = read_labels(FILES[(second, scaling)]).reindex(a.index)
            row[f"accuracy_{scaling}"] = round(aligned_accuracy(a.values, b.values), 4)
            row[f"ari_{scaling}"] = round(adjusted_rand_score(a.values, b.values), 3)
        records.append(row)
    table = pd.DataFrame(records)
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "cluster_comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
