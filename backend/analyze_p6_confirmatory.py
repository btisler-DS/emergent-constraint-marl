"""Protocol 6 Confirmatory Statistical Analysis.

Preregistered hypotheses (DOI: 10.5281/zenodo.19297509):

  H1: Condition A (emergent local) > Condition C (fixed external)
      on entropy_sss_correlation (field structuring of behavior).
      Test: Mann-Whitney U, one-tailed (A > C).

  H2: Condition A < Condition D (no constraint) on final_avg_reward_A
      (emergent constraint imposes coordination overhead vs baseline).
      Test: Mann-Whitney U, one-tailed (A < D).
      Secondary: A vs C.

  H3: Var(query_rate, A) > Var(query_rate, B)
      (local field produces more behavioral heterogeneity than global).
      Test: Levene's test (two-sample), one-tailed interpretation.

  Mechanistic prediction:
      entropy_sss_correlation in A is significantly < 0
      (field entropy negatively tracks sustained structure score —
      temporal coupling signature).
      Test: one-sample Wilcoxon signed-rank vs 0, one-tailed.

  H4: behavioral_differentiation_index: std(query_rate) across seeds
      in A > B > C ≈ D.
      Descriptive comparison of per-condition query_rate std.

Output: analysis/p6_confirmatory_analysis.json
         analysis/p6_confirmatory_analysis_report.txt
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats


RESULTS_DIR = Path(__file__).parent.parent / "results" / "p6_confirmatory"
OUTPUT_DIR = Path(__file__).parent.parent / "analysis"
PREREG_DOI = "10.5281/zenodo.19297509"
ALPHA = 0.05


def load_condition(condition: str) -> list[dict]:
    records = []
    for seed in range(50):
        path = RESULTS_DIR / f"p6_confirmatory_cond{condition}_seed{seed}.json"
        with open(path) as f:
            records.append(json.load(f))
    return records


def extract(records: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in records], dtype=float)


def mann_whitney(a: np.ndarray, b: np.ndarray, alternative: str) -> dict:
    stat, p = stats.mannwhitneyu(a, b, alternative=alternative)
    n1, n2 = len(a), len(b)
    # rank-biserial correlation as effect size
    r = 1 - (2 * stat) / (n1 * n2)
    return {"U": round(float(stat), 4), "p": round(float(p), 6), "r": round(float(r), 4),
            "n1": n1, "n2": n2, "mean_a": round(float(np.mean(a)), 4),
            "mean_b": round(float(np.mean(b)), 4)}


def levene_test(a: np.ndarray, b: np.ndarray) -> dict:
    stat, p = stats.levene(a, b)
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    return {"W": round(float(stat), 4), "p": round(float(p), 6),
            "var_a": round(var_a, 6), "var_b": round(var_b, 6),
            "var_ratio": round(var_a / var_b if var_b > 0 else float("inf"), 4)}


def wilcoxon_vs_zero(a: np.ndarray, alternative: str) -> dict:
    stat, p = stats.wilcoxon(a, alternative=alternative)
    return {"W": round(float(stat), 4), "p": round(float(p), 6),
            "median": round(float(np.median(a)), 4),
            "mean": round(float(np.mean(a)), 4)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    cond = {c: load_condition(c) for c in ["A", "B", "C", "D"]}

    # Extract per-seed metric arrays
    entropy_sss = {c: extract(cond[c], "entropy_sss_correlation") for c in "ABCD"}
    reward_A    = {c: extract(cond[c], "final_avg_reward_A") for c in "ABCD"}
    query_rate  = {c: extract(cond[c], "final_query_rate") for c in "ABCD"}
    field_mean  = {c: extract(cond[c], "final_field_mean") for c in "ABCD"}

    results: dict = {
        "protocol": "P6_confirmatory",
        "preregistration_doi": PREREG_DOI,
        "alpha": ALPHA,
        "n_per_condition": 50,
    }

    # ------------------------------------------------------------------
    # H1: A > C on entropy_sss_correlation
    # (positive entropy_sss_corr = field entropy covaries with SSS;
    #  negative = high entropy suppresses SSS = field structuring behavior)
    # Preregistered direction: A has stronger (more negative) correlation
    # → Mann-Whitney one-tailed: A < C (less value = more negative)
    # ------------------------------------------------------------------
    h1 = mann_whitney(entropy_sss["A"], entropy_sss["C"], alternative="less")
    h1["hypothesis"] = "H1: entropy_sss_correlation(A) < entropy_sss_correlation(C) [A more negative = stronger field structuring]"
    h1["supported"] = h1["p"] < ALPHA
    results["H1"] = h1

    # ------------------------------------------------------------------
    # H2a: A < D on final_avg_reward_A
    # (emergent constraint imposes overhead; unconstrained D earns more)
    # ------------------------------------------------------------------
    h2a = mann_whitney(reward_A["A"], reward_A["D"], alternative="less")
    h2a["hypothesis"] = "H2a: reward_A(A) < reward_A(D) [constraint overhead vs unconstrained]"
    h2a["supported"] = h2a["p"] < ALPHA

    # H2b: A vs C on reward (secondary)
    h2b = mann_whitney(reward_A["A"], reward_A["C"], alternative="greater")
    h2b["hypothesis"] = "H2b: reward_A(A) > reward_A(C) [emergent > fixed constraint efficiency]"
    h2b["supported"] = h2b["p"] < ALPHA
    results["H2"] = {"H2a": h2a, "H2b": h2b}

    # ------------------------------------------------------------------
    # H3: Var(query_rate, A) > Var(query_rate, B)
    # Levene's test — two-sided; we report direction separately
    # ------------------------------------------------------------------
    h3 = levene_test(query_rate["A"], query_rate["B"])
    h3["hypothesis"] = "H3: Var(query_rate, A) > Var(query_rate, B) [local > global field heterogeneity]"
    h3["direction_supported"] = h3["var_a"] > h3["var_b"]
    h3["supported"] = (h3["p"] / 2 < ALPHA) and h3["direction_supported"]
    results["H3"] = h3

    # ------------------------------------------------------------------
    # Mechanistic: entropy_sss_correlation in A is significantly < 0
    # ------------------------------------------------------------------
    mech = wilcoxon_vs_zero(entropy_sss["A"], alternative="less")
    mech["hypothesis"] = "Mechanistic: entropy_sss_correlation(A) < 0 [temporal coupling: high entropy suppresses SSS]"
    mech["supported"] = mech["p"] < ALPHA
    results["mechanistic"] = mech

    # ------------------------------------------------------------------
    # H4: behavioral differentiation — query_rate std A > B > C ≈ D
    # ------------------------------------------------------------------
    h4 = {
        "hypothesis": "H4: std(query_rate) ordering: A > B > C ≈ D",
        "std_A": round(float(np.std(query_rate["A"], ddof=1)), 6),
        "std_B": round(float(np.std(query_rate["B"], ddof=1)), 6),
        "std_C": round(float(np.std(query_rate["C"], ddof=1)), 6),
        "std_D": round(float(np.std(query_rate["D"], ddof=1)), 6),
    }
    h4["A_gt_B"] = h4["std_A"] > h4["std_B"]
    h4["B_gt_C"] = h4["std_B"] > h4["std_C"]
    h4["supported"] = h4["A_gt_B"] and h4["B_gt_C"]
    results["H4"] = h4

    # ------------------------------------------------------------------
    # Descriptive statistics per condition
    # ------------------------------------------------------------------
    descriptives = {}
    for c in "ABCD":
        descriptives[c] = {
            "mean_entropy_sss_correlation": round(float(np.mean(entropy_sss[c])), 4),
            "std_entropy_sss_correlation":  round(float(np.std(entropy_sss[c], ddof=1)), 4),
            "mean_final_reward_A":          round(float(np.mean(reward_A[c])), 4),
            "std_final_reward_A":           round(float(np.std(reward_A[c], ddof=1)), 4),
            "mean_query_rate":              round(float(np.mean(query_rate[c])), 4),
            "std_query_rate":               round(float(np.std(query_rate[c], ddof=1)), 4),
            "mean_final_field_mean":        round(float(np.mean(field_mean[c])), 4),
            "field_collapsed_count":        int(sum(r["field_collapsed"] for r in cond[c])),
            "field_formed_count":           int(sum(r["field_formed"] for r in cond[c])),
        }
    results["descriptives"] = descriptives

    # ------------------------------------------------------------------
    # Write JSON
    # ------------------------------------------------------------------
    json_path = OUTPUT_DIR / "p6_confirmatory_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Analysis saved to {json_path}")

    # ------------------------------------------------------------------
    # Write plain-text report
    # ------------------------------------------------------------------
    report_path = OUTPUT_DIR / "p6_confirmatory_analysis_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        def w(line=""):
            f.write(line + "\n")

        w("=" * 70)
        w("PROTOCOL 6 CONFIRMATORY STATISTICAL ANALYSIS")
        w(f"Preregistration DOI: {PREREG_DOI}")
        w(f"Alpha: {ALPHA}  |  N per condition: 50")
        w("=" * 70)

        w()
        w("DESCRIPTIVE STATISTICS")
        w("-" * 70)
        w(f"{'Condition':<12} {'mean_reward_A':>14} {'mean_query_rate':>16} {'mean_entropy_sss':>17} {'field_collapsed':>16}")
        for c in "ABCD":
            d = descriptives[c]
            w(f"  {c:<10} {d['mean_final_reward_A']:>14.4f} {d['mean_query_rate']:>16.4f} "
              f"{d['mean_entropy_sss_correlation']:>17.4f} {d['field_collapsed_count']:>16}/50")

        w()
        w("HYPOTHESIS TESTS")
        w("-" * 70)

        def report_mw(label, res):
            sig = "* SUPPORTED" if res["supported"] else "  not supported"
            w(f"{label}")
            w(f"  {res['hypothesis']}")
            w(f"  Mann-Whitney U={res['U']}, p={res['p']:.6f}, r={res['r']:.4f}  {sig}")
            w(f"  mean(A)={res['mean_a']:.4f}  mean(B)={res['mean_b']:.4f}")

        w()
        report_mw("H1", h1)
        w()
        report_mw("H2a", h2a)
        w()
        report_mw("H2b", h2b)

        w()
        h3r = results["H3"]
        sig3 = "* SUPPORTED" if h3r["supported"] else "  not supported"
        w("H3")
        w(f"  {h3r['hypothesis']}")
        w(f"  Levene W={h3r['W']}, p={h3r['p']:.6f}  {sig3}")
        w(f"  Var(A)={h3r['var_a']:.6f}  Var(B)={h3r['var_b']:.6f}  ratio={h3r['var_ratio']:.4f}")

        w()
        mr = results["mechanistic"]
        sigm = "* SUPPORTED" if mr["supported"] else "  not supported"
        w("MECHANISTIC PREDICTION")
        w(f"  {mr['hypothesis']}")
        w(f"  Wilcoxon W={mr['W']}, p={mr['p']:.6f}  {sigm}")
        w(f"  median={mr['median']:.4f}  mean={mr['mean']:.4f}")

        w()
        h4r = results["H4"]
        sigh4 = "* SUPPORTED" if h4r["supported"] else "  not supported"
        w("H4 (BEHAVIORAL DIFFERENTIATION)")
        w(f"  H4: std(query_rate) ordering: A > B > C ~ D")
        w(f"  std_A={h4r['std_A']:.6f}  std_B={h4r['std_B']:.6f}  std_C={h4r['std_C']:.6f}  std_D={h4r['std_D']:.6f}")
        w(f"  A>B: {h4r['A_gt_B']}  B>C: {h4r['B_gt_C']}  {sigh4}")

        w()
        w("=" * 70)
        supported = sum([
            h1["supported"], h2a["supported"], h2b["supported"],
            h3["supported"], mr["supported"], h4r["supported"]
        ])
        w(f"SUMMARY: {supported}/6 tests supported at alpha={ALPHA}")
        w("=" * 70)

    print(f"Report saved to {report_path}")

    # Print report to stdout
    with open(report_path, encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
