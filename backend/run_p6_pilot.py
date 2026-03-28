"""Protocol 6 Pilot Run Script.

Runs the 6-combination x 10-seed parameter sweep (60 total runs, 200 epochs each).
Produces: confirmatory_run_p6_pilot.log and p6_pilot_summary.json

Usage:
  cd backend
  python run_p6_pilot.py [--config ../config_p6_pilot.yaml]

Output files are written to the project root (parent of backend/).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import yaml

from simulation.p6_engine import P6Config, P6SimulationEngine
from simulation.metrics.field_diagnostics import compute_run_summary


def setup_logging(log_path: str) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_one(
    diffusion_coefficient: float,
    decay_rate: float,
    seed: int,
    base_config: dict,
) -> dict:
    """Run one combination/seed pair. Returns result dict."""
    config = P6Config(
        seed=seed,
        num_epochs=base_config["num_epochs"],
        episodes_per_epoch=base_config["episodes_per_epoch"],
        grid_size=base_config["grid_size"],
        num_obstacles=base_config["num_obstacles"],
        z_layers=base_config["z_layers"],
        max_steps=base_config["max_steps"],
        energy_budget=base_config["energy_budget"],
        move_cost=base_config["move_cost"],
        collision_penalty=base_config["collision_penalty"],
        signal_dim=base_config["signal_dim"],
        hidden_dim=base_config["hidden_dim"],
        depth=base_config["depth"],
        learning_rate=base_config["learning_rate"],
        gamma=base_config["gamma"],
        communication_tax_rate=base_config["communication_tax_rate"],
        survival_bonus=base_config["survival_bonus"],
        declare_cost=base_config["declare_cost"],
        query_cost=base_config["query_cost"],
        respond_cost=base_config["respond_cost"],
        signal_cost_sensitivity=base_config["signal_cost_sensitivity"],
        diffusion_coefficient=diffusion_coefficient,
        decay_rate=decay_rate,
        device=base_config.get("device", "cpu"),
    )

    engine = P6SimulationEngine(config)
    t0 = time.time()
    epoch_series = engine.run()
    elapsed = time.time() - t0

    run_summary = compute_run_summary(epoch_series)

    # Correlation between field_entropy and sustained_structure_score across epochs
    entropies = [e["field_entropy"] for e in epoch_series]
    sss_scores = [e["sustained_structure_score"] for e in epoch_series]
    if len(set(entropies)) > 1 and len(set(sss_scores)) > 1:
        corr = float(np.corrcoef(entropies, sss_scores)[0, 1])
    else:
        corr = 0.0

    return {
        "diffusion_coefficient": diffusion_coefficient,
        "decay_rate": decay_rate,
        "seed": seed,
        "elapsed_s": round(elapsed, 1),
        "field_formed": run_summary["field_formed"],
        "field_saturated": run_summary["field_saturated"],
        "field_collapsed": run_summary["field_collapsed"],
        "entropy_sss_correlation": round(corr, 4),
        "final_field_mean": round(epoch_series[-1]["field_mean"], 6),
        "final_field_std": round(epoch_series[-1]["field_std"], 6),
        "final_field_max": round(epoch_series[-1]["field_max"], 6),
        "epoch_series": epoch_series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Protocol 6 pilot parameter sweep")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent.parent / "config_p6_pilot.yaml"),
        help="Path to config_p6_pilot.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    _EXPECTED_DC = {0.1, 0.5, 0.9}
    _EXPECTED_DR = {0.05, 0.3}
    loaded_dc = set(cfg["sweep"]["diffusion_coefficient"])
    loaded_dr = set(cfg["sweep"]["decay_rate"])
    if loaded_dc != _EXPECTED_DC:
        raise ValueError(f"Config diffusion_coefficient {loaded_dc} != expected {_EXPECTED_DC}")
    if loaded_dr != _EXPECTED_DR:
        raise ValueError(f"Config decay_rate {loaded_dr} != expected {_EXPECTED_DR}")
    if len(cfg["seeds"]) != 10:
        raise ValueError(f"Expected 10 seeds, got {len(cfg['seeds'])}")

    root = Path(__file__).parent.parent
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)
    log_path = str(root / cfg["output_log"])
    json_path = str(root / cfg["output_json"])

    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    logger.info(
        "Config audit | signal_cost_sensitivity=%.1f interaction_radius=%s "
        "signal_weights declare=%.1f query=%.1f respond=%.1f",
        cfg["signal_cost_sensitivity"],
        cfg["interaction_radius"],
        cfg["signal_weights"]["declare"],
        cfg["signal_weights"]["query"],
        cfg["signal_weights"]["respond"],
    )

    sweep = cfg["sweep"]
    dc_values = sweep["diffusion_coefficient"]
    dr_values = sweep["decay_rate"]
    seeds = cfg["seeds"]

    combos = list(product(dc_values, dr_values))
    total = len(combos) * len(seeds)
    logger.info(
        "Protocol 6 Pilot: %d combinations x %d seeds = %d total runs, "
        "%d epochs each",
        len(combos), len(seeds), total, cfg["num_epochs"],
    )

    all_results: list[dict] = []
    run_idx = 0

    for dc, dr in combos:
        combo_results = []
        for seed in seeds:
            run_idx += 1
            logger.info(
                "Run %d/%d | dc=%.1f dr=%.2f seed=%d",
                run_idx, total, dc, dr, seed,
            )
            result = run_one(dc, dr, seed, cfg)
            run_key = f"dc{dc}_dr{dr}_seed{seed}".replace(".", "p")
            run_path = results_dir / f"p6_pilot_{run_key}.json"
            run_record = {k: v for k, v in result.items() if k != "epoch_series"}
            with open(run_path, "w") as rf:
                json.dump(run_record, rf)
            combo_results.append(result)
            all_results.append(result)

        formed = sum(r["field_formed"] for r in combo_results)
        saturated = sum(r["field_saturated"] for r in combo_results)
        collapsed = sum(r["field_collapsed"] for r in combo_results)
        logger.info(
            "  COMBO dc=%.1f dr=%.2f | formed=%d/10 saturated=%d/10 collapsed=%d/10",
            dc, dr, formed, saturated, collapsed,
        )

    # Build summary JSON
    combo_summaries = []
    for dc, dr in combos:
        runs = [
            r for r in all_results
            if r["diffusion_coefficient"] == dc and r["decay_rate"] == dr
        ]
        formed_count = sum(r["field_formed"] for r in runs)
        saturated_count = sum(r["field_saturated"] for r in runs)
        collapsed_count = sum(r["field_collapsed"] for r in runs)
        mean_corr = float(np.mean([r["entropy_sss_correlation"] for r in runs]))
        meets_regime = (
            formed_count == 10 and saturated_count == 0 and collapsed_count == 0
        )
        combo_summaries.append({
            "diffusion_coefficient": dc,
            "decay_rate": dr,
            "field_formed_count": formed_count,
            "field_saturated_count": saturated_count,
            "field_collapsed_count": collapsed_count,
            "mean_entropy_sss_correlation": round(mean_corr, 4),
            "meets_target_regime": meets_regime,
        })

    def regime_score(s: dict) -> tuple:
        return (
            int(s["meets_target_regime"]),
            s["field_formed_count"],
            -s["field_saturated_count"],
            -s["field_collapsed_count"],
        )

    best = max(combo_summaries, key=regime_score)
    viable_combos = [s for s in combo_summaries if s["meets_target_regime"]]

    summary = {
        "protocol": "P6_pilot",
        "condition": "condition_a",
        "total_runs": total,
        "combinations": combo_summaries,
        "viable_combinations": viable_combos,
        "best_combination": best,
        "design_review_required": len(viable_combos) == 0,
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Summary written to %s", json_path)

    if summary["design_review_required"]:
        logger.warning(
            "NO combination meets all three target regime criteria. "
            "Flag for design review before preregistration."
        )
    else:
        logger.info(
            "Viable combinations: %s",
            [(s["diffusion_coefficient"], s["decay_rate"]) for s in viable_combos],
        )


if __name__ == "__main__":
    main()
