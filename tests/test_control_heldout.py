from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from control_heldout_workload import build_heldout_schedule
from run_control_heldout_campaign import HeldoutCampaign


def test_heldout_schedules_respect_safety_and_coverage() -> None:
    expected = {
        "interpolation": {0.4, 0.5, 0.6},
        "unseen-composite": {0.3, 0.4, 0.5, 0.6, 0.7},
        "slow-ramp": {0.3, 0.4, 0.5, 0.6, 0.7},
        "boundary-near": {0.2, 0.4, 0.5, 0.6, 0.8},
    }
    for family, levels in expected.items():
        for seed in range(100):
            rows = build_heldout_schedule(family, seed)
            assert levels <= {row["rho_A"] for row in rows}
            assert min(b["offset_seconds"] - a["offset_seconds"] for a, b in zip(rows, rows[1:])) >= 15
            assert max(abs(b["rho_A"] - a["rho_A"]) for a, b in zip(rows, rows[1:])) <= 0.2000000001


def test_campaign_matrix_is_isolated_and_unique() -> None:
    campaign = HeldoutCampaign(argparse.Namespace(
        project_root=ROOT,
        config="configs/servingrom_control_heldout_v1.json",
    ))
    rows = campaign.plan()
    assert len(rows) == 10
    assert all(row["split"] == "test/control-heldout" for row in rows)
    assert len({row["arrival_seed"] for row in rows}) == 10
    assert len({row["trajectory_seed"] for row in rows}) == 10
    assert sum(row["benchmark_class"] == "core" for row in rows) == 8
    assert sum(row["benchmark_class"] == "robustness" for row in rows) == 2
