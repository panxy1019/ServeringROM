from __future__ import annotations

from scripts.rom_workload import backlog_trend


def test_backlog_trend_ignores_isolated_low_load_spike() -> None:
    health = [
        {"ts_wall_ns": index * 1_000_000_000, "request_num": value}
        for index, value in enumerate([0] * 10 + [0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    ]
    _slope, growth, end_inventory = backlog_trend(health)
    assert growth <= 2.0
    assert end_inventory == 0


def test_backlog_trend_detects_sustained_growth() -> None:
    health = [
        {"ts_wall_ns": index * 1_000_000_000, "request_num": index}
        for index in range(20)
    ]
    slope, growth, end_inventory = backlog_trend(health)
    assert slope > 0.05
    assert growth > 1.0
    assert end_inventory > 1.0


def test_backlog_trend_excludes_post_measurement_recovery() -> None:
    health = [
        {"ts_wall_ns": index * 1_000_000_000, "request_num": value}
        for index, value in enumerate([0, 1, 2, 3, 4, 5, 4, 3, 2, 0])
    ]
    slope, _growth, end_inventory = backlog_trend(
        health,
        start_wall_ns=0,
        end_wall_ns=6_000_000_000,
    )
    assert slope > 0
    assert end_inventory == 5
