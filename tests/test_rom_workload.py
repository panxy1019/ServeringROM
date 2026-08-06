from __future__ import annotations

from scripts.rom_workload import backlog_trend


def test_backlog_trend_ignores_isolated_low_load_spike() -> None:
    health = [
        {"ts_wall_ns": index * 1_000_000_000, "request_num": value}
        for index, value in enumerate([0] * 10 + [0, 0, 1, 0, 1, 0, 0, 0, 0, 0])
    ]
    _slope, growth = backlog_trend(health)
    assert growth <= 1.0


def test_backlog_trend_detects_sustained_growth() -> None:
    health = [
        {"ts_wall_ns": index * 1_000_000_000, "request_num": index}
        for index in range(20)
    ]
    slope, growth = backlog_trend(health)
    assert slope > 0.05
    assert growth > 1.0
