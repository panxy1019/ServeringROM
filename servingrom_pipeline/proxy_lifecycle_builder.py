from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .proxy_event_reader import read_proxy_events
from .proxy_state_machine import ProxyLifecycleAnalysis, analyze_proxy_lifecycle


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to write lifecycle Parquet files") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        table = pa.Table.from_pylist(rows)
    else:
        table = pa.table({"trace_id": pa.array([], type=pa.string())})
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _quality_markdown(analysis: ProxyLifecycleAnalysis) -> str:
    metrics = analysis.metrics
    lines = [
        "# Proxy 生命周期数据质量报告",
        "",
        f"- 事件数：{metrics['event_count']}",
        f"- Trace 数：{metrics['trace_count']}",
        f"- Attempt 数：{metrics['attempt_count']}",
        f"- 违规数：{metrics['violation_count']}",
        f"- JSONL 损坏行：{metrics['damaged_line_count']}",
        f"- event_seq 缺口/重复：{metrics['event_seq_gap_count']}",
        "",
        "## 违规分布",
        "",
    ]
    if metrics["violation_counts"]:
        lines.extend(
            f"- `{code}`：{count}" for code, count in metrics["violation_counts"].items()
        )
    else:
        lines.append("- 无")
    return "\n".join(lines) + "\n"


def build_proxy_lifecycle(run_root: Path) -> ProxyLifecycleAnalysis:
    root = Path(run_root)
    dataset = read_proxy_events(root / "raw" / "proxy")
    analysis = analyze_proxy_lifecycle(dataset.events, dataset.summaries, dataset.damaged_lines)
    _write_parquet(analysis.trace_rows, root / "derived" / "trace_lifecycle.parquet")
    _write_parquet(analysis.attempt_rows, root / "derived" / "attempt_lifecycle.parquet")
    report = {"metrics": analysis.metrics, "violations": analysis.violations}
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "proxy_lifecycle_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (reports / "proxy_lifecycle_quality.md").write_text(
        _quality_markdown(analysis), encoding="utf-8"
    )
    return analysis
