from __future__ import annotations

import ast
from pathlib import Path


ROOTS = {
    "core": Path("/vllm-workspace/vllm/vllm/v1/engine/core.py"),
    "scheduler_output": Path("/vllm-workspace/vllm/vllm/v1/core/sched/output.py"),
    "mooncake": Path(
        "/vllm-workspace/vllm/vllm/distributed/kv_transfer/kv_connector/v1/"
        "mooncake/mooncake_connector.py"
    ),
    "model_runner": Path(
        "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
    ),
    "device": Path("/opt/servingrom/scripts/collect_servingrom_device_metrics.py"),
}


def main() -> None:
    sources = {name: path.read_text(encoding="utf-8") for name, path in ROOTS.items()}
    for name, source in sources.items():
        ast.parse(source, filename=str(ROOTS[name]))
    assert "servingrom_iteration_id: int | None = None" in sources["scheduler_output"]
    assert "scheduler_output.servingrom_iteration_id = iteration_id" in sources["core"]
    assert '"kv_transfer_completed"' in sources["mooncake"]
    assert '"total_bytes": sum(request_lengths)' in sources["mooncake"]
    assert '"model_execution_batch"' in sources["model_runner"]
    added_model_runner = Path(
        "/opt/servingrom/patches/vllm_ascend/0001-servingrom-model-batch-telemetry.patch"
    ).read_text(encoding="utf-8")
    assert not any(
        "_sync_device" in line for line in added_model_runner.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )
    assert "import subprocess" not in sources["device"]
    assert "subprocess." not in sources["device"]
    print("ServingROM internal patch static tests: PASS")


if __name__ == "__main__":
    main()
