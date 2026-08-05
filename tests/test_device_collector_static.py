from pathlib import Path


def test_device_collector_has_no_process_spawning() -> None:
    source = Path("scripts/collect_servingrom_device_metrics.py").read_text()
    assert "import subprocess" not in source
    assert "subprocess." not in source
    assert "os.system" not in source
    assert 'Path("/proc")' in source
    assert "urllib.request.urlopen" in source


if __name__ == "__main__":
    test_device_collector_has_no_process_spawning()
    print("ServingROM device collector static test: PASS")
