from pathlib import Path
from unittest import TestCase

from servingrom_telemetry.config import TelemetryConfig


class TelemetryConfigTest(TestCase):
    def test_defaults_disabled(self) -> None:
        config = TelemetryConfig.from_env({})
        self.assertFalse(config.enabled)
        self.assertEqual(config.queue_capacity, 65_536)

    def test_enabled_environment(self) -> None:
        config = TelemetryConfig.from_env(
            {
                "SERVINGROM_TELEMETRY_ENABLED": "yes",
                "SERVINGROM_EXPERIMENT_ID": "experiment",
                "SERVINGROM_RUN_ID": "run",
                "SERVINGROM_CONFIG_ID": "config",
                "SERVINGROM_COMPONENT": "proxy",
                "SERVINGROM_HOST_ID": "host",
                "SERVINGROM_OUTPUT_DIR": "/tmp/output",
                "SERVINGROM_QUEUE_CAPACITY": "128",
                "SERVINGROM_BATCH_SIZE": "16",
                "SERVINGROM_FLUSH_INTERVAL_MS": "50",
                "SERVINGROM_MAX_FILE_BYTES": "4096",
            }
        )
        self.assertTrue(config.enabled)
        self.assertEqual(config.output_dir, Path("/tmp/output"))
        self.assertEqual(config.batch_size, 16)

    def test_enabled_requires_identity(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryConfig.from_env({"SERVINGROM_TELEMETRY_ENABLED": "true"})

    def test_invalid_boolean_and_sizes(self) -> None:
        with self.assertRaises(ValueError):
            TelemetryConfig.from_env({"SERVINGROM_TELEMETRY_ENABLED": "maybe"})
        with self.assertRaises(ValueError):
            TelemetryConfig(queue_capacity=0)
        with self.assertRaises(ValueError):
            TelemetryConfig(queue_capacity=4, batch_size=5)
