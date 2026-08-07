from __future__ import annotations

from unittest import TestCase

from servingrom_pipeline.snapshot_validation import _expected_active_attempts


class SnapshotValidationTest(TestCase):
    def test_rejected_request_is_balanced_as_arrival_and_terminal_outflow(self):
        self.assertEqual(
            _expected_active_attempts(
                active=2,
                arrivals=1,
                completed=0,
                cancelled=0,
                failed=0,
                rejected=0,
            ),
            3,
        )
        self.assertEqual(
            _expected_active_attempts(
                active=3,
                arrivals=0,
                completed=0,
                cancelled=0,
                failed=0,
                rejected=1,
            ),
            2,
        )
