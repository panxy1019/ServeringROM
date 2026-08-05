from unittest import TestCase

from servingrom_telemetry.request_context import RequestTraceContext


class RequestTraceContextTest(TestCase):
    def test_recompute_preserves_trace_and_rotates_request(self) -> None:
        context = RequestTraceContext.create("client-request")
        original_trace = context.trace_id
        request_ids = [context.request_id]
        for expected_attempt in (1, 2, 3):
            previous_attempt, previous_request = context.begin_recompute()
            self.assertEqual(previous_attempt, expected_attempt - 1)
            self.assertEqual(previous_request, request_ids[-1])
            self.assertEqual(context.trace_id, original_trace)
            self.assertEqual(context.attempt_id, expected_attempt)
            self.assertNotIn(context.request_id, request_ids)
            request_ids.append(context.request_id)

    def test_arrival_and_external_id(self) -> None:
        context = RequestTraceContext.create("external")
        self.assertEqual(context.external_request_id, "external")
        self.assertGreater(context.arrival_wall_ns, 0)
        self.assertGreater(context.arrival_mono_ns, 0)
