from __future__ import annotations

import httpx
import argparse
import asyncio
from unittest import IsolatedAsyncioTestCase
from unittest.mock import patch

from scripts import pd_proxy
from servingrom_telemetry.request_context import RequestTraceContext


class RecordingEmitter:
    def __init__(self, *, fail: bool = False) -> None:
        self.events: list[dict] = []
        self.fail = fail

    def emit(self, event_type, payload, **identity):
        if self.fail:
            raise OSError("injected telemetry failure")
        self.events.append({"event_type": event_type, "payload": dict(payload), **identity})
        return True

    def flush(self, timeout_s=None):
        return True

    def close(self, timeout_s=None):
        return True

    def health_snapshot(self):
        return {"enabled": True}


class ProxyTelemetryIntegrationTest(IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.previous = pd_proxy.telemetry_emitter

    async def asyncTearDown(self) -> None:
        pd_proxy.telemetry_emitter = self.previous

    async def test_prefill_retry_is_recorded_without_rotating_attempt(self) -> None:
        calls = 0

        async def backend(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("retry me", request=request)
            return httpx.Response(200, json={"kv_transfer_params": {"remote": "ok"}})

        recorder = RecordingEmitter()
        pd_proxy.telemetry_emitter = recorder
        context = RequestTraceContext.create("external")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(backend), base_url="http://prefill"
        ) as client:
            response = await pd_proxy.send_request_to_service(
                client,
                "/completions",
                {"prompt": "not emitted", "max_tokens": 8},
                context.request_id,
                context,
                "prefill",
                max_retries=2,
                base_delay=0,
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([event["event_type"] for event in recorder.events], ["backend_retry"])
        self.assertEqual(recorder.events[0]["attempt_id"], 0)
        self.assertNotIn("prompt", recorder.events[0]["payload"])

    async def test_decode_stream_retry_is_recorded_and_body_survives(self) -> None:
        calls = 0

        async def backend(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise httpx.ConnectError("retry me", request=request)
            return httpx.Response(200, content=b'data: {"choices":[{"text":"ok"}]}\n')

        recorder = RecordingEmitter()
        pd_proxy.telemetry_emitter = recorder
        context = RequestTraceContext.create()
        chunks = []
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(backend), base_url="http://decode"
        ) as client:
            async for chunk in pd_proxy.stream_service_response_with_retry(
                client,
                "/completions",
                {"prompt": "not emitted"},
                context.request_id,
                context,
                "decode",
                max_retries=2,
                base_delay=0,
            ):
                chunks.append(chunk)
        self.assertIn(b"ok", b"".join(chunks))
        self.assertEqual([event["event_type"] for event in recorder.events], ["backend_retry"])

    async def test_telemetry_failure_is_isolated_from_proxy(self) -> None:
        pd_proxy.telemetry_emitter = RecordingEmitter(fail=True)
        context = RequestTraceContext.create()
        self.assertFalse(pd_proxy.emit_proxy_event(context, "request_arrival", {"input_tokens": 1}))

    async def test_cancel_during_prefill_releases_reserved_scheduler_load(self) -> None:
        class Runtime:
            def __init__(self) -> None:
                self.calls = []

            async def schedule(self, method, *args, **kwargs):
                self.calls.append((method, args, kwargs))
                if method == "begin_request":
                    return {"key": "prefill", "host": "127.0.0.1", "port": 13700}
                return None

            async def get_client(self, role, key):
                return object()

        async def cancel(*args, **kwargs):
            raise asyncio.CancelledError

        runtime = Runtime()
        args = argparse.Namespace(
            max_prefill_inflight_tokens=8192,
            max_retries=3,
            retry_delay=0,
        )
        context = RequestTraceContext.create()
        with (
            patch.object(pd_proxy, "get_runtime", return_value=runtime),
            patch.object(pd_proxy, "get_global_args", return_value=args),
            patch.object(pd_proxy, "send_request_to_service", side_effect=cancel),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await pd_proxy.assign_instances(
                    "/completions", {}, 32, 16, context, is_initial_request=True
                )
        self.assertEqual(runtime.calls[0][0], "begin_request")
        self.assertEqual(runtime.calls[-1][0], "finish_request")
        self.assertTrue(runtime.calls[-1][2]["release_prefill_kv"])

    async def test_chat_token_count_uses_input_ids_not_batch_encoding_keys(self) -> None:
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [[10, 11, 12, 13]], "attention_mask": [[1, 1, 1, 1]]}

        estimator = pd_proxy.RequestTokenEstimator.__new__(pd_proxy.RequestTokenEstimator)
        estimator._tokenizer = Tokenizer()
        self.assertEqual(
            estimator.request_tokens({"messages": [{"role": "user", "content": "x"}]}),
            4,
        )
