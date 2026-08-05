from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO


class RotatingJSONLSink:
    """Single-process JSONL sink with size-based rotation."""

    def __init__(
        self,
        output_dir: Path,
        component: str,
        process_instance_id: str,
        max_file_bytes: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.component = component.replace("/", "_")
        self.process_instance_id = process_instance_id
        self.max_file_bytes = max_file_bytes
        self._index = 0
        self._size = 0
        self._file: BinaryIO | None = None
        self.paths: list[Path] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        path = self.output_dir / f"{self.process_instance_id}.{self._index:05d}.jsonl"
        self._file = path.open("ab", buffering=256 * 1024)
        self._size = path.stat().st_size
        self.paths.append(path)

    def _rotate(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
        self._index += 1
        self._file = None
        self._size = 0

    def write_batch(self, encoded_events: list[bytes]) -> int:
        if not encoded_events:
            return 0
        total_written = 0
        chunk: list[bytes] = []
        chunk_size = 0

        def write_chunk() -> None:
            nonlocal chunk, chunk_size, total_written
            if not chunk:
                return
            if self._file is None:
                self._open()
            payload = b"".join(chunk)
            assert self._file is not None
            written = self._file.write(payload)
            if written != len(payload):
                raise OSError(f"short JSONL write: {written}/{len(payload)} bytes")
            self._size += written
            total_written += written
            chunk = []
            chunk_size = 0

        for encoded in encoded_events:
            if self._file is None:
                self._open()
            projected = self._size + chunk_size + len(encoded)
            if (self._size > 0 or chunk_size > 0) and projected > self.max_file_bytes:
                write_chunk()
                self._rotate()
                self._open()
            chunk.append(encoded)
            chunk_size += len(encoded)
        write_chunk()
        return total_written

    def flush(self) -> None:
        if self._file is not None:
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            self._file = None
