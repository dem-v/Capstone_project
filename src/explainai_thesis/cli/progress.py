"""Terminal progress UX shared by long-running CLI scripts.

The classifier-outcome scan needs an interactive 6-line live progress
display plus a non-interactive fallback that prints each composed
update once. The helpers below are output-channel-agnostic so they can
be reused by any future long-run script (improvement experiment, CT
pilot smoke).
"""
from __future__ import annotations

import shutil
import sys
import textwrap
import time
from datetime import datetime


def log_progress(message: str, start_time: float) -> None:
    """Emit a one-line timestamped progress message to stdout.

    `start_time` is a `time.perf_counter()` reference captured at the
    start of the run. The output is prefixed with elapsed minutes since
    that reference, flushed immediately so long-running CUDA loops
    surface progress in real time. Shared by long-running CLI scripts
    (calibration, single-case threshold visualization, etc.).
    """
    elapsed_minutes = (time.perf_counter() - start_time) / 60.0
    print(f"[{elapsed_minutes:6.1f} min] {message}", flush=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def estimate_eta(completed: int, total: int, elapsed: float) -> float | None:
    if completed <= 0 or total <= 0:
        return None
    remaining = max(0, total - completed)
    return elapsed / completed * remaining


def progress_stats_line(
    *,
    candidate_number: int,
    candidate_total: int,
    selected_total: int,
    target_total: int,
    outcome_counts: dict[str, int],
    elapsed: float,
    eta: float | None,
) -> str:
    return (
        f"[{timestamp()}] Candidate {candidate_number}/{candidate_total} | "
        f"kept={selected_total}/{target_total} | "
        f"TP={outcome_counts['tp']} FP={outcome_counts['fp']} "
        f"TN={outcome_counts['tn']} FN={outcome_counts['fn']} | "
        f"elapsed={format_duration(elapsed)} | ETA≈{format_duration(eta)}"
    )


class LiveProgress:
    """Six-line in-place progress display with a non-TTY fallback."""

    LINE_COUNT = 6

    def __init__(self) -> None:
        self._started = False
        self._live = sys.stdout.isatty()
        self._latest_lines: list[str] = []

    @staticmethod
    def _terminal_width() -> int:
        try:
            return max(40, shutil.get_terminal_size(fallback=(120, 20)).columns)
        except OSError:
            return 120

    def _fit_line(self, text: str) -> str:
        width = self._terminal_width()
        max_len = max(1, width - 1)
        clean = text.replace("\n", " ")
        if len(clean) > max_len:
            return clean[: max(1, max_len - 1)] + "…"
        return clean.ljust(max_len)

    def _compose_lines(self, stats: str, detail: str) -> list[str]:
        width = self._terminal_width()
        max_len = max(20, width - 1)
        parts = [
            part.strip()
            for part in f"{stats} | {detail}".split(" | ")
            if part.strip()
        ]
        lines: list[str] = []
        current = ""
        for part in parts:
            candidate = part if not current else f"{current} | {part}"
            if len(candidate) <= max_len:
                current = candidate
                continue
            if current:
                lines.extend(textwrap.wrap(current, width=max_len) or [current])
            current = part
        if current:
            lines.extend(textwrap.wrap(current, width=max_len) or [current])
        if len(lines) > self.LINE_COUNT:
            overflow = " | ".join(lines[self.LINE_COUNT - 1 :])
            lines = lines[: self.LINE_COUNT - 1] + [overflow]
        lines.extend([""] * (self.LINE_COUNT - len(lines)))
        return lines[: self.LINE_COUNT]

    def update(self, stats: str, detail: str) -> None:
        lines = self._compose_lines(stats, detail)
        self._latest_lines = lines
        if not self._live:
            return
        if self._started:
            sys.stdout.write(f"\x1b[{self.LINE_COUNT}F")
        else:
            self._started = True
        for line in lines:
            sys.stdout.write(f"\r\x1b[2K{self._fit_line(line)}\n")
        sys.stdout.flush()

    def finish(self) -> None:
        if self._live and self._started:
            sys.stdout.write("\n")
            sys.stdout.flush()
        elif not self._live and self._latest_lines:
            for line in self._latest_lines:
                if line:
                    print(line, flush=True)
