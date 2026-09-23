# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""Evidence that the instance, not the load generator, was the bottleneck.

Two probes, recorded on every LaneResult and asserted by
`utils.metrics.assert_measurement_valid`:

- `RunnerCpu`: whole-runner CPU busy fraction over a window from /proc/stat
  (Linux; None elsewhere). Above ~70% the load generator itself is saturated
  and its throughput number says nothing about Vespa.
- `server_cpu_util`: node CPU utilization per Vespa cluster from the
  application's data-plane /prometheus/v1/values (`cpu_util`, percent, from
  the metrics proxy). Near 100% on the container cluster at the measured
  concurrency is the proof the instance was the limit.
"""

import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from vespa.application import Vespa, VespaSync

_PROC_STAT = "/proc/stat"
_CPU_UTIL = re.compile(
    r"^cpu_util\{([^}]*)\}\s+([0-9.eE+-]+)(?:\s+(\d+))?", re.MULTILINE
)
_CLUSTER = re.compile(r'clusterId="([^"]+)"')


def _read_proc_stat() -> Optional[tuple]:
    """(busy, total) jiffies summed over all CPUs, or None when unavailable."""
    try:
        with open(_PROC_STAT) as f:
            fields = f.readline().split()
    except OSError:
        return None
    if not fields or fields[0] != "cpu":
        return None
    values = [int(v) for v in fields[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
    total = sum(values)
    return total - idle, total


@dataclass
class RunnerCpu:
    """Sample /proc/stat at start and stop; `fraction` is busy/total."""

    _start: Optional[tuple] = None
    fraction: Optional[float] = None

    def start(self) -> "RunnerCpu":
        self._start = _read_proc_stat()
        return self

    def stop(self) -> Optional[float]:
        end = _read_proc_stat()
        if self._start is None or end is None:
            self.fraction = None
        else:
            busy = end[0] - self._start[0]
            total = end[1] - self._start[1]
            self.fraction = busy / total if total > 0 else None
        return self.fraction


def server_cpu_util(app: Vespa) -> Tuple[Dict[str, float], Optional[float]]:
    """(cpu_util 0..1 per clusterId, snapshot time as epoch seconds) from the
    data-plane Prometheus endpoint. The metrics proxy publishes a new snapshot
    about once a minute; the timestamp says which interval a value covers.

    Returns ({}, None) if the endpoint is unreachable so a metrics hiccup never
    fails a run by itself; the validity check treats a missing value as unknown.
    """
    try:
        with VespaSync(app=app, pool_connections=1, pool_maxsize=1) as session:
            response = session.http_client.get(
                f"{app.end_point}/prometheus/v1/values", timeout=30
            )
        if response.status_code != 200:
            return {}, None
        text = response.text
    except Exception:
        return {}, None
    util: Dict[str, float] = {}
    snapshot: Optional[float] = None
    for labels, value, timestamp_ms in _CPU_UTIL.findall(text):
        cluster = _CLUSTER.search(labels)
        if cluster:
            # One node per cluster here; keep the max if there are several.
            util[cluster.group(1)] = max(
                util.get(cluster.group(1), 0.0), float(value) / 100.0
            )
            if timestamp_ms:
                snapshot = max(snapshot or 0.0, int(timestamp_ms) / 1000.0)
    return util, snapshot


class ServerCpuSampler:
    """Poll `server_cpu_util` on a thread while the load runs and report the
    peak per cluster over the snapshots that cover the load.

    A snapshot stamped T covers roughly [T - 60 s, T]. Only snapshots stamped
    at least `SNAPSHOT_S` after the load started (fully inside the load) and no
    later than `SNAPSHOT_S` after it ended count. `stop()` keeps polling for up
    to `SNAPSHOT_S` after the load so the last covering snapshot is not missed.
    Falls back to the peak of all samples when no snapshot qualifies."""

    SNAPSHOT_S = 60.0

    def __init__(self, app: Vespa, interval_s: float = 10.0):
        self._app = app
        self._interval_s = interval_s
        self._samples: List[Tuple[Optional[float], Dict[str, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at: Optional[float] = None

    def _poll(self) -> None:
        util, snapshot = server_cpu_util(self._app)
        if util:
            self._samples.append((snapshot, util))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._poll()

    def start(self) -> "ServerCpuSampler":
        self._started_at = time.time()
        self._thread.start()
        return self

    def stop(self, load_start: Optional[float] = None) -> Dict[str, float]:
        """Stop polling. `load_start` (epoch seconds) defaults to start()."""
        load_end = time.time()
        load_start = load_start if load_start is not None else self._started_at
        self._stop.set()
        self._thread.join(timeout=60)
        # Wait for a snapshot stamped after the load ended, so the interval
        # covering the last part of the load is included.
        deadline = load_end + self.SNAPSHOT_S + 15
        while time.time() < deadline:
            self._poll()
            if self._samples and (self._samples[-1][0] or 0) >= load_end:
                break
            time.sleep(self._interval_s)
        covering = [
            util
            for snapshot, util in self._samples
            if snapshot is not None
            and load_start + self.SNAPSHOT_S <= snapshot <= load_end + self.SNAPSHOT_S
        ]
        if not covering:
            covering = [util for _, util in self._samples]
        peak: Dict[str, float] = {}
        for util in covering:
            for cluster, value in util.items():
                peak[cluster] = max(peak.get(cluster, 0.0), value)
        return peak
