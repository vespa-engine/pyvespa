# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""CPU probes for the two sides of a measurement: the load generator (the GitHub
runner or laptop running the tests) and the Vespa instance. Both feed the
validity checks in utils/metrics.py."""

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
class LoadGeneratorCpu:
    """Sample /proc/stat at start and stop; `fraction` is busy/total."""

    _start: Optional[tuple] = None
    fraction: Optional[float] = None

    def start(self) -> "LoadGeneratorCpu":
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


def instance_cpu_util(app: Vespa) -> Tuple[Dict[str, float], Optional[float]]:
    """Read per-cluster CPU fractions and snapshot time; return ({}, None) if unavailable."""
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


class InstanceCpuSampler:
    """Report peak CPU from snapshots covering the load (each spans about 60 seconds)."""

    SNAPSHOT_S = 60.0

    def __init__(self, app: Vespa, interval_s: float = 10.0):
        self._app = app
        self._interval_s = interval_s
        self._samples: List[Tuple[Optional[float], Dict[str, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at: Optional[float] = None

    def _poll(self) -> None:
        util, snapshot = instance_cpu_util(self._app)
        if util:
            self._samples.append((snapshot, util))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._poll()

    def start(self) -> "InstanceCpuSampler":
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
