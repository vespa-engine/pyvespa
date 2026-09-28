# Copyright Vespa.ai. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

"""CPU probes for the two sides of a measurement: the load generator (the GitHub
runner or laptop running the tests) and the Vespa instance. The values feed the
validity checks in utils/asserts.py and are exported with every record."""

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


def _read_proc_stat() -> Optional[Tuple[int, int]]:
    """(busy, total) jiffies of the whole machine; None off Linux."""
    try:
        with open(_PROC_STAT) as f:
            fields = f.readline().split()
    except OSError:
        return None
    if not fields or fields[0] != "cpu":
        return None
    # The "cpu" line lists time spent per state. Only the first eight states add
    # up to the total; the two after them (guest time) are already included.
    user, nice, system, idle, iowait, irq, softirq, steal = (
        [int(v) for v in fields[1:9]] + [0] * 8
    )[:8]
    total = user + nice + system + idle + iowait + irq + softirq + steal
    busy = total - idle - iowait
    return busy, total


@dataclass
class LoadGeneratorCpu:
    """Busy share of the whole load-generator machine between start and stop."""

    _start: Optional[Tuple[int, int]] = None
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
    """CPU per cluster (0..1) from the instance's own metrics endpoint, and the
    time of the snapshot they belong to. Empty when the probe fails."""
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
    """Peak instance CPU per cluster while a load ran.

    The instance's cpu_util is a 60-second average stamped at the end of its
    window, so the reading that covers the load arrives up to a minute after
    the load stops, and a snapshot covers the load only when its timestamp is
    between load start + 60 s and load end + 60 s. Snapshot times are compared
    with the runner's clock; both sides run NTP. A load start that includes
    k6's ramp only widens what the peak is taken over."""

    SNAPSHOT_S = 60.0

    def __init__(self, app: Vespa, interval_s: float = 10.0):
        self._app = app
        self._interval_s = interval_s
        self._samples: List[Tuple[Optional[float], Dict[str, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at: Optional[float] = None

    def start(self) -> "InstanceCpuSampler":
        self._started_at = time.time()
        self._thread.start()
        return self

    def stop(self, load_start: Optional[float] = None) -> Dict[str, float]:
        """Peak per cluster over the snapshots covering the load; empty when
        no snapshot covered it, so the caller reports the CPU as unknown."""
        load_end = time.time()
        load_start = load_start if load_start is not None else self._started_at
        self._stop.set()
        self._thread.join(timeout=60)
        self._wait_for_snapshot_after(load_end)
        return self._peak_between(load_start, load_end)

    def _poll(self) -> None:
        util, snapshot = instance_cpu_util(self._app)
        if util:
            self._samples.append((snapshot, util))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._poll()

    def _wait_for_snapshot_after(self, load_end: float) -> None:
        deadline = load_end + self.SNAPSHOT_S + 15
        while time.time() < deadline:
            self._poll()
            if self._samples and (self._samples[-1][0] or 0) >= load_end:
                return
            time.sleep(self._interval_s)

    def _peak_between(self, load_start: float, load_end: float) -> Dict[str, float]:
        first = load_start + self.SNAPSHOT_S
        last = load_end + self.SNAPSHOT_S
        peak: Dict[str, float] = {}
        for snapshot, util in self._samples:
            if snapshot is None or not first <= snapshot <= last:
                continue
            for cluster, value in util.items():
                peak[cluster] = max(peak.get(cluster, 0.0), value)
        return peak
