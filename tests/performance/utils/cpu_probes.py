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


def instance_cpu_util(app: Vespa) -> Dict[str, float]:
    """CPU per cluster (0..1) from the instance's own metrics endpoint; empty
    when the probe fails. The timestamp on the values is the time of the
    request, not of the window they average, so it is not returned."""
    try:
        with VespaSync(app=app, pool_connections=1, pool_maxsize=1) as session:
            response = session.http_client.get(
                f"{app.end_point}/prometheus/v1/values", timeout=30
            )
        if response.status_code != 200:
            return {}
        text = response.text
    except Exception:
        return {}
    util: Dict[str, float] = {}
    for labels, value, _ in _CPU_UTIL.findall(text):
        cluster = _CLUSTER.search(labels)
        if cluster:
            # One node per cluster here; keep the max if there are several.
            util[cluster.group(1)] = max(
                util.get(cluster.group(1), 0.0), float(value) / 100.0
            )
    return util


class InstanceCpuSampler:
    """Mean instance CPU per cluster while a load ran.

    The instance's cpu_util is an average over about the last 60 s, and the metrics
    proxy refreshes it about every 30 s. Its timestamp is the time of the
    request, not of the window, so a reading is placed by when the sampler took
    it: one taken at least FULL_WINDOW_S after the load started, and before it
    ended, averages load only. `samples` counts those readings. A load too
    short for any (the pyvespa batches) still gets a value, the peak of the
    readings taken after PARTIAL_WINDOW_S, which also averages some of the time
    before the load; `samples` is 0 then."""

    FULL_WINDOW_S = 90.0
    PARTIAL_WINDOW_S = 30.0

    def __init__(self, app: Vespa, interval_s: float = 10.0):
        self._app = app
        self._interval_s = interval_s
        self._readings: List[Tuple[float, Dict[str, float]]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started_at: Optional[float] = None
        self.samples = 0

    def start(self) -> "InstanceCpuSampler":
        self._started_at = time.time()
        self._thread.start()
        return self

    def stop(self, load_start: Optional[float] = None) -> Dict[str, float]:
        """CPU per cluster over the load; empty when nothing was read, so the
        caller reports the CPU as unknown."""
        load_end = time.time()
        load_start = load_start if load_start is not None else self._started_at
        self._stop.set()
        self._thread.join(timeout=60)
        full = self._between(load_start + self.FULL_WINDOW_S, load_end)
        if full:
            # A cluster whose value happened not to change still had its window
            # refreshed, so the busiest-changing cluster counts the readings.
            self.samples = max(len(values) for values in full.values())
            return {c: sum(values) / len(values) for c, values in full.items()}
        partial = self._between(load_start + self.PARTIAL_WINDOW_S, load_end)
        return {c: max(values) for c, values in partial.items()}

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            util = instance_cpu_util(self._app)
            if util:
                self._readings.append((time.time(), util))

    def _between(self, first: float, last: float) -> Dict[str, List[float]]:
        """Distinct readings per cluster taken in [first, last]; a value read
        again before the proxy refreshed it counts once."""
        values: Dict[str, List[float]] = {}
        for taken_at, util in self._readings:
            if not first <= taken_at <= last:
                continue
            for cluster, value in util.items():
                seen = values.setdefault(cluster, [])
                if not seen or seen[-1] != value:
                    seen.append(value)
        return values
