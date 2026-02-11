"""cpu, memory, and gpu resource monitoring."""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import psutil


@dataclass
class GPUInfo:
    """gpu device information."""

    index: int
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature_c: float | None = None

    @property
    def memory_percent(self) -> float:
        """memory usage as percentage."""
        if self.memory_total_gb == 0:
            return 0.0
        return (self.memory_used_gb / self.memory_total_gb) * 100


@dataclass
class ResourceSnapshot:
    """snapshot of system resource usage."""

    timestamp: float
    cpu_percent: float
    cpu_count: int
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpus: list[GPUInfo] = field(default_factory=list)

    @property
    def has_gpu(self) -> bool:
        """check if gpu info is available."""
        return len(self.gpus) > 0

    @property
    def primary_gpu(self) -> GPUInfo | None:
        """get primary gpu (index 0)."""
        return self.gpus[0] if self.gpus else None


def get_gpu_info() -> list[GPUInfo]:
    """get gpu information using nvidia-smi or torch."""
    gpus = []

    # try torch first (more reliable when torch is loaded).
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_memory / (1024**3)
                mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)

                # use reserved as "used" for better accuracy.
                mem_used = mem_reserved
                mem_free = mem_total - mem_used

                gpus.append(
                    GPUInfo(
                        index=i,
                        name=props.name,
                        memory_total_gb=mem_total,
                        memory_used_gb=mem_used,
                        memory_free_gb=mem_free,
                        utilization_percent=0.0,  # torch doesn't provide this.
                    )
                )

            # try to get utilization from pynvml if available.
            try:
                import pynvml

                pynvml.nvmlInit()
                for gpu in gpus:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu.index)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu.utilization_percent = util.gpu

                    try:
                        temp = pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                        gpu.temperature_c = temp
                    except Exception:
                        pass
                pynvml.nvmlShutdown()
            except Exception:
                pass

            return gpus
    except Exception:
        pass

    # fallback to pynvml directly.
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            temp = None
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                pass

            gpus.append(
                GPUInfo(
                    index=i,
                    name=name,
                    memory_total_gb=mem_info.total / (1024**3),
                    memory_used_gb=mem_info.used / (1024**3),
                    memory_free_gb=mem_info.free / (1024**3),
                    utilization_percent=util.gpu,
                    temperature_c=temp,
                )
            )

        pynvml.nvmlShutdown()
    except Exception:
        pass

    return gpus


def get_resource_snapshot() -> ResourceSnapshot:
    """get current resource usage snapshot."""
    # cpu info.
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count()

    # memory info.
    mem = psutil.virtual_memory()
    memory_used_gb = mem.used / (1024**3)
    memory_total_gb = mem.total / (1024**3)
    memory_percent = mem.percent

    # gpu info.
    gpus = get_gpu_info()

    return ResourceSnapshot(
        timestamp=time.time(),
        cpu_percent=cpu_percent,
        cpu_count=cpu_count,
        memory_used_gb=memory_used_gb,
        memory_total_gb=memory_total_gb,
        memory_percent=memory_percent,
        gpus=gpus,
    )


def get_system_info() -> dict:
    """get static system information."""
    cpu_count = psutil.cpu_count()
    cpu_count_physical = psutil.cpu_count(logical=False)
    mem = psutil.virtual_memory()

    info = {
        "cpu_count": cpu_count,
        "cpu_count_physical": cpu_count_physical,
        "memory_total_gb": mem.total / (1024**3),
        "gpus": [],
    }

    # get gpu names.
    gpus = get_gpu_info()
    for gpu in gpus:
        info["gpus"].append(
            {
                "index": gpu.index,
                "name": gpu.name,
                "memory_gb": gpu.memory_total_gb,
            }
        )

    return info


class ResourceMonitor:
    """background resource monitoring with callbacks."""

    def __init__(
        self,
        interval: float = 1.0,
        callback: Callable[[ResourceSnapshot], None] | None = None,
    ):
        """
        initialize resource monitor.

        args:
            interval: sampling interval in seconds.
            callback: function called with each snapshot.
        """
        self.interval = interval
        self.callback = callback
        self._running = False
        self._thread: threading.Thread | None = None
        self._snapshots: list[ResourceSnapshot] = []
        self._lock = threading.Lock()

    def start(self) -> None:
        """start background monitoring."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _monitor_loop(self) -> None:
        """background monitoring loop."""
        # initialize cpu_percent (first call returns 0).
        psutil.cpu_percent(interval=None)

        while self._running:
            snapshot = get_resource_snapshot()

            with self._lock:
                self._snapshots.append(snapshot)
                # keep last 100 snapshots.
                if len(self._snapshots) > 100:
                    self._snapshots = self._snapshots[-100:]

            if self.callback:
                try:
                    self.callback(snapshot)
                except Exception:
                    pass

            time.sleep(self.interval)

    def get_latest(self) -> ResourceSnapshot | None:
        """get most recent snapshot."""
        with self._lock:
            return self._snapshots[-1] if self._snapshots else None

    def get_average(self) -> ResourceSnapshot | None:
        """get average of recent snapshots."""
        with self._lock:
            if not self._snapshots:
                return None

            n = len(self._snapshots)
            avg_cpu = sum(s.cpu_percent for s in self._snapshots) / n
            avg_mem = sum(s.memory_percent for s in self._snapshots) / n
            avg_mem_used = sum(s.memory_used_gb for s in self._snapshots) / n

            # average gpu stats if available.
            gpus = []
            if self._snapshots[0].gpus:
                for i in range(len(self._snapshots[0].gpus)):
                    gpu_snapshots = [s.gpus[i] for s in self._snapshots if len(s.gpus) > i]
                    if gpu_snapshots:
                        gpus.append(
                            GPUInfo(
                                index=i,
                                name=gpu_snapshots[0].name,
                                memory_total_gb=gpu_snapshots[0].memory_total_gb,
                                memory_used_gb=sum(g.memory_used_gb for g in gpu_snapshots) / len(gpu_snapshots),
                                memory_free_gb=sum(g.memory_free_gb for g in gpu_snapshots) / len(gpu_snapshots),
                                utilization_percent=sum(g.utilization_percent for g in gpu_snapshots) / len(gpu_snapshots),
                            )
                        )

            return ResourceSnapshot(
                timestamp=time.time(),
                cpu_percent=avg_cpu,
                cpu_count=self._snapshots[0].cpu_count,
                memory_used_gb=avg_mem_used,
                memory_total_gb=self._snapshots[0].memory_total_gb,
                memory_percent=avg_mem,
                gpus=gpus,
            )

    def __enter__(self) -> "ResourceMonitor":
        """context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """context manager exit."""
        self.stop()
