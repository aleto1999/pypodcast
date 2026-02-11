# Resource Monitoring

Real-time CPU, memory, and GPU monitoring utilities with Rich terminal displays for pipeline execution.

## Overview

This package provides system resource monitoring during long-running pipeline operations. It displays real-time usage statistics for CPU, memory, and GPU resources using Rich terminal components, helping users monitor system utilization and identify bottlenecks.

## Features

- **CPU Monitoring**: Usage percentage and core count
- **Memory Monitoring**: Used/total RAM with percentage
- **GPU Monitoring**: VRAM usage, utilization, and temperature (NVIDIA)
- **Background Sampling**: Non-blocking resource polling
- **Rich Displays**: Beautiful terminal UI with progress bars
- **Average Statistics**: Summary of resource usage after processing
- **Context Manager Support**: Easy integration with `with` statements
- **Multi-GPU Support**: Monitors all available GPUs

## Installation

The package is automatically installed with the main project:

```bash
uv sync
```

## Usage

### Quick Start

```python
from rich.console import Console
from podcast_conversations.monitoring import (
    ResourceDisplay,
    print_system_info,
)

console = Console()

# Print system info at startup
print_system_info(console)

# Use ResourceDisplay for progress with live resource monitoring
with ResourceDisplay(console=console) as display:
    task = display.progress.add_task("Processing files...", total=100)

    for i in range(100):
        # Your processing code here
        display.progress.advance(task)

    # Print resource usage summary
    display.print_summary()
```

### Basic Resource Snapshot

```python
from podcast_conversations.monitoring import get_resource_snapshot, ResourceSnapshot

# Get current resource usage
snapshot: ResourceSnapshot = get_resource_snapshot()

print(f"CPU: {snapshot.cpu_percent:.1f}% ({snapshot.cpu_count} cores)")
print(f"Memory: {snapshot.memory_percent:.1f}% ({snapshot.memory_used_gb:.1f}/{snapshot.memory_total_gb:.1f} GB)")

if snapshot.has_gpu:
    gpu = snapshot.primary_gpu
    print(f"GPU: {gpu.name}")
    print(f"  Memory: {gpu.memory_percent:.1f}% ({gpu.memory_used_gb:.1f}/{gpu.memory_total_gb:.1f} GB)")
    print(f"  Utilization: {gpu.utilization_percent:.1f}%")
    if gpu.temperature_c:
        print(f"  Temperature: {gpu.temperature_c}°C")
```

### Background Monitoring

```python
from podcast_conversations.monitoring import ResourceMonitor

def on_snapshot(snapshot):
    print(f"CPU: {snapshot.cpu_percent:.1f}%")

# Start background monitoring
monitor = ResourceMonitor(interval=1.0, callback=on_snapshot)
monitor.start()

# ... do work ...

# Get statistics
latest = monitor.get_latest()
average = monitor.get_average()
print(f"Average CPU: {average.cpu_percent:.1f}%")

monitor.stop()

# Or use context manager
with ResourceMonitor(interval=1.0) as monitor:
    # ... do work ...
    average = monitor.get_average()
```

### System Information

```python
from podcast_conversations.monitoring import get_system_info, print_system_info
from rich.console import Console

# Get as dictionary
info = get_system_info()
print(f"CPUs: {info['cpu_count']} ({info['cpu_count_physical']} physical)")
print(f"Memory: {info['memory_total_gb']:.1f} GB")
for gpu in info['gpus']:
    print(f"GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.0f} GB)")

# Or print formatted
console = Console()
print_system_info(console)
```

### Resource Panel

```python
from rich.console import Console
from rich.live import Live
from podcast_conversations.monitoring import create_resource_panel

console = Console()

# Single display
panel = create_resource_panel()
console.print(panel)

# Live updating display
with Live(create_resource_panel(), refresh_per_second=2, console=console):
    import time
    time.sleep(10)  # Updates every 0.5 seconds
```

## API Reference

### Data Classes

#### `ResourceSnapshot`

Snapshot of system resource usage.

```python
@dataclass
class ResourceSnapshot:
    timestamp: float
    cpu_percent: float
    cpu_count: int
    memory_used_gb: float
    memory_total_gb: float
    memory_percent: float
    gpus: list[GPUInfo]

    @property
    def has_gpu(self) -> bool: ...

    @property
    def primary_gpu(self) -> GPUInfo | None: ...
```

#### `GPUInfo`

GPU device information.

```python
@dataclass
class GPUInfo:
    index: int
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_percent: float
    temperature_c: float | None

    @property
    def memory_percent(self) -> float: ...
```

### Functions

#### `get_resource_snapshot() -> ResourceSnapshot`

Get current resource usage snapshot.

#### `get_system_info() -> dict`

Get static system information (CPU count, memory, GPUs).

#### `get_gpu_info() -> list[GPUInfo]`

Get GPU information using PyTorch or pynvml.

#### `create_resource_panel(snapshot=None) -> Panel`

Create a Rich panel showing resource usage.

#### `print_system_info(console=None) -> None`

Print formatted system information.

### Classes

#### `ResourceMonitor`

Background resource monitoring with callbacks.

```python
class ResourceMonitor:
    def __init__(
        self,
        interval: float = 1.0,
        callback: Callable[[ResourceSnapshot], None] | None = None,
    ): ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def get_latest(self) -> ResourceSnapshot | None: ...
    def get_average(self) -> ResourceSnapshot | None: ...
```

#### `ResourceDisplay`

Live resource display with progress bars.

```python
class ResourceDisplay:
    def __init__(
        self,
        console: Console | None = None,
        refresh_rate: float = 2.0,
        show_resources: bool = True,
    ): ...

    @property
    def progress(self) -> Progress: ...

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def update(self) -> None: ...
    def print_summary(self) -> None: ...
```

## Display Output Example

```
╭─────────────── Resources ───────────────╮
│ CPU      ████████████░░░░░░░░  58.3% (16 cores)           │
│ Memory   ██████░░░░░░░░░░░░░░  32.5% (10.4GB/32.0GB)      │
│ GPU      ████████████████████  98.2% (78.5GB/80.0GB) 65°C │
╰─────────────────────────────────────────╯
⠋ Processing files... ━━━━━━━━━╸━━━━━━━  45/100 45% 0:01:23 0:01:41
```

## Integration with Pipelines

The monitoring package is used by the transcription, diarization, and other pipelines:

```python
# In transcribe_batch.py
from podcast_conversations.monitoring import ResourceDisplay, print_system_info

console = Console()
print_system_info(console)

with ResourceDisplay(console=console) as display:
    task = display.progress.add_task("transcribing", total=len(files))

    for file in files:
        process_file(file)
        display.progress.advance(task)

    display.print_summary()
```

## GPU Detection

GPU information is gathered in order of preference:

1. **PyTorch + pynvml**: Uses PyTorch for memory info, pynvml for utilization/temperature
2. **PyTorch only**: Memory from CUDA API (no utilization)
3. **pynvml only**: Full info from NVIDIA Management Library
4. **None**: No GPU detected

### Supported GPUs

- NVIDIA GPUs with CUDA support
- Requires `pynvml` (from `nvidia-ml-py`) for utilization and temperature

## Performance

- **Sampling Overhead**: <1ms per snapshot
- **Memory Usage**: ~1KB per snapshot (keeps last 100)
- **CPU Impact**: Negligible (<0.1% CPU)
- **Thread-Safe**: All operations are thread-safe

## Credits

Built with:
- [psutil](https://psutil.readthedocs.io/) - Cross-platform system monitoring
- [Rich](https://rich.readthedocs.io/) - Terminal UI
- [pynvml](https://pypi.org/project/nvidia-ml-py/) - NVIDIA GPU monitoring
