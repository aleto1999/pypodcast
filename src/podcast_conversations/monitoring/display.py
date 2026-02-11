"""rich display components for resource monitoring."""

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.table import Table
from rich.text import Text

from podcast_conversations.monitoring.resources import (
    ResourceMonitor,
    ResourceSnapshot,
    get_resource_snapshot,
    get_system_info,
)


def format_memory(gb: float) -> str:
    """format memory in appropriate units."""
    if gb < 1:
        return f"{gb * 1024:.0f}MB"
    return f"{gb:.1f}GB"


def get_usage_color(percent: float) -> str:
    """get color based on usage percentage."""
    if percent < 50:
        return "green"
    elif percent < 80:
        return "yellow"
    else:
        return "red"


def create_resource_panel(snapshot: ResourceSnapshot | None = None) -> Panel:
    """
    create a rich panel showing current resource usage.

    args:
        snapshot: resource snapshot to display (fetches current if None).

    returns:
        rich Panel with resource information.
    """
    if snapshot is None:
        snapshot = get_resource_snapshot()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", width=12)
    table.add_column(width=40)

    # cpu row.
    cpu_color = get_usage_color(snapshot.cpu_percent)
    cpu_bar = _create_bar(snapshot.cpu_percent, 20)
    table.add_row(
        "CPU",
        Text.assemble(
            (cpu_bar, cpu_color),
            f" {snapshot.cpu_percent:5.1f}% ",
            (f"({snapshot.cpu_count} cores)", "dim"),
        ),
    )

    # memory row.
    mem_color = get_usage_color(snapshot.memory_percent)
    mem_bar = _create_bar(snapshot.memory_percent, 20)
    table.add_row(
        "Memory",
        Text.assemble(
            (mem_bar, mem_color),
            f" {snapshot.memory_percent:5.1f}% ",
            (f"({format_memory(snapshot.memory_used_gb)}/{format_memory(snapshot.memory_total_gb)})", "dim"),
        ),
    )

    # gpu rows.
    for gpu in snapshot.gpus:
        gpu_mem_color = get_usage_color(gpu.memory_percent)
        gpu_bar = _create_bar(gpu.memory_percent, 20)

        gpu_label = f"GPU {gpu.index}" if len(snapshot.gpus) > 1 else "GPU"

        temp_str = f" {gpu.temperature_c}°C" if gpu.temperature_c else ""
        util_str = f" {gpu.utilization_percent:.0f}% util" if gpu.utilization_percent > 0 else ""

        table.add_row(
            gpu_label,
            Text.assemble(
                (gpu_bar, gpu_mem_color),
                f" {gpu.memory_percent:5.1f}% ",
                (f"({format_memory(gpu.memory_used_gb)}/{format_memory(gpu.memory_total_gb)})", "dim"),
                (temp_str, "yellow" if gpu.temperature_c and gpu.temperature_c > 70 else "dim"),
                (util_str, "dim"),
            ),
        )

    return Panel(table, title="[bold]Resources[/bold]", border_style="blue", padding=(0, 1))


def _create_bar(percent: float, width: int = 20) -> str:
    """create a text-based progress bar."""
    filled = int(percent / 100 * width)
    empty = width - filled
    return "█" * filled + "░" * empty


class ResourceDisplay:
    """
    live resource display that updates alongside progress bars.

    usage:
        with ResourceDisplay() as display:
            with display.progress as progress:
                task = progress.add_task("processing", total=100)
                for i in range(100):
                    progress.advance(task)
    """

    def __init__(
        self,
        console: Console | None = None,
        refresh_rate: float = 2.0,
        show_resources: bool = True,
    ):
        """
        initialize resource display.

        args:
            console: rich console to use.
            refresh_rate: how often to refresh display (seconds).
            show_resources: whether to show resource panel.
        """
        self.console = console or Console()
        self.refresh_rate = refresh_rate
        self.show_resources = show_resources
        self._monitor: ResourceMonitor | None = None
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._latest_snapshot: ResourceSnapshot | None = None

    @property
    def progress(self) -> Progress:
        """get the progress bar instance."""
        if self._progress is None:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=4,
            )
        return self._progress

    def _make_renderable(self) -> Group:
        """create the combined renderable for live display."""
        components = []

        if self.show_resources and self._latest_snapshot:
            components.append(create_resource_panel(self._latest_snapshot))

        if self._progress:
            components.append(self._progress)

        return Group(*components)

    def _on_snapshot(self, snapshot: ResourceSnapshot) -> None:
        """callback when new resource snapshot is available."""
        self._latest_snapshot = snapshot
        if self._live:
            self._live.update(self._make_renderable())

    def start(self) -> None:
        """start the live display."""
        # start resource monitoring.
        if self.show_resources:
            self._monitor = ResourceMonitor(
                interval=self.refresh_rate,
                callback=self._on_snapshot,
            )
            self._monitor.start()
            # get initial snapshot.
            self._latest_snapshot = get_resource_snapshot()

        # start live display.
        self._live = Live(
            self._make_renderable(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

        if self._monitor:
            self._monitor.stop()
            self._monitor = None

    def update(self) -> None:
        """manually trigger display update."""
        if self._live:
            self._live.update(self._make_renderable())

    def print_summary(self) -> None:
        """print resource usage summary after processing."""
        if self._monitor:
            avg = self._monitor.get_average()
            if avg:
                self.console.print()
                self.console.print("[bold]Resource Usage Summary:[/bold]")
                self.console.print(f"  • Average CPU: {avg.cpu_percent:.1f}%")
                self.console.print(f"  • Average Memory: {avg.memory_percent:.1f}%")
                for gpu in avg.gpus:
                    self.console.print(
                        f"  • Average GPU {gpu.index} Memory: {gpu.memory_percent:.1f}%"
                    )

    def __enter__(self) -> "ResourceDisplay":
        """context manager entry."""
        self.start()
        return self

    def __exit__(self, *args) -> None:
        """context manager exit."""
        self.stop()


def print_system_info(console: Console | None = None) -> None:
    """print system information header."""
    console = console or Console()
    info = get_system_info()

    console.print()
    console.print("[bold cyan]System Information[/bold cyan]")
    console.print(f"  • CPU: {info['cpu_count_physical']} cores ({info['cpu_count']} threads)")
    console.print(f"  • Memory: {info['memory_total_gb']:.1f} GB")

    for gpu in info["gpus"]:
        console.print(f"  • GPU {gpu['index']}: {gpu['name']} ({gpu['memory_gb']:.0f} GB)")

    if not info["gpus"]:
        console.print("  • GPU: [dim]None detected[/dim]")

    console.print()
