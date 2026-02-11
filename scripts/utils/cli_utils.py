"""CLI utility functions shared across pipeline scripts."""

import os


def get_default_workers(memory_per_worker_gb: float = 0.1, cpu_fraction: float = 0.8) -> int:
    """
    Get default number of workers based on available resources.

    Considers:
    - SLURM_CPUS_PER_TASK environment variable (HPC)
    - PBS_NUM_PPN environment variable (HPC)
    - Available CPU cores
    - Available system memory

    Args:
        memory_per_worker_gb: Estimated memory per worker in GB (default: 0.1 GB)
        cpu_fraction: Fraction of CPU cores to use (default: 0.8)

    Returns:
        int: Recommended number of workers
    """
    # check for SLURM allocation first (HPC environments).
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    # check for PBS/Torque allocation.
    pbs_cpus = os.environ.get("PBS_NUM_PPN")
    if pbs_cpus:
        try:
            return max(1, int(pbs_cpus))
        except ValueError:
            pass

    # fallback to system detection.
    cpu_count = os.cpu_count() or 1

    # check available memory.
    try:
        import psutil

        available_mem_gb = psutil.virtual_memory().available / (1024**3)
        # allow specified memory per worker, with 2GB reserved for system.
        mem_based_workers = max(1, int((available_mem_gb - 2) / memory_per_worker_gb))
    except ImportError:
        # psutil not available, assume enough memory.
        mem_based_workers = cpu_count

    # use specified fraction of available cores.
    cpu_based_workers = max(1, int(cpu_count * cpu_fraction))

    # use the minimum of CPU and memory constraints.
    return min(cpu_based_workers, mem_based_workers)


def resolve_log_level(log_level: str, verbose: bool, quiet: bool) -> str:
    """
    Resolve log level from flags.

    Args:
        log_level: Explicit log level (DEBUG, INFO, WARNING, ERROR)
        verbose: If True, use DEBUG level
        quiet: If True, use WARNING level

    Returns:
        str: Resolved log level (verbose takes precedence over quiet)
    """
    if verbose:
        return "DEBUG"
    if quiet:
        return "WARNING"
    return log_level
