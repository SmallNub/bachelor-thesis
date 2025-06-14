import os
import sys
import psutil
import pynvml
import multiprocessing
import logging
import random
import torch
import numpy as np
from transformers import TrainerCallback


# Custom StreamHandler that always flushes
class FlushingStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


# Custom FileHandler that always flushes
class FlushingFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()


def init_logging(home_dir: str = None, job_id: str = None):
    """Initialize logging config."""
    handlers = [FlushingStreamHandler(sys.stdout)]

    # File logging is disabled outside SLURM
    if home_dir is not None and job_id is not None:
        log_file_path = f"{home_dir}/bachelor-thesis/logs/python_{job_id}.log"
        handlers.append(FlushingFileHandler(log_file_path, mode="w"))

    logging.basicConfig(
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        level=logging.INFO,
        handlers=handlers,
    )


def enable_tf32():
    """Lower precision for higher performance (negligible impact on results)"""
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def detect_resources(logger: logging.Logger):
    """Detect number of CPUs and GPUs."""
    num_cpus = int(os.getenv("SLURM_JOB_CPUS_PER_NODE", multiprocessing.cpu_count()))
    logger.info(f"Using {num_cpus} CPU core(s)")

    num_gpus = torch.cuda.device_count()
    logger.info(f"Using {num_gpus} CUDA device(s)")

    return num_cpus, num_gpus


def set_seed(seed: int):
    """Set the seed for rng for torch, numpy, random."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class ResourceMonitorCallback(TrainerCallback):
    """
    Monitors resources used on the system.\\
    Can be used with trainer and on its own.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

        self.process = psutil.Process(os.getpid())

        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.device_count)]

    def __del__(self):
        pynvml.nvmlShutdown()

    def format_gpu_info(self, i):
        """Get the GPU info and format it into a string"""
        handle = self.handles[i]
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        used_mb = mem.used / 1024**2
        total_mb = mem.total / 1024**2
        return f"GPU {i}: {used_mb:.2f}/{total_mb:.2f} MB, {util.gpu}% util"

    def log_resources(self):
        """Log the resources used. (RAM, CPU, GPU)"""
        # CPU Memory and usage
        rss = self.process.memory_info().rss / 1024**2  # in MB
        cpu_percent = self.process.cpu_percent(interval=None)  # % of single core
        cpu_total = psutil.cpu_percent(interval=None)     # overall system %
        cpu_line = f"CPU: {rss:.2f} MB, {cpu_percent:.1f}% (proc), {cpu_total:.1f}% (sys)"

        # GPU memory and utilization
        gpu_infos = [self.format_gpu_info(i) for i in range(self.device_count)]
        gpu_line = " | ".join(gpu_infos)

        self.logger.info(f"[MONITOR] {cpu_line} | {gpu_line}")

    def on_epoch_end(self, args, state, control, **kwargs):
        self.log_resources()
