# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""

# %%
import datetime
import gc
import io
import math
import os
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Union

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from einops import rearrange

print(mpl.get_cachedir())
plt.style.use(["default"])


home_dir = os.path.expanduser("~")
# print(f"Home directory: {home_dir}")

# font_path = Path(home_dir) / ".fonts" / "Times.TTF"
# print(f"Font path: {font_path}")

# my_font = fm.FontProperties(fname=font_path)
# plt.rcParams["font.family"] = my_font.get_name()

# print(f"current font: {my_font.get_name()}")
plt.rcParams["font.size"] = 10


def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility."""
    random.seed(seed)  # Set Python random seed
    np.random.seed(seed)  # Set NumPy random seed
    torch.manual_seed(seed)  # Set PyTorch random seed
    torch.cuda.manual_seed(seed)  # Set PyTorch CUDA random seed
    torch.cuda.manual_seed_all(seed)  # Set all PyTorch CUDA random seeds
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = (
        False  # Disable CuDNN auto-tuner for reproducibility
    )


class CholeskyStats:
    """Cholesky decomposition statistics tracker."""

    def __init__(self):
        self.total_calls = 0
        self.failures = 0

    def record_call(self, failed: bool = False):
        self.total_calls += 1
        if failed:
            self.failures += 1

    def get_failure_rate(self):
        if self.total_calls == 0:
            return 0.0
        return self.failures / self.total_calls

    def reset(self):
        self.total_calls = 0
        self.failures = 0

    def __str__(self):
        return f"Cholesky Stats: {self.failures}/{self.total_calls} failures ({self.get_failure_rate():.2%})"


cholesky_stats = CholeskyStats()


def cholesky_orthogonalization(Y: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Perform Cholesky orthogonalization on the input tensor.

    Args:
        Y (torch.Tensor): Input tensor of shape (b, d, k).
        eps (float): Small value to ensure numerical stability in the Cholesky decomposition.

    Returns:
        torch.Tensor: Orthogonalized tensor of shape (b, d, k).

    """
    is_batched = Y.dim() == 3
    if not is_batched:
        Y = Y.unsqueeze(0)

    orig_dtype = Y.dtype

    _, d, k = Y.shape
    Y_norm = nn.functional.normalize(Y, p=2, dim=-2)

    failed = False
    try:
        G = torch.matmul(Y_norm.transpose(-1, -2), Y_norm).float()
        eye = torch.eye(k, device=Y.device).unsqueeze(0)
        G = G + eps * eye
        L = torch.linalg.cholesky(G)
        LT = L.transpose(-1, -2)
        Q_coffs = torch.linalg.solve_triangular(LT, eye, upper=True)
        Q = torch.matmul(Y_norm, Q_coffs)
    except RuntimeError as e:
        failed = True
        print(f"Cholesky decomposition failed: {e}")
        Q, _ = torch.linalg.qr(Y_norm, mode="reduced")
    finally:
        cholesky_stats.record_call(failed)
        # if failed:
        #     print(cholesky_stats)

    if not is_batched:
        Q = Q.squeeze(0)
    Q = Q.to(orig_dtype)
    return Q

def cholesky_orthogonalization_QR(Y: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
    """Perform Cholesky orthogonalization on the input tensor.

    Args:
        Y (torch.Tensor): Input tensor of shape (b, d, k).
        eps (float): Small value to ensure numerical stability in the Cholesky decomposition.

    Returns:
        torch.Tensor: Orthogonalized tensor of shape (b, d, k).

    """
    is_batched = Y.dim() == 3
    if not is_batched:
        Y = Y.unsqueeze(0)

    orig_dtype = Y.dtype

    _, d, k = Y.shape
    Y_norm = nn.functional.normalize(Y, p=2, dim=-2)

    # failed = False
    # try:
    #     G = torch.matmul(Y_norm.transpose(-1, -2), Y_norm).float()
    #     eye = torch.eye(k, device=Y.device).unsqueeze(0)
    #     G = G + eps * eye
    #     L = torch.linalg.cholesky(G)
    #     LT = L.transpose(-1, -2)
    #     Q_coffs = torch.linalg.solve_triangular(LT, eye, upper=True)
    #     Q = torch.matmul(Y_norm, Q_coffs)
    # except RuntimeError as e:
        # failed = True
        # print(f"Cholesky decomposition failed: {e}")
    Q, _ = torch.linalg.qr(Y_norm, mode="reduced")
    # finally:
    #     cholesky_stats.record_call(failed)
    #     # if failed:
        #     print(cholesky_stats)

    if not is_batched:
        Q = Q.squeeze(0)
    Q = Q.to(orig_dtype)
    return Q


def generate_gaussian_data(
    K: int = 6,
    n: int = 196,
    d: int = 384,
    p: int = 64,
    sigma: float = 0.0,
    actual_dim: int = 10,
    device: Union[str, torch.device] = "cuda",
) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
    """Generate Gaussian data in a subspace of dimension `actual_dim`.

    Args:
        K (int): Number of subspaces.
        n (int): Number of samples.
        d (int): Dimension of the data.
        p (int): Dimension of each subspace.
        sigma (float): Standard deviation of the Gaussian noise to be added.
        actual_dim (int): Actual dimension within the subspace.
        device (Union[str, torch.device]): Device to run the computation on (default is "cuda").

    Returns:
        tuple: A tuple containing:
            - Z (torch.Tensor): Generated data of shape (d, n).
            - U (torch.Tensor): Orthogonal matrix of shape (K, d, p).
            - s (torch.Tensor): Subspace indices of shape (n,).

    """
    with torch.no_grad():
        # K: number of subspaces
        # U: orthogonal matrix of shape (d, K * p)
        U = torch.nn.init.orthogonal_(torch.empty(d, K * p, device=device))
        # U is reshaped to (K, d, p) for K subspaces
        U = rearrange(U, "d (k p) -> k d p", p=p)
        # s_i ∈ {0, 1, ..., K-1}
        s = torch.randint(0, K, (n,), device=device)

        # Gaussion coefficients alpha_i ∈ R^(n, p)
        alpha = torch.zeros(n, p, device=device)  # initialize alpha with zeros
        alpha[:, :actual_dim] = torch.randn(
            n, actual_dim, device=device
        )  # fill the first `actual_dim` columns with random values

        # Z_i = U[s[i]] @ alpha[i]
        Z = torch.zeros(n, d, device=device)
        for i in range(n):
            Z[i] = torch.matmul(U[s[i]], alpha[i])

        # Add Gaussian noise to the data if sigma > 0
        if sigma > 0:
            noise = torch.normal(0, sigma, size=(n, d), device=device)
            Z = Z + noise

        # Calculate the rank of each subspace
        dot = torch.matmul(Z.t(), Z)
        rank = torch.linalg.matrix_rank(dot)
        print(f"The Whole space data rank: {rank}")
        for k in range(K):
            Z_k = Z[s == k]
            # print(f"Subspace {k} generated data shape: {Z_k.shape}")
            dot = torch.matmul(Z_k.t(), Z_k)
            rank = torch.linalg.matrix_rank(dot)
            print(f"subspace {k} generated data rank: {rank}")
            del Z_k, dot
        print(20 * "#" + "\n")
        return Z.t(), U, s


def print_gpu_memory() -> None:
    """Print the GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        allocated = torch.cuda.memory_allocated(0)
        free = total_memory - reserved
        print("====================\n")
        print(f"Total GPU Memory: {total_memory / (1024**3):.2f} GB")
        print(f"Reserved Memory: {reserved / (1024**3):.2f} GB")
        print(f"Allocated Memory: {allocated / (1024**3):.2f} GB")
        print(f"Free Memory: {free / (1024**3):.2f} GB")
        print("====================\n")


def plot_loss_vs_epoch(
    loss_history: list, lossofR: bool = True, CEloss: bool = False
) -> None:
    """Plot the loss (or objective function) curve over training epochs.

    Args:
        loss_history: List or NumPy array containing loss values for each epoch.
                     For example, if neg_coding_rate was recorded during training loop.

        lossofR (bool): If True, the title will indicate that the loss is the negative
                        coding rate of the whole feature space. Default is True.
                        Otherwise, it will indicate that the loss is the coding rate of
                        the subspace.

    """
    if not loss_history:
        print("Warning: Loss history is empty, cannot plot graph.")
        return

    if CEloss:
        title = "Training Loss Over Epochs"
        ylabel = "Cross Entropy Loss"
    elif lossofR:
        title = "Training Loss Over Epochs"
        ylabel = "Coding Rate of $R$"
        loss_history = [-loss for loss in loss_history]  # Negate if it's -R
    else:
        title = "Training Loss Over Epochs"
        ylabel = "Coding Rate of Subspaces $R_c$"

    epochs = range(1, len(loss_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(
        epochs,
        loss_history,
        marker="o",
        linestyle="-",
        label="CE" if CEloss else ("R" if lossofR else "Rc"),
    )
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    # Show all epoch ticks if the number of epochs is not too large
    if len(epochs) <= 50:
        plt.xticks(epochs)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()  # Close the plot to free memory


def plot_metric_vs_layer(
    layer_metrics: Union[list, np.ndarray],
    metric_name: str = "Coding Rate of $R$",
    title: str = "Coding Rate Changes Across Layers",
):
    """Plot how a metric (e.g., coding rate R) changes after processing through each layer of the model.

    Args:
        layer_metrics: List or NumPy array containing metric values computed after each layer.

        metric_name: Name of the metric, used for legend and Y-axis label.

        title: Title of the plot.

    """
    if len(layer_metrics) == 0:
        print("Warning: Layer metrics record is empty, cannot plot graph.")
        return

    layers = range(len(layer_metrics))
    plt.figure(figsize=(10, 6))
    # plt.plot(layers, layer_metrics, marker="s", linestyle="--", label=f"{metric_name}")
    plt.plot(layers, layer_metrics, label=f"{metric_name}")
    ax = plt.gca()  # Get current axes
    ax.spines["right"].set_color("none")
    ax.spines["top"].set_color("none")
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", right=False)
    plt.xlabel("Layer")
    plt.ylabel(f"{metric_name}")
    plt.title(title)
    if len(layers) <= 50:
        plt.xticks(layers)
    # plt.xticks(layers) # Show all layer indices
    # plt.grid(True)
    plt.grid(color="lightgray", linestyle="-", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    # figure_folder = Path("../figures")
    # figure_folder.mkdir(parents=True, exist_ok=True)
    # save_path = figure_folder / f"layer_metric_plot_{metric_name}.pdf"
    # plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.show()
    # return (
    #     plt.gcf()
    # )  # Return the current figure object for further manipulation if needed
    plt.close()  # Close the plot to free memory


def compare_eigenvalues(
    Z_before,
    Z_after,
    plot=True,
    use_cpu=True,
):
    """Compare eigenvalue distributions of Z matrices before and after MSSA.

    Args:
        Z_before: Z matrix before transformation (e.g., before MSSA)
        Z_after: Final Z matrix after LayerNorm
        plot: Whether to plot graphs
        use_cpu: Whether to move to CPU for eigenvalue computation (to reduce GPU memory usage)

    Returns:
        Dictionary containing eigenvalue statistics

    """
    with torch.no_grad():
        # Calculate eigenvalues
        stats = {"rank": {}, "condition": {}, "max": {}, "min_positive": {}}

        try:
            # Process each matrix sequentially to reduce peak memory usage
            # Process Z_before
            dot_before = torch.matmul(Z_before, Z_before.t())
            if use_cpu and dot_before.device.type == "cuda":
                dot_before_cpu = dot_before.cpu()
                eigenvalues_before = torch.linalg.eigvalsh(dot_before_cpu)
                rank_before = torch.linalg.matrix_rank(dot_before_cpu).item()
                del dot_before_cpu
            else:
                eigenvalues_before = torch.linalg.eigvalsh(dot_before)
                rank_before = torch.linalg.matrix_rank(dot_before).item()

            # Save statistics
            stats["rank"]["before"] = rank_before
            stats["max"]["before"] = eigenvalues_before.max().item()

            # Safely compute condition number (handle possible zero eigenvalues)
            ev_before_pos = eigenvalues_before[eigenvalues_before > 0]
            if len(ev_before_pos) > 0:
                stats["min_positive"]["before"] = ev_before_pos.min().item()
                stats["condition"]["before"] = (
                    eigenvalues_before.max() / ev_before_pos.min()
                ).item()
            else:
                stats["min_positive"]["before"] = float("nan")
                stats["condition"]["before"] = float("inf")

            # Save CPU eigenvalues for subsequent plotting
            if plot:
                eigenvalues_before_cpu = (
                    eigenvalues_before.cpu()
                    if eigenvalues_before.device.type == "cuda"
                    else eigenvalues_before
                )

            # Release memory
            del dot_before, eigenvalues_before, ev_before_pos
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Process Z_after
            dot_after = torch.matmul(Z_after, Z_after.t())
            if use_cpu and dot_after.device.type == "cuda":
                dot_after_cpu = dot_after.cpu()
                eigenvalues_after = torch.linalg.eigvalsh(dot_after_cpu)
                rank_after = torch.linalg.matrix_rank(dot_after_cpu).item()
                del dot_after_cpu
            else:
                eigenvalues_after = torch.linalg.eigvalsh(dot_after)
                rank_after = torch.linalg.matrix_rank(dot_after).item()

            # Save statistics
            stats["rank"]["after"] = rank_after
            stats["max"]["after"] = eigenvalues_after.max().item()

            # Safely compute condition number
            ev_after_pos = eigenvalues_after[eigenvalues_after > 0]
            if len(ev_after_pos) > 0:
                stats["min_positive"]["after"] = ev_after_pos.min().item()
                stats["condition"]["after"] = (
                    eigenvalues_after.max() / ev_after_pos.min()
                ).item()
            else:
                stats["min_positive"]["after"] = float("nan")
                stats["condition"]["after"] = float("inf")

            if plot:
                eigenvalues_after_cpu = (
                    eigenvalues_after.cpu()
                    if eigenvalues_after.device.type == "cuda"
                    else eigenvalues_after
                )

            # Release memory
            del dot_after, eigenvalues_after, ev_after_pos
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Visualization - separate plots
            if plot:
                # 1. Eigenvalue distribution curves
                plt.figure(figsize=(10, 6))
                plt.plot(
                    eigenvalues_before_cpu.numpy(), label="Before ", linestyle="--"
                )
                plt.plot(
                    eigenvalues_after_cpu.numpy(),
                    label="Final Result",
                    linestyle="solid",
                )
                plt.xlabel("Index")
                plt.ylabel("Eigenvalue")
                plt.title("Eigenvalue Distribution Comparison")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.close()

                # 2. Eigenvalue logarithmic distribution
                plt.figure(figsize=(10, 6))
                plt.semilogy(
                    eigenvalues_before_cpu.numpy(), label="Before", linestyle="--"
                )
                plt.semilogy(
                    eigenvalues_after_cpu.numpy(),
                    label="Final Result",
                    linestyle="solid",
                )
                plt.xlabel("Index")
                plt.ylabel("Eigenvalue (Log Scale)")
                plt.title("Eigenvalue Logarithmic Distribution")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.close()

                # 3. Cumulative energy distribution
                plt.figure(figsize=(10, 6))
                linestyle = ["--", "-.", "-"]
                for (name, ev), style in zip(
                    {
                        "Before MSSA": eigenvalues_before_cpu,
                        "Final Result": eigenvalues_after_cpu,
                    }.items(),
                    linestyle,
                    strict=False,
                ):
                    ev_sorted = torch.sort(ev, descending=True)[0]
                    ev_cumsum = torch.cumsum(ev_sorted, dim=0) / ev_sorted.sum()
                    plt.plot(ev_cumsum.numpy(), label=name, linestyle=style)
                plt.xlabel("Eigenvalue Index (Sorted by Size)")
                plt.ylabel("Cumulative Energy Ratio")
                plt.title("Eigenvalue Energy Distribution")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
                plt.close()
        except Exception as e:
            print(f"Error occurred while comparing eigenvalues: {e}")
            # Ensure partial statistics are returned even if an error occurs
            for phase in ["before", "after"]:
                for stat_type in ["rank", "condition", "max", "min_positive"]:
                    if phase not in stats.get(stat_type, {}):
                        stats.setdefault(stat_type, {})[phase] = float("nan")

        return stats


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Warning: does not synchronize the deque!"""
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = [
            header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            "{} Total time: {} ({:.4f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            )
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """Workaround for ModelEma._load_checkpoint to accept an already-loaded object"""
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """This function disables printing when not in master process"""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_distributed_mode(args):
    # print(40 * "#")
    # print(args)
    # print(40 * "#")
    # print(os.environ)

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
        # print(40 * "=GPU")
        # print(args.gpu)
        # print(40 * "=GPU")

        # print(40 * "=")
        # print(
        #     torch.cuda.device_count(),
        #     "GPUs detected (rank {} on {} GPUs)".format(
        #         args.rank, torch.cuda.device_count()
        #     ),
        # )
        # print(40 * "=")
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        #  device_id=args.gpu
        device_id=torch.device(f"cuda:{args.gpu}"),
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


try:
    _, term_width = os.popen("stty size", "r").read().split()
except:
    term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.0
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(" [")
    for i in range(cur_len):
        sys.stdout.write("=")
    sys.stdout.write(">")
    for i in range(rest_len):
        sys.stdout.write(".")
    sys.stdout.write("]")

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append("  Step: %s" % format_time(step_time))
    L.append(" | Tot: %s" % format_time(tot_time))
    if msg:
        L.append(" | " + msg)

    msg = "".join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(" ")

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write("\b")
    sys.stdout.write(" %d/%d " % (current + 1, total))

    if current < total - 1:
        sys.stdout.write("\r")
    else:
        sys.stdout.write("\n")
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ""
    i = 1
    if days > 0:
        f += str(days) + "D"
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + "h"
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + "m"
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + "s"
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + "ms"
        i += 1
    if f == "":
        f = "0ms"
    return f


def check_gradients(model):  # do not check gradients after optimizer.zero_grad()!!!
    unused_params = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_params.append(name)

    if unused_params:
        print(f"Unused parameters: {unused_params}")
        # print shapes and requires_grad status
        for name in unused_params:
            param = dict(model.named_parameters())[name]
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
    else:
        print("All parameters have gradients.")


def increase_rank(model, new_subspace_rank, ECAttn):
    for name, module in model.named_modules():
        if isinstance(module, ECAttn):
            old_rank = module.subspace_rank
            print(f"==> {name}: {old_rank} --> {new_subspace_rank}")
            module.subspace_rank = new_subspace_rank
            module.expandoperator.rank_of_space = new_subspace_rank * module.num_heads
            module.compressoperator.subspace_rank = new_subspace_rank

            assert new_subspace_rank * module.num_heads <= module.dim, (
                f"Error in increasing rank:  "
                f"heads {module.heads}, old rank {old_rank}, new rank {new_subspace_rank}, dim {module.dim}"
            )
            module.expandoperator._regenerate_random_matrices(
                module.expandoperator.default_n
            )
            module.compressoperator._regenerate_random_matrix(
                module.compressoperator.default_n
            )
    return model


# %%
if __name__ == "__main__":
    # Example usage of the utility functions
    set_seed(42)

    # Plotting example
    # plot_loss_vs_epoch([0.1, 0.05, 0.02], lossofR=True)
    plot_metric_vs_layer(
        layer_metrics=[0.1, 0.08, 0.05, 0.03],
        metric_name="Coding Rate of $R$",
        title="Coding Rate Changes Across Layers",
    )
# %%
