# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""A script to run multinode training with submitit.
Modified from: https://github.com/facebookresearch/deit
"""

import argparse
import logging
import os
import uuid
from pathlib import Path

import submitit

import main as classification


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser(
        "Submitit for ECAttention", parents=[classification_parser]
    )
    parser.add_argument(
        "--ngpus", default=2, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument(
        "--timeout", default=2880, type=int, help="Duration of the job in minutes"
    )
    parser.add_argument(
        "--job_dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--shared_folder",
        default="/nesi/nobackup/uoo03860/ECAttentionExperiments/",
        type=str,
        help="Shared folder for init file",
    )
    parser.add_argument(
        "--partition", default="genoa", type=str, help="Partition where to submit"
    )
    # parser.add_argument(
    #     "--use_volta32", action="store_true", help="Big models? Use this"
    # )
    parser.add_argument(
        "--GPU_type", default="H100", type=str, help="GPU type to use, e.g. A100, H100"
    )

    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    return parser.parse_args()


def get_shared_folder(args) -> Path:
    shared_folder = Path(args.shared_folder)
    # if shared_folder.is_dir():
    shared_folder.mkdir(parents=True, exist_ok=True)
    return shared_folder
    # raise RuntimeError("No shared folder available")


def get_init_file(args) -> Path:
    # Init file must not exist, but it's parent dir must exist.
    # os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    shared_folder = get_shared_folder(args)
    init_file = shared_folder / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os

        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path

        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder(args) / args.model / args.data_set / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    num_gpus_per_node_with_type = args.GPU_type + ":" + str(num_gpus_per_node)
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    # if args.use_volta32:
    #     kwargs["slurm_constraint"] = "volta32gb"
    if args.comment:
        kwargs["slurm_comment"] = args.comment

    executor.update_parameters(
        # mem_gb=40 * num_gpus_per_node,
        # mem_gb=10 * num_gpus_per_node, # for cifar data
        mem_gb=40 * num_gpus_per_node,  # for ImageNet data 1 GPU 1 Node
        # 32-64 GB for 12 Layers, 64-128 GB for 24 Layers #TODO: check this
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        # slurm_qos="debug",   # works as #SBATCH --qos=debug #TODO: check this
        # gpus_per_node=num_gpus_per_node,  # if use this, then slurm_gpus_per_node should not be set
        slurm_gpus_per_node=num_gpus_per_node_with_type,  # e.g. H100:2 #TODO: using this when run on NeSI
        **kwargs,
    )

    executor.update_parameters(
        slurm_setup=[
            "module --force purge",
            "module load NeSI",
            # "module load CUDA/12.2.2",
            # "module load NCCL/2.17.1-CUDA-12.0.0",
            # "module load cuDNN/8.9.7.29-CUDA-12.2.2",
            "module load CUDA/12.0.0",
            "module load NCCL/2.17.1-CUDA-12.0.0",
            "module load cuDNN/8.8.0.121-CUDA-12.0.0",
            "module load Miniconda3/23.10.0-1",
            "source $(conda info --base)/etc/profile.d/conda.sh",
            'eval "$(conda shell.bash hook)"',
            "conda activate /nesi/project/uoo03860/ECAttention_env",
            # "conda activate /nesi/project/uoo03860/tost",
            # "export NCCL_DEBUG=INFO",
            # "export TORCH_NCCL_TRACE_BUFFER_SIZE=1000000",
            # "export NCCL_DEBUG_SUBSYS=ALL",
            "export NCCL_TIMEOUT=1800000",
        ]
    )

    name = args.model + "_" + args.data_set + "_" + args.opt
    executor.update_parameters(name=name)

    args.dist_url = get_init_file(args).as_uri()
    # print("Using dist_url:", args.dist_url)
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()
