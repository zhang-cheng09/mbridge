# Example to use tp/pp/cp/vpp to test dense model
# torchrun --nproc_per_node=8 example/internvl3/load_model_and_forward.py --model_path /path/to/model

import argparse

import torch
from megatron.core import parallel_state as mpu
from megatron.core.tensor_parallel.mappings import (
    gather_from_tensor_model_parallel_region,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from transformers import AutoModel

from mbridge import AutoBridge


def loss_func(output_tensor: torch.Tensor):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    losses = output_tensor.float()
    loss = torch.stack([torch.sum(losses.view(-1)).view(1), losses.sum().view(1)])

    return loss[0] / loss[1]


def init_distributed(tp=2, pp=1, cp=1, vpp=1, ep=1, etp=None):
    """Initialize distributed environment"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(torch.distributed.get_rank())
    if pp <= 1:
        vpp = None
    mpu.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vpp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=etp,
    )
    model_parallel_cuda_manual_seed(0)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Load model and generate text")
    parser.add_argument(
        "--model_path", type=str, required=True, help="HuggingFace model path"
    )
    parser.add_argument("--tp", type=int, default=2, help="Tensor model parallel size")
    parser.add_argument(
        "--pp", type=int, default=1, help="Pipeline model parallel size"
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size")
    parser.add_argument(
        "--vpp", type=int, default=1, help="Virtual pipeline model parallel size"
    )
    parser.add_argument("--ep", type=int, default=1, help="Expert model parallel size")
    parser.add_argument(
        "--etp", type=int, default=None, help="Expert tensor parallel size"
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="Path to save weights"
    )
    args = parser.parse_args()

    # Initialize distributed environment
    init_distributed(
        tp=args.tp,
        pp=args.pp,
        cp=args.cp,
        vpp=args.vpp,
        ep=args.ep,
        etp=args.etp,
    )

    # Load megatron model
    hf_model_path = args.model_path
    print(f"rank{torch.distributed.get_rank()}: start loading model ...")
    bridge = AutoBridge.from_pretrained(
        hf_model_path, trust_remote_code=True, make_vocab_size_divisible_by=256
    )
    # set sequence_parallel = False for forward
    bridge.config.sequence_parallel = False
    model = bridge.get_model()
    bridge.load_weights(model, hf_model_path, memory_efficient=True)
    print(f"rank{torch.distributed.get_rank()}: end load weight, start forward ...")

    # check the export
    keys = bridge.safetensor_io.load_hf_weight_names()
    loaded_keys = set()
    # export weights
    for k, v in bridge.export_weights(model):
        gt = bridge.safetensor_io.load_one_hf_weight(k).cuda()
        assert v.shape == gt.shape, f"mismatch of {k}"
        assert torch.equal(v, gt), f"mismatch of {k}"
        loaded_keys.add(k)

    missing_keys = set(keys) - loaded_keys
    missing_keys = sorted(list(missing_keys))
    assert len(missing_keys) == 0

    device = torch.cuda.current_device()
    images = torch.load("./example/nan_demo/images.pt").to(device)

    with torch.autograd.detect_anomaly():
        # 参见mbridge/models/internvl3/model.py中InternVLModel的forward
        megatron_output = model[0](
            images=images,
            input_ids=None,
            position_ids=None,
            attention_mask=None,
            labels=None,
            image_token_index=151667,
        )
        loss = loss_func(megatron_output)
        loss.backward()

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
