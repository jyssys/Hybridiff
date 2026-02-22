import argparse
import os
import tqdm
import aiohttp
import torch
import torch.distributed as dist

import numpy as np
import matplotlib.pyplot as plt

from datasets import load_dataset
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import trange

from hybridiff.pipelines.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
from hybridiff.hybrid_sd3 import HybridDiff

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # Diffuser specific arguments
    parser.add_argument("--output_root", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of inference steps")
    parser.add_argument("--image_size", type=int, nargs="*", default=1024, help="Image size of generation")
    parser.add_argument("--guidance_scale", type=float, default=7.0)

    # HybridDiff specific arguments
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_n", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--time_shift", type=bool, default=False)
    
    # HybridDiff parameters
    parser.add_argument("--L_slope", type=int, default=15)
    parser.add_argument("--eps_slope", type=float, default=0.5e-3)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--tau_cap", type=int, default=40)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    dist.init_process_group(backend="nccl")

    if isinstance(args.image_size, int):
        args.image_size = [args.image_size, args.image_size]
    else:
        if len(args.image_size) == 1:
            args.image_size = [args.image_size[0], args.image_size[0]]
        else:
            assert len(args.image_size) == 2

    pretrained_model_name_or_path = args.model
    
    # SD3 uses FlowMatchEulerDiscreteScheduler
    # scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    #     pretrained_model_name_or_path, 
    #     subfolder="scheduler"
    # )
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        L_slope=args.L_slope, eps_slope=args.eps_slope,
        k=args.k, tau_cap=args.tau_cap
    )
    hybrid_diff = HybridDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)
    
    if args.output_root is None:
        args.output_root = os.path.join(
            f"results-sd3-hybridiff-{args.model_n}gpu-{args.stride}stride",
            "coco",
            f"flow-match-euler-{args.num_inference_steps}",
        )
    if dist.get_rank() == 0:
        os.makedirs(args.output_root, exist_ok=True)

    dataset = load_dataset("HuggingFaceM4/COCO", name="2014_captions", split="validation", trust_remote_code=True, cache_dir="/data/ms-coco",
                           storage_options={'client_kwargs':{'timeout':aiohttp.ClientTimeout(total=72000)}})

    
    start_idx = 0
    end_idx = 5000

    for i in trange(start_idx, end_idx, disable=dist.get_rank() != 0, position=0, leave=False):
        prompt = dataset["sentences_raw"][i][i % len(dataset["sentences_raw"][i])]
        seed = i
        
        print(f"Processing image {i}/{end_idx-start_idx}...")
        print(f"Prompt: {prompt}")

        # warm up
        hybrid_diff.reset_state(warm_up=1)
        image = pipeline(
            prompt=prompt,
            height=args.image_size[0],
            width=args.image_size[1],
            generator=torch.Generator(device="cuda").manual_seed(seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images[0]
        
        # inference
        hybrid_diff.reset_state(warm_up=1)
        image = pipeline(
            prompt=prompt,
            height=args.image_size[0],
            width=args.image_size[1],
            generator=torch.Generator(device="cuda").manual_seed(seed),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images[0]
        
        if dist.get_rank() == 0:
            output_path = os.path.join(args.output_root, f"{i:04d}.png")
            image.save(output_path)
            
            del image
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()