import torch
from hybridiff.pipelines.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline
import torch.distributed as dist
from hybridiff.hybrid_sd3 import HybridDiff
from hybridiff.metrics import get_gpu_memory, memory_callback
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers')     
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says hello world')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    parser.add_argument("--image_size", type=int, default=1024, help="Image size of generation")
    
    # HybridDiff parameters
    parser.add_argument("--L_slope", type=int, default=15, help="Window size for tau1 slope calculation")
    parser.add_argument("--eps_slope", type=float, default=0.5e-3, help="Threshold for tau1 detection")
    parser.add_argument("--k", type=int, default=5, help="Offset for tau2 (tau2 = tau1 + k)")
    parser.add_argument("--tau_cap", type=int, default=40, help="Safety cap for tau1 detection")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    pipeline = StableDiffusion3Pipeline.from_pretrained(
            args.model, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True, low_cpu_mem_usage=True,
            L_slope=args.L_slope, eps_slope=args.eps_slope,
            k=args.k, tau_cap=args.tau_cap
    )
    hybrid_diff = HybridDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hybrid_diff.reset_state(warm_up=1)
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=50,
        height=args.image_size,
        width=args.image_size,
        guidance_scale=7.0,
    ).images[0]
    
    memory_before = get_gpu_memory()
    print("GPU Memory Usage Before Execution:", memory_before)

    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    hybrid_diff.reset_state(warm_up=1)
    start_second = time.time()
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=50,
        height=args.image_size,
        width=args.image_size,
        guidance_scale=7.0,
    ).images[0]
    print(f"[second] Rank {dist.get_rank()} Time taken: {time.time()-start_second:.2f} seconds.")
    
    # GPU Memory after inference
    memory_after = get_gpu_memory()
    print("GPU Memory Usage After Execution:", memory_after)

    if dist.get_rank() == 0:
        image.save(f"output-{args.model_n}-{args.stride}-{args.image_size}-sd3-hybridiff.png")
        print("✅ Image saved as output-size-sd3-hybridiff.png")