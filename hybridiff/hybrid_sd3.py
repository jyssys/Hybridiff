import torch.distributed as dist
import torch
from .tools import ResultPicker
from .pipe_config import splite_model

class ModulePlugin(object):
    def __init__(self, module, model_i, stride=1, run_mode=None) -> None:
        self.model_i, self.stride, self.run_mode = model_i, stride, run_mode
        self.module = module
        self.module.plugin = self
        self.init_state()
        self.inject_forward()
        self.rank = dist.get_rank()
        self.pp_enabled = True  # PP can be dynamically enabled/disabled

    def init_state(self, warmup_n=1):
        self.warmup_n = warmup_n
        self.result_structure, self.cached_result = None, None
        self.infer_step = 0
    
    def cache_sync(self, async_flag):
        """Broadcast cached activation to other ranks"""
        if self.cached_result is None:
            return
        if self.infer_step >= self.warmup_n:
            dist.broadcast(self.cached_result, self.model_i, async_op=async_flag)

    def inject_forward(self):
        assert not hasattr(self.module, 'old_forward'), "Module already has old_forward attribute."
        module = self.module
        module.old_forward = module.forward
        
        def new_forward(*args, **kwargs):
            # If PP is disabled, always execute locally
            if not self.pp_enabled:
                self.infer_step += 1
                return module.old_forward(*args, **kwargs)
            
            # PP logic: execute locally or use cached
            run_locally = (self.run_mode[0] == self.model_i) and ((self.infer_step-1) % self.stride == 0)

            # Execute if: before warmup, run_locally, OR cached_result is None
            if self.infer_step < self.warmup_n or run_locally or self.cached_result is None:
                # Execute this block
                result = module.old_forward(*args, **kwargs)
                # Cache result for potential reuse
                if (self.infer_step+1 == self.warmup_n) or \
                   (self.infer_step + 1 > self.warmup_n and self.run_mode[1] == 0):
                    self.cached_result, self.result_structure = ResultPicker.dump(result)
            else:
                # Use cached result
                result = ResultPicker.load(self.cached_result, self.result_structure)
            
            self.infer_step += 1
            return result
        
        module.forward = new_forward

class HybridDiff(object):
    def __init__(self, pipeline, model_n=2, stride=1, warm_up=1, time_shift=False):
        # dist.init_process_group("nccl")
        if not dist.get_rank(): assert model_n + stride - 1 == dist.get_world_size(), "[ERROR]: The strategy is not compatible with the number of devices. (model_n + stride - 1) should be equal to world_size."
        assert stride==1 or stride==2, "[ERROR]: The stride should be set as 1 or 2"
        self.model_n = model_n
        self.stride = stride
        self.warm_up = warm_up
        self.time_shift = time_shift
        self.pipeline = pipeline.to(f"cuda:{dist.get_rank()}")
        torch.cuda.set_device(f"cuda:{dist.get_rank()}")
        self.pipe_id = pipeline.config._name_or_path
        self.reformed_modules = {}
        self.reform_pipeline()
        step = 24 // model_n
        self.comm_index = [(i + 1) * step for i in range(model_n - 1)]

    def reset_state(self,warm_up=1):
        self.warm_up = warm_up
        for each in self.reformed_modules.values():
            each.plugin.init_state(warmup_n=warm_up)

    def reform_module(self, module, module_id, model_i):
        run_mode = (dist.get_rank(), 0) if dist.get_rank() < self.model_n else (self.model_n -1, 1)
        ModulePlugin(module, model_i, self.stride, run_mode)
        self.reformed_modules[(model_i, module_id)] = module
    
    def reform_transformer(self):
        transformer = self.pipeline.transformer
        assert not hasattr(transformer, 'old_forward'), "transformer already has old_forward attribute."
        transformer.old_forward = transformer.forward

        def transformer_forward(*args, **kwargs):
            rank = dist.get_rank()
            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            
            # ===== Sync tau1 & tau2 from pipeline =====
            if not hasattr(self, "_tau_buf"):
                device = kwargs['hidden_states'].device
                self._tau_buf = torch.full((2,), -1, dtype=torch.int32, device=device)
            
            if rank == 0:
                self._tau_buf.copy_(self.pipeline.tau_tensor.to(torch.int32))
            dist.broadcast(self._tau_buf, src=0)
            tau1_sync, tau2_sync = map(int, self._tau_buf.tolist())
            
            # Update warmup if tau1 changed
            if tau1_sync >= 0 and tau1_sync != self.warm_up:
                self.warm_up = tau1_sync
                # Update warmup_n but keep cached results
                for mod in self.reformed_modules.values():
                    mod.plugin.warmup_n = tau1_sync
            
            # Determine if PP should be active
            if tau1_sync < 0 or infer_step < tau1_sync:
                pp_active = False
            elif tau2_sync < 0 or infer_step <= tau2_sync:
                pp_active = True
            else:
                pp_active = False
            
            # Debug logging (only at transitions)
            if not hasattr(self, '_last_pp_state'):
                self._last_pp_state = None
            if self._last_pp_state != pp_active:
                mode = "PP+CFG" if pp_active else "CFG-only"
                print(f"[Rank {rank}][Step {infer_step}] Mode: {mode} (tau1={tau1_sync}, tau2={tau2_sync})", flush=True)
                self._last_pp_state = pp_active
            # ==========================================
            
            # ===== Control PP plugins =====
            if not pp_active:
                # Disable PP: all blocks execute locally
                if not hasattr(self, '_pp_disabled') or not self._pp_disabled:
                    for mod in self.reformed_modules.values():
                        mod.plugin.pp_enabled = False
                    self._pp_disabled = True
            else:
                # Enable PP: blocks use cached results
                if not hasattr(self, '_pp_disabled') or self._pp_disabled:
                    for mod in self.reformed_modules.values():
                        mod.plugin.pp_enabled = True
                    self._pp_disabled = False
            # ==============================
            
            # ===== Execute transformer =====
            if self.stride == 1:
                # Synchronize cached activations BEFORE execution (for PP mode)
                if pp_active and (infer_step-1) % self.stride == 0:
                    # comm_index: [12] for model_n=2 (between rank 0 and rank 1)
                    module_list = list(self.reformed_modules.values())
                    for comm_idx in self.comm_index:
                        if comm_idx < len(module_list):
                            module_list[comm_idx].plugin.cache_sync(False)
                
                # Time shift (if enabled)
                if self.time_shift and infer_step >= self.warm_up:
                    kwargs["timestep"] = torch.cat([
                        self.pipeline.scheduler.timesteps[infer_step-1].unsqueeze(0), 
                        self.pipeline.scheduler.timesteps[infer_step-1].unsqueeze(0)
                    ])
                
                # Execute transformer (with PP if enabled)
                sample = transformer.old_forward(*args, **kwargs)[0]
                
                # Broadcast final result (only in PP mode)
                infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
                if pp_active and infer_step >= self.warm_up and (infer_step-1) % self.stride == 0:
                    dist.broadcast(sample, self.model_n-1)
                
                return sample,
                
            else:  # stride == 2
                # Synchronize cached activations (stride=2 pattern)
                if pp_active and (infer_step-1) % self.stride == 1:
                    module_list = list(self.reformed_modules.values())
                    for comm_idx in self.comm_index:
                        if comm_idx < len(module_list):
                            module_list[comm_idx].plugin.cache_sync(False)

                # Time shift logic
                shift = 1 if self.time_shift else 0

                if infer_step >= self.warm_up:
                    if dist.get_rank() < self.model_n and (infer_step-1) % self.stride == 0 and \
                       infer_step < len(self.pipeline.scheduler.timesteps) - 1:
                        kwargs["timestep"] = torch.cat([
                            self.pipeline.scheduler.timesteps[infer_step+1-shift].unsqueeze(0), 
                            self.pipeline.scheduler.timesteps[infer_step+1-shift].unsqueeze(0)
                        ])
                    else:
                        kwargs["timestep"] = torch.cat([
                            self.pipeline.scheduler.timesteps[infer_step-shift].unsqueeze(0), 
                            self.pipeline.scheduler.timesteps[infer_step-shift].unsqueeze(0)
                        ])
                
                # Execute transformer
                sample = transformer.old_forward(*args, **kwargs)[0]

                # Broadcast final result
                infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
                if pp_active and infer_step >= self.warm_up and (infer_step-1) % self.stride == 1:
                    dist.broadcast(sample, self.model_n)
    
                return sample,

        transformer.forward = transformer_forward


    def reform_pipeline(self):
        models = splite_model(self.pipeline, self.pipe_id, self.model_n)
        for model_i, sub_model in enumerate(models):
            for module_id, module in enumerate(sub_model):
                self.reform_module(module, module_id, model_i)
        self.reform_transformer()