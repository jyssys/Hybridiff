import torch.distributed as dist
import torch
from .tools import ResultPicker
from .pipe_config import splite_model

# from torchprofile import profile_macs
    
class ModulePlugin(object):
    def __init__(self, module, model_i, stride=1, run_mode=None) -> None:
        self.model_i, self.stride, self.run_mode = model_i, stride, run_mode
        self.module = module
        self.module.plugin = self

        self.init_state()
        self.inject_forward()
        self.rank = dist.get_rank()
        self.pipeline_disabled = False

    def init_state(self,warmup_n=1):
        self.warmup_n = warmup_n
        self.result_structure, self.cached_result = None, None
        self.infer_step = 0
        self.pipeline_disabled = False
        self.pp_enabled = False
    
    def cache_sync(self, async_flag):
        if self.cached_result is None:
            return
        
        if self.infer_step >= self.warmup_n:
            dist.broadcast(self.cached_result, self.model_i, async_op=async_flag)

    def inject_forward(self):
        assert not hasattr(self.module, 'old_forward'), "Module already has old_forward attribute."
        module = self.module
        module.old_forward = module.forward

        def new_forward(*args, **kwargs):
            # ==== cut-off ====
            # if self.infer_step > self.pp_time_step:
            #     self.infer_step += 1
            #     return module.old_forward(*args, **kwargs)
            
            if self.pipeline_disabled or not self.pp_enabled:
                self.infer_step += 1
                return module.old_forward(*args, **kwargs)
            # =====================
        
            # ==== sync tau1 & tau2 ====    
            if (self.infer_step > self.warmup_n and self.cached_result is None) or (self.infer_step > self.warmup_n and self.result_structure is None):
                result = module.old_forward(*args, **kwargs)
                self.cached_result, self.result_structure = ResultPicker.dump(result)
                self.infer_step += 1
                return result
            # =========================
        
            run_locally = (self.run_mode[0]==self.model_i) and ((self.infer_step-1)%self.stride==0)

            if self.infer_step<self.warmup_n or run_locally:
                result = module.old_forward(*args, **kwargs)
                if (self.infer_step+1==self.warmup_n) or (self.infer_step + 1 > self.warmup_n and self.run_mode[1]==0):
                    self.cached_result, self.result_structure = ResultPicker.dump(result)
            else:
                result = ResultPicker.load(self.cached_result, self.result_structure)
                
            self.infer_step += 1
            return result
        
        module.forward = new_forward
    
    def set_warmup(self, n: int):
        if n >= 0:
            self.warmup_n = n

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
        self._pp_disabled = False
        
        self.reformed_modules = {}
        self.reform_pipeline()
        

    def reset_state(self, warm_up=20):
        self.warm_up = warm_up
        self._pp_disabled = False
        if hasattr(self, '_last_pp_state'):
            del self._last_pp_state
        if hasattr(self, '_tau_buf'):
            del self._tau_buf
        for each in self.reformed_modules.values():
            each.plugin.init_state(warmup_n=warm_up)

    def reform_module(self, module, module_id, model_i):
        run_mode = (dist.get_rank(), 0) if dist.get_rank() < self.model_n else (self.model_n -1, 1)
        ModulePlugin(module, model_i, self.stride, run_mode)
        self.reformed_modules[(model_i, module_id)] = module
    
    def reform_unet(self):
        unet = self.pipeline.unet
        assert not hasattr(unet, 'old_forward'), "Unet already has old_forward attribute."
        unet.old_forward = unet.forward

        def unet_forward(*args, **kwargs):

            # return unet.old_forward(*args, **kwargs)
        
            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            
            # ===== sync tau1 & tau2 ========
            if not hasattr(self, "_tau_buf"):
                self._tau_buf = torch.full((2,), -1, dtype=torch.int32, device=args[0].device)

            if dist.get_rank() == 0:
                self._tau_buf.copy_(self.pipeline.tau_tensor.to(torch.int32))
                
            dist.broadcast(self._tau_buf, src = 0)
            tau1_sync, tau2_sync = map(int, self._tau_buf.tolist())
            print(f"tau1: {tau1_sync}, tau2: {tau2_sync}")
            
            if tau1_sync >= 0 and tau1_sync != self.warm_up:
                self.warm_up = tau1_sync
                for mod in self.reformed_modules.values():
                    mod.plugin.cached_result = None
                    mod.plugin.set_warmup(tau1_sync)
            
            if tau1_sync < 0 or infer_step < tau1_sync:
                pp_active = False
            elif tau2_sync < 0 or infer_step <= tau2_sync:
                pp_active = True
            else:
                pp_active = False
                                
            # ===============================
             
            # ==== Control PP plugins =====
            if not pp_active:
                # Disable PP
                if not hasattr(self, '_pp_disabled') or not self._pp_disabled:
                    for mod in self.reformed_modules.values():
                        mod.plugin.pp_enabled = False
                    self._pp_disabled = True
                
                # Permanently disable after tau2 (legacy logic)
                if tau2_sync > 0 and infer_step > tau2_sync:
                    for mod in self.reformed_modules.values():
                        mod.plugin.pipeline_disabled = True
                
                return unet.old_forward(*args, **kwargs)
            else:
                # Enable PP
                if not hasattr(self, '_pp_disabled') or self._pp_disabled:
                    for mod in self.reformed_modules.values():
                        mod.plugin.pp_enabled = True
                    self._pp_disabled = False
            # ==============================

            if self.stride==1:
                if (infer_step-1)%self.stride == 0:
                    for each in self.reformed_modules.values():
                        each.plugin.cache_sync(False)
                        
                if self.time_shift:
                    if infer_step>=self.warm_up:
                        args = list(args)
                        args[1] = self.pipeline.scheduler.timesteps[infer_step-1]
                
                sample = unet.old_forward(*args, **kwargs)[0]
                infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
                if infer_step>=self.warm_up and (infer_step-1)%self.stride == 0:      
                    dist.broadcast(sample, self.model_n-1)

                return sample,
            else:
                if (infer_step-1)%self.stride == 1:
                    for each in self.reformed_modules.values():
                        each.plugin.cache_sync(False)

                if self.time_shift:
                    shift = 1
                else:
                    shift = 0

                if infer_step>=self.warm_up:
                    if dist.get_rank() < self.model_n and (infer_step-1)%self.stride == 0 and infer_step< len(self.pipeline.scheduler.timesteps)-1:
                        args = list(args)
                        args[1] = self.pipeline.scheduler.timesteps[infer_step+1-shift]
                    else:
                        args = list(args)
                        args[1] = self.pipeline.scheduler.timesteps[infer_step-shift]
                sample = unet.old_forward(*args, **kwargs)[0]

                infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
                if infer_step>=self.warm_up and (infer_step-1)%self.stride == 1:
                    dist.broadcast(sample, self.model_n)
    
                return sample,

        unet.forward = unet_forward


    def reform_pipeline(self):
        models = splite_model(self.pipeline, self.pipe_id, self.model_n)
        for model_i, sub_model in enumerate(models):
            for module_id, module in enumerate(sub_model):
                self.reform_module(module, module_id, model_i)
        self.reform_unet()