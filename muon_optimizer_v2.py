"""
Оптимизатор моментум-свободного Муона с прямой работой через ctypes data_ptr.
Устраняет конвертации через numpy.
"""
import torch
import numpy as np
import ctypes
import os

# ============================================================================
# ИНИЦИАЛИЗАЦИЯ МУОН-МОДУЛЯ
# ============================================================================
_lib = None
MUON_RUST_AVAILABLE = False
HAS_PTR_SUPPORT = False

try:
    import muon_neon
    MUON_RUST_AVAILABLE = True
except ImportError:
    muon_neon = None
    print("[Muon] ⚠️ Rust-модуль недоступен, будет использован чистый PyTorch")

if MUON_RUST_AVAILABLE:
    module_dir = os.path.dirname(muon_neon.__file__)
    so_path = None
    for fname in os.listdir(module_dir):
        if fname.endswith('.so'):
            so_path = os.path.join(module_dir, fname)
            break
    
    if so_path is not None:
        try:
            _lib = ctypes.CDLL(so_path)
            
            if hasattr(_lib, 'newton_schulz_5steps_inplace'):
                _lib.newton_schulz_5steps_inplace.argtypes = [
                    ctypes.c_uint64, ctypes.c_size_t, ctypes.c_size_t
                ]
                _lib.newton_schulz_5steps_inplace.restype = ctypes.c_int
            
            if hasattr(_lib, 'newton_schulz_5steps_out'):
                _lib.newton_schulz_5steps_out.argtypes = [
                    ctypes.c_uint64, ctypes.c_uint64, 
                    ctypes.c_size_t, ctypes.c_size_t
                ]
                _lib.newton_schulz_5steps_out.restype = ctypes.c_int
            
            if (hasattr(_lib, 'newton_schulz_5steps_inplace') or 
                hasattr(_lib, 'newton_schulz_5steps_out')):
                HAS_PTR_SUPPORT = True
                print("[Muon] ✅ Режим data_ptr доступен (быстрый путь через ctypes)")
            else:
                print("[Muon] ℹ️  C-функции не найдены, будет использован Python API")
        except Exception as e:
            print(f"[Muon] ℹ️  Не удалось загрузить C-функции ({e}), будет использован Python API")


class MomentumFreeMuon(torch.optim.Optimizer):
    """Моментум-свободный Муон."""
    
    def __init__(self, params, lr=0.02, adamw_params=None, adamw_lr=3e-4):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)
        
        self.adamw_params = adamw_params or []
        self.adamw_optimizer = None
        if self.adamw_params:
            self.adamw_optimizer = torch.optim.AdamW(self.adamw_params, lr=adamw_lr)
        
        self._buffer_cache = {}
    
    def _get_buffer(self, shape, dtype, device):
        key = (shape, dtype, device)
        if key not in self._buffer_cache:
            self._buffer_cache[key] = torch.empty(shape, dtype=dtype, device=device)
        return self._buffer_cache[key]
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                if MUON_RUST_AVAILABLE and grad.dim() >= 2:
                    # Используем быстрый путь через data_ptr
                    if HAS_PTR_SUPPORT and grad.is_contiguous() and grad.dtype == torch.float32:
                        output_buffer = self._get_buffer(grad.shape, grad.dtype, grad.device)
                        input_ptr = ctypes.c_uint64(grad.data_ptr())
                        output_ptr = ctypes.c_uint64(output_buffer.data_ptr())
                        
                        result = _lib.newton_schulz_5steps_out(
                            input_ptr, output_ptr,
                            ctypes.c_size_t(grad.shape[0]), 
                            ctypes.c_size_t(grad.shape[1])
                        )
                        
                        if result == 0:
                            p.add_(output_buffer, alpha=-lr)
                        else:
                            self._fallback_update(p, grad, lr)
                    else:
                        self._fallback_update(p, grad, lr)
                else:
                    p.add_(grad, alpha=-lr)
        
        if self.adamw_optimizer is not None:
            # ВАЖНО: перед AdamW приводим градиенты к типу параметров
            # Это критично для FP16, где AdamW требует совпадения типов
            for group in self.adamw_optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        target_dtype = p.dtype
                        if p.grad.dtype != target_dtype:
                            try:
                                # Разрешаем градиентам быть любого типа
                                if hasattr(p, 'grad_dtype'):
                                    p.grad_dtype = None
                                # Приводим градиент к типу параметра
                                p.grad = p.grad.to(target_dtype)
                            except Exception as e:
                                # Если не удалось привести, пропускаем параметр
                                print(f"⚠️ Не удалось привести градиент для параметра: {e}")
                                p.grad = None
            
            # Вызываем step только если есть хотя бы один градиент
            has_grads = any(p.grad is not None for group in self.adamw_optimizer.param_groups 
                           for p in group['params'])
            if has_grads:
                self.adamw_optimizer.step()
        
        return loss
    
    def _fallback_update(self, p, grad, lr):
        """Fallback через Python API."""
        grad_detached = grad.detach()
        
        if grad_detached.dtype == torch.float32:
            grad_np = grad_detached.numpy()
        else:
            grad_np = grad_detached.float().numpy()
        
        updated_np = muon_neon.newton_schulz_5steps(grad_np)
        
        if p.dtype == torch.float32:
            p.add_(torch.from_numpy(updated_np), alpha=-lr)
        else:
            updated = torch.from_numpy(updated_np).to(p.dtype)
            p.add_(updated, alpha=-lr)


def create_muon_optimizer(model, muon_lr=0.02, adamw_lr=3e-4):
    """Создаёт оптимизатор с разделением параметров."""
    muon_params = []
    adamw_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'embedding' in name or 'lm_head' in name:
            adamw_params.append(param)
        else:
            muon_params.append(param)
    
    optimizer = MomentumFreeMuon(
        muon_params, lr=muon_lr, 
        adamw_params=adamw_params, adamw_lr=adamw_lr
    )
    
    return optimizer, len(muon_params), len(adamw_params)
