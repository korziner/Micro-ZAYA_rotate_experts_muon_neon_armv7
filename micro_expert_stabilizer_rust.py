"""
MicroExpertStabilizer на базе Rust INT8 с корректным autograd.

Ключевой принцип:
- ВЕСЬ микроэксперт обёрнут в torch.autograd.Function
- Градиент течёт через вход и выход (для обучения проекций)
- INT8 веса фиксированы (стабилизаторы не обучаются)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import micro_moe_int8


class MicroExpertFunction(torch.autograd.Function):
    """Autograd-совместимый микроэксперт.
    
    Forward: input_proj → clamp → INT8 up → ReLU → INT8 down → output_proj
    Backward: градиенты только через проекции (не через INT8)
    """
    
    @staticmethod
    def forward(ctx, x, input_proj_weight, output_proj_weight, 
                up_int8, up_scale, down_int8, down_scale, scale):
        # Сохраняем для backward
        ctx.save_for_backward(x, input_proj_weight, output_proj_weight,
                             up_int8, up_scale, down_int8, down_scale, scale)
        
        original_dtype = x.dtype
        device = x.device
        B, T, D = x.shape
        micro_hidden = up_int8.shape[0]
        
        # === ПРОЕКЦИЯ ВХОДА (в типе входа) ===
        h = F.linear(x, input_proj_weight)  # [B, T, micro_hidden]
        h = torch.clamp(h, -64.0, 64.0)
        
        # === INT8 ОПЕРАЦИИ (через numpy) ===
        h_np = h.detach().cpu().to(torch.float32).numpy()
        h_flat = h_np.reshape(-1, micro_hidden)
        
        up_int8_np = up_int8.cpu().numpy()
        up_scale_np = up_scale.cpu().numpy().astype(np.float32)
        down_int8_np = down_int8.cpu().numpy()
        down_scale_np = down_scale.cpu().numpy().astype(np.float32)
        
        # Up projection
        try:
            h_up = micro_moe_int8.int8_linear_forward(h_flat, up_int8_np, up_scale_np)
        except Exception:
            w_up_f32 = up_int8_np.astype(np.float32) * up_scale_np[:, None]
            h_up = h_flat @ w_up_f32.T
        
        h_up = np.maximum(h_up, 0)  # ReLU
        h_up = np.clip(h_up, 0, 64)
        
        # Down projection
        try:
            h_down = micro_moe_int8.int8_linear_forward(h_up, down_int8_np, down_scale_np)
        except Exception:
            w_down_f32 = down_int8_np.astype(np.float32) * down_scale_np[:, None]
            h_down = h_up @ w_down_f32.T
        
        # === ПРОЕКЦИЯ ВЫХОДА ===
        h_down_t = torch.from_numpy(h_down).to(device, dtype=original_dtype)
        h_down_t = h_down_t.view(B, T, micro_hidden)
        
        out = F.linear(h_down_t, output_proj_weight)  # [B, T, D]
        out = out * scale
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        (x, input_proj_weight, output_proj_weight,
         up_int8, up_scale, down_int8, down_scale, scale) = ctx.saved_tensors
        
        B, T, D = x.shape
        micro_hidden = up_int8.shape[0]
        
        # === ГРАДИЕНТ ПО ВЫХОДУ ПРОЕКЦИИ ===
        # out = linear(h_down, output_proj_weight) * scale
        # grad_h_down = grad_output @ output_proj_weight.T / scale
        grad_output_scaled = grad_output / scale.clamp(min=1e-8)
        grad_h_down = F.linear(grad_output_scaled, output_proj_weight.t())  # [B, T, micro_hidden]
        
        # Градиент по output_proj_weight
        # grad_w_out = grad_output_scaled.T @ h_down
        # Нужен h_down — восстанавливаем из forward
        # Для экономии: используем приближение через x
        grad_h_down_flat = grad_h_down.reshape(-1, micro_hidden)
        
        # Пересчитываем h (input_proj → clamp → INT8 up → ReLU → INT8 down)
        with torch.no_grad():
            h = F.linear(x, input_proj_weight)
            h = torch.clamp(h, -64.0, 64.0)
            h_np = h.cpu().to(torch.float32).numpy().reshape(-1, micro_hidden)
            
            up_int8_np = up_int8.cpu().numpy()
            up_scale_np = up_scale.cpu().numpy().astype(np.float32)
            down_int8_np = down_int8.cpu().numpy()
            down_scale_np = down_scale.cpu().numpy().astype(np.float32)
            
            try:
                h_up = micro_moe_int8.int8_linear_forward(h_np, up_int8_np, up_scale_np)
            except Exception:
                w_up_f32 = up_int8_np.astype(np.float32) * up_scale_np[:, None]
                h_up = h_np @ w_up_f32.T
            h_up = np.maximum(h_up, 0)
            h_up = np.clip(h_up, 0, 64)
            
            try:
                h_down_np = micro_moe_int8.int8_linear_forward(h_up, down_int8_np, down_scale_np)
            except Exception:
                w_down_f32 = down_int8_np.astype(np.float32) * down_scale_np[:, None]
                h_down_np = h_up @ w_down_f32.T
        
        # grad_output_proj_weight = grad_output_scaled_flat.T @ h_down_flat
        grad_output_flat = grad_output_scaled.reshape(-1, D)
        h_down_flat = torch.from_numpy(h_down_np).to(grad_output.device, dtype=grad_output.dtype)
        grad_output_proj_weight = torch.matmul(grad_output_flat.t(), h_down_flat)
        
        # === ГРАДИЕНТ ПО ВХОДУ ПРОЕКЦИИ ===
        # Приближение: считаем, что INT8 слой — линейный
        # Это даёт корректный градиент для input_proj
        
        # Градиент по h_down (уже есть) → градиент по h_up (через down_int8)
        with torch.no_grad():
            w_down_f32 = down_int8_np.astype(np.float32) * down_scale_np[:, None]
            grad_h_up_np = grad_h_down_flat.cpu().numpy() @ w_down_f32  # [B*T, micro_hidden]
        
        # Через ReLU (маска)
        relu_mask = (h_up > 0).astype(np.float32)
        grad_h_up_np = grad_h_up_np * relu_mask
        
        # Через up_int8
        with torch.no_grad():
            w_up_f32 = up_int8_np.astype(np.float32) * up_scale_np[:, None]
            grad_h_np = grad_h_up_np @ w_up_f32  # [B*T, micro_hidden]
        
        grad_h = torch.from_numpy(grad_h_np).to(grad_output.device, dtype=grad_output.dtype)
        grad_h = grad_h.view(B, T, micro_hidden)
        
        # Градиент через clamp (маска)
        with torch.no_grad():
            h_pre_clamp = F.linear(x, input_proj_weight)
            clamp_mask = ((h_pre_clamp >= -64.0) & (h_pre_clamp <= 64.0)).to(grad_h.dtype)
        grad_h = grad_h * clamp_mask
        
        # Градиент по input_proj_weight
        # grad_w_in = grad_h_flat.T @ x_flat
        x_flat = x.reshape(-1, D)
        grad_h_flat = grad_h.reshape(-1, micro_hidden)
        grad_input_proj_weight = torch.matmul(grad_h_flat.t(), x_flat)
        
        # Градиент по входу x
        grad_x = F.linear(grad_h, input_proj_weight.t())
        
        # Градиент по scale (правильная формула)
        # out = linear(h_down, output_proj_weight) * scale
        # d(loss)/d(scale) = sum(grad_output * out_before_scale)
        out_before_scale = F.linear(h_down_flat.to(grad_output.device, dtype=grad_output.dtype), 
                                     output_proj_weight)
        out_before_scale = out_before_scale.view(grad_output.shape)
        grad_scale = (grad_output * out_before_scale).sum()
        
        return (grad_x, grad_input_proj_weight, grad_output_proj_weight,
                None, None, None, None, grad_scale.unsqueeze(0) if scale.dim() == 0 else grad_scale)


class INT8MicroExpertRust(nn.Module):
    """Один микроэксперт на базе Rust INT8 с корректным autograd."""
    
    def __init__(self, dim, micro_hidden=32):
        super().__init__()
        self.dim = dim
        self.micro_hidden = micro_hidden
        
        # Проекции входа/выхода (обучаются)
        self.input_proj = nn.Linear(dim, micro_hidden, bias=False)
        self.output_proj = nn.Linear(micro_hidden, dim, bias=False)
        
        # Малая инициализация
        nn.init.normal_(self.input_proj.weight, std=0.01)
        nn.init.normal_(self.output_proj.weight, std=0.01)
        
        # Веса эксперта в float32 (для квантизации)
        self.register_buffer(
            'up_weight_f32',
            torch.randn(micro_hidden, micro_hidden) * 0.1
        )
        self.register_buffer(
            'down_weight_f32',
            torch.randn(micro_hidden, micro_hidden) * 0.1
        )
        
        # Масштаб (обучается)
        self.scale = nn.Parameter(torch.tensor(0.05))
        
        # Кэш квантизованных весов
        self._up_int8 = None
        self._up_scale = None
        self._down_int8 = None
        self._down_scale = None
        self._cache_valid = False
    
    def _quantize_weights(self):
        """Квантизует веса через Rust quantize_per_row."""
        up_f32 = self.up_weight_f32.cpu().numpy().astype(np.float32)
        down_f32 = self.down_weight_f32.cpu().numpy().astype(np.float32)
        
        try:
            result = micro_moe_int8.quantize_per_row(up_f32)
            self._up_int8, self._up_scale = result
        except Exception:
            max_abs = np.abs(up_f32).max(axis=1, keepdims=True) + 1e-8
            self._up_scale = (max_abs / 127.0).astype(np.float32).flatten()
            self._up_int8 = np.clip(up_f32 / max_abs * 127, -127, 127).astype(np.int8)
        
        try:
            result = micro_moe_int8.quantize_per_row(down_f32)
            self._down_int8, self._down_scale = result
        except Exception:
            max_abs = np.abs(down_f32).max(axis=1, keepdims=True) + 1e-8
            self._down_scale = (max_abs / 127.0).astype(np.float32).flatten()
            self._down_int8 = np.clip(down_f32 / max_abs * 127, -127, 127).astype(np.int8)
        
        self._cache_valid = True
    
    def forward(self, x):
        """Forward через autograd-совместимую функцию."""
        if not self._cache_valid:
            self._quantize_weights()
        
        # Конвертируем квантизованные веса в тензоры
        device = x.device
        up_int8_t = torch.from_numpy(self._up_int8).to(device)
        up_scale_t = torch.from_numpy(self._up_scale).to(device)
        down_int8_t = torch.from_numpy(self._down_int8).to(device)
        down_scale_t = torch.from_numpy(self._down_scale).to(device)
        
        # Вызов через autograd Function
        return MicroExpertFunction.apply(
            x, self.input_proj.weight, self.output_proj.weight,
            up_int8_t, up_scale_t, down_int8_t, down_scale_t, self.scale
        )
    
    def reinit_weights(self):
        """Переинициализация весов."""
        with torch.no_grad():
            device = self.up_weight_f32.device
            self.up_weight_f32.copy_(torch.randn_like(self.up_weight_f32) * 0.1)
            self.down_weight_f32.copy_(torch.randn_like(self.down_weight_f32) * 0.1)
        self._cache_valid = False


class MicroExpertPoolRust(nn.Module):
    """Пул микроэкспертов на базе Rust INT8."""
    
    def __init__(self, dim, num_micro_experts=4, micro_hidden=32):
        super().__init__()
        self.num_micro_experts = num_micro_experts
        
        self.experts = nn.ModuleList([
            INT8MicroExpertRust(dim, micro_hidden)
            for _ in range(num_micro_experts)
        ])
        
        self.router = nn.Linear(dim, num_micro_experts, bias=True)
        # Случайная инициализация для исследования всех экспертов
        nn.init.normal_(self.router.weight, std=0.05)
        nn.init.zeros_(self.router.bias)
        
        # Шум для исследования (добавляется при обучении)
        self.register_buffer('router_noise_std', torch.tensor(0.1))
        
        self.register_buffer('total_activations', torch.tensor(0))
    
    def forward(self, x, top_k=2):
        """Роутинг по микроэкспертам с шумом для исследования."""
        router_logits = self.router(x)
        
        # Добавляем шум при обучении (для исследования всех экспертов)
        if self.training and hasattr(self, 'router_noise_std'):
            noise = torch.randn_like(router_logits) * self.router_noise_std
            router_logits = router_logits + noise
        
        router_weights = F.softmax(router_logits, dim=-1)
        
        outputs = [expert(x) for expert in self.experts]
        outputs = torch.stack(outputs, dim=0)
        
        top_weights, top_indices = torch.topk(router_weights, k=top_k, dim=-1)
        
        expert_mask = torch.zeros_like(router_weights)
        for k in range(top_k):
            expert_mask.scatter_(-1, top_indices[..., k:k+1], 1.0)
        
        masked_weights = router_weights * expert_mask
        masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        mixed = torch.einsum('ebtd,bte->btd', outputs, masked_weights)
        
        self.total_activations += x.shape[0] * x.shape[1]
        
        return mixed
    
    def get_memory_usage(self):
        """Оценка памяти в МБ."""
        total_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        total_bytes += sum(b.numel() * b.element_size() for b in self.buffers())
        return total_bytes / 1024 / 1024


class AdaptiveStabilizationManager:
    """Менеджер адаптивной стабилизации.
    
    Управляет переключением между режимами:
    - 'micro_only': микроэксперты как стабилизаторы (вес 0.3) + большие (0.7)
    - 'micro_plus_big': переходный режим (0.2 + 0.8)
    - 'big_only': только большие эксперты (0.0 + 1.0)
    """
    
    def __init__(self, warmup_steps=200):
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.mode = 'micro_only'
        self.stability_score = 0.0
        self.nan_count = 0
        self.consecutive_good_steps = 0
        self.loss_history = []
    
    def record_step(self, loss_value, grad_norm):
        """Записывает результат шага."""
        self.current_step += 1
        
        try:
            is_valid = (loss_value == loss_value and 
                       abs(loss_value) < 1e10 and
                       grad_norm < 100.0)
        except (TypeError, ValueError):
            is_valid = False
        
        if is_valid:
            self.consecutive_good_steps += 1
            self.nan_count = 0
            self.loss_history.append(loss_value)
            if grad_norm < 10.0:
                self.stability_score = min(1.0, self.stability_score + 0.02)
            if len(self.loss_history) > 100:
                self.loss_history.pop(0)
        else:
            self.nan_count += 1
            self.consecutive_good_steps = 0
            self.stability_score = max(0.0, self.stability_score - 0.3)
        
        self._update_mode()
    
    def _update_mode(self):
        """Обновляет режим работы."""
        old_mode = self.mode
        
        if self.current_step < self.warmup_steps // 2:
            self.mode = 'micro_only'
        elif self.current_step < self.warmup_steps:
            self.mode = 'micro_plus_big' if self.stability_score > 0.3 else 'micro_only'
        else:
            if self.stability_score > 0.8 and self.consecutive_good_steps > 50:
                self.mode = 'big_only'
            elif self.stability_score > 0.5:
                self.mode = 'micro_plus_big'
            else:
                self.mode = 'micro_only'
        
        if self.nan_count > 0:
            self.mode = 'micro_only'
        
        if old_mode != self.mode:
            mode_names = {
                'micro_only': '🟡 микро + большие (стабилизация)',
                'micro_plus_big': '🟠 переходный режим',
                'big_only': '🟢 только большие'
            }
            print(f"\n🔄 Смена режима: {mode_names.get(old_mode, old_mode)} → "
                  f"{mode_names.get(self.mode, self.mode)}")
    
    def should_use_micro_experts(self):
        """Использовать ли микроэксперты."""
        return self.mode in ('micro_only', 'micro_plus_big')
    
    def should_use_big_experts(self):
        """Использовать ли большие эксперты (всегда активны)."""
        return True
    
    def get_micro_weight(self):
        """Вес микроэкспертов (стабилизаторы, не замена)."""
        if self.mode == 'micro_only':
            return 0.3
        elif self.mode == 'micro_plus_big':
            return 0.2
        else:
            return 0.0
    
    def get_big_weight(self):
        """Вес больших экспертов (всегда активны)."""
        if self.mode == 'micro_only':
            return 0.7
        elif self.mode == 'micro_plus_big':
            return 0.8
        else:
            return 1.0
    
    def get_status(self):
        """Текущий статус."""
        return {
            'step': self.current_step,
            'mode': self.mode,
            'stability': self.stability_score,
            'nan_count': self.nan_count,
            'good_steps': self.consecutive_good_steps,
        }
