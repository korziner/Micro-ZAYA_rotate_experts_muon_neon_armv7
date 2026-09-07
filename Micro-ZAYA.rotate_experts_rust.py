#!/usr/bin/env python3
"""
Micro-ZAYA: обучение MoE-модели с ротацией экспертов и поиском через Optuna.

Поддерживает:
- Гибкую параметризацию через аргументы командной строки
- Учёт предыдущих попыток поиска (базы SQLite)
- Seekable zstd корпус (сжатие ~3x)
- Safetensors чекпоинты
- Периодический инференс с русскими затравками
- Ротацию экспертов с заморозкой
"""

import argparse
import sys
import os
import time
import gc
import json
import warnings
import re
import glob
import shutil

# === МИКРОЭКСПЕРТЫ ДЛЯ СТАБИЛИЗАЦИИ ===
HAS_MICRO_EXPERTS = False
try:
    from micro_expert_stabilizer_rust import MicroExpertPoolRust as MicroExpertPool, AdaptiveStabilizationManager
    HAS_MICRO_EXPERTS = True
    print("✅ Микроэксперты доступны")
except Exception as e:
    HAS_MICRO_EXPERTS = False


import psutil


# === ПЛАНИРОВЩИК ТОЧНОСТИ: старт в BF16 → обучение в FP16 ===
class PrecisionScheduler:
    """Управляет переходом между фазами точности.
    
    Фаза 1 (stabilizing): хранение в BF16 для защиты от переполнения
    Фаза 2 (training): конвертация в FP16 для SIMD-оптимизации
    
    Критерии перехода:
    - Минимум stabilization_steps шагов
    - Отсутствие NaN в последних good_steps_threshold шагах
    - Лосс ниже порога
    """
    
    def __init__(self, 
                 stabilization_steps=200,
                 good_steps_threshold=100,
                 loss_threshold=9.0):
        self.stabilization_steps = stabilization_steps
        self.good_steps_threshold = good_steps_threshold
        self.loss_threshold = loss_threshold
        
        self.current_phase = 'stabilizing'  # 'stabilizing' | 'training'
        self.current_step = 0
        self.good_steps = 0
        self.last_loss = None
        self.converted = False
        
        print(f"   🎯 Планировщик точности:")
        print(f"      Фаза стабилизации: {stabilization_steps} шагов (BF16)")
        print(f"      Фаза обучения: после конвертации в FP16")
        print(f"      Порог лосса: {loss_threshold}")
    
    def record_step(self, loss_value):
        """Записывает результат шага."""
        self.current_step += 1
        self.last_loss = loss_value
        
        # Проверяем валидность
        try:
            is_valid = (loss_value == loss_value and 
                       abs(loss_value) < 1e10 and
                       loss_value < 100.0)
        except (TypeError, ValueError):
            is_valid = False
        
        if is_valid:
            self.good_steps += 1
        else:
            self.good_steps = 0  # Сброс счётчика при сбое
    
    def should_convert(self):
        """Проверяет, нужно ли конвертировать в FP16."""
        if self.converted:
            return False
        
        if self.current_step < self.stabilization_steps:
            return False
        
        if self.good_steps < self.good_steps_threshold:
            return False
        
        if self.last_loss is not None and self.last_loss > self.loss_threshold:
            return False
        
        return True
    
    def convert_to_fp16(self, model):
        """Конвертирует модель в FP16 для SIMD-оптимизации."""
        print(f"\n🔄 КОНВЕРТАЦИЯ: хранение в FP16 для SIMD-оптимизации")
        print(f"   Шаг: {self.current_step}")
        print(f"   Хороших шагов подряд: {self.good_steps}")
        print(f"   Последний лосс: {self.last_loss:.4f}")
        
        # Конвертируем модель
        model = model.to(torch.float16)
        
        self.converted = True
        self.current_phase = 'training'
        
        print(f"   ✅ Конвертация завершена")
        print(f"   🟢 Фаза: обучение (хранение в FP16)")
        
        return model
    
    def get_status(self):
        """Текущий статус."""
        return {
            'phase': self.current_phase,
            'step': self.current_step,
            'good_steps': self.good_steps,
            'last_loss': self.last_loss,
            'converted': self.converted,
        }


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Markovian RSA для соответствия техрепорту ZAYA1-8B
try:
    from markovian_rsa import MarkovianRSA, SimpleMarkovianRSA
    HAS_MARKOVIAN_RSA = True
except ImportError:
    HAS_MARKOVIAN_RSA = False
import optuna
from optuna.trial import TrialState



# Опциональные зависимости
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from safetensors.torch import save_file as st_save, load_file as st_load
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from muon_optimizer_v2 import create_muon_optimizer, MUON_RUST_AVAILABLE
except ImportError:
    try:
        from muon_optimizer import create_muon_optimizer, MUON_RUST_AVAILABLE
    except ImportError:
        MUON_RUST_AVAILABLE = False
        create_muon_optimizer = None

try:
    import seekable_corpus as sc
    SEEKABLE_AVAILABLE = True
except ImportError:
    SEEKABLE_AVAILABLE = False

try:
    from rotation_tracker import RotationTracker
    HAS_ROTATION_TRACKER = True
except ImportError:
    HAS_ROTATION_TRACKER = False

try:
    from inference_sampler import InferenceSampler
    HAS_INFERENCE_SAMPLER = True
except ImportError:
    HAS_INFERENCE_SAMPLER = False


# ============================================================================
# АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Парсер аргументов с подробной справкой."""
    
    examples = """
ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:

  1. Поиск конфигурации ~77M параметров:
     python Micro-ZAYA.rotate_experts_rust.py --target-params 77

  2. Поиск для ~50M с фиксированным числом экспертов:
     python Micro-ZAYA.rotate_experts_rust.py --target-params 50 --num-experts 24

  3. Продолжить предыдущий поиск:
     python Micro-ZAYA.rotate_experts_rust.py --target-params 77 --resume

  4. Начать поиск заново:
     python Micro-ZAYA.rotate_experts_rust.py --target-params 77 --fresh-start

  5. Пропустить поиск и обучать заданную конфигурацию:
     python Micro-ZAYA.rotate_experts_rust.py --skip-search \\
         --dim 256 --num-layers 4 --num-experts 32 --expert-hidden 1024 --num-heads 2

  6. Больше триалов:
     python Micro-ZAYA.rotate_experts_rust.py --target-params 77 --n-trials 200

  7. Сохранить лучшую конфигурацию:
     python Micro-ZAYA.rotate_experts_rust.py --target-params 77 --save-config best.json

ПРИМЕЧАНИЯ:
  * Число экспертов НЕ обязано быть степенью двойки
  * База Optuna хранится отдельно для каждого target: micro_zaya_<target>M.db
  * Optuna балансирует между историей и разведкой
"""
    
    parser = argparse.ArgumentParser(
        prog="Micro-ZAYA.rotate_experts_rust.py",
        description="Обучение MoE-модели с ротацией и поиском через Optuna.",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Целевая архитектура
    ga = parser.add_argument_group("Целевая архитектура")
    ga.add_argument("--target-params", type=float, default=77.0, metavar="M",
                   help="Целевой размер в M параметров (по умолчанию: 77.0)")
    ga.add_argument("--params-tolerance", type=float, default=25.0, metavar="%",
                   help="Допуск от целевого размера в %% (по умолчанию: 25.0)")
    ga.add_argument("--memory-limit", type=float, default=2500.0, metavar="MB",
                   help="Лимит памяти (по умолчанию: 2500)")
    ga.add_argument("--precision", type=str, default="fp32",
                   choices=["fp16", "fp32", "bf16"], metavar="PREC",
                   help="Точность (по умолчанию: fp32)")
    
    # Число экспертов
    ge = parser.add_argument_group("Число экспертов")
    ge.add_argument("--num-experts", type=int, default=None, metavar="N",
                   help="Желательное число экспертов (±8 для поиска)")
    ge.add_argument("--experts-min", type=int, default=4, metavar="N")
    ge.add_argument("--experts-max", type=int, default=64, metavar="N")
    
    # Диапазоны поиска
    gs = parser.add_argument_group("Диапазоны поиска")
    gs.add_argument("--dim-options", type=int, nargs="+",
                   default=[128, 192, 256, 320], metavar="D")
    gs.add_argument("--layers-min", type=int, default=4, metavar="N")
    gs.add_argument("--layers-max", type=int, default=12, metavar="N")
    gs.add_argument("--heads-options", type=int, nargs="+",
                   default=[2, 4, 8], metavar="H")
    gs.add_argument("--expert-hidden-options", type=int, nargs="+",
                   default=[256, 512, 1024], metavar="H")
    
    # Optuna
    go = parser.add_argument_group("Поиск Optuna")
    go.add_argument("--n-trials", type=int, default=100, metavar="N")
    go.add_argument("--timeout", type=int, default=3600, metavar="SEC")
    go.add_argument("--resume", action="store_true",
                   help="Продолжить предыдущий поиск")
    go.add_argument("--fresh-start", action="store_true",
                   help="Начать заново, удалив базу")
    go.add_argument("--exploration-rate", type=float, default=0.25, metavar="R",
                   help="Вес разведки (по умолчанию: 0.25)")
    go.add_argument("--startup-trials", type=int, default=10, metavar="N")
    go.add_argument("--skip-search", action="store_true",
                   help="Пропустить поиск")
    go.add_argument("--save-config", type=str, default=None, metavar="PATH")
    
    # Фиксированная конфигурация
    gf = parser.add_argument_group("Фиксированная конфигурация (--skip-search)")
    gf.add_argument("--dim", type=int, default=256, metavar="D")
    gf.add_argument("--num-layers", type=int, default=4, metavar="N")
    gf.add_argument("--num-heads", type=int, default=2, metavar="H")
    gf.add_argument("--expert-hidden", type=int, default=1024, metavar="H")
    
    # Обучение
    gt = parser.add_argument_group("Обучение")
    gt.add_argument("--batch-size", type=int, default=4, metavar="N")
    gt.add_argument("--seq-length", type=int, default=256, metavar="N")
    gt.add_argument("--max-steps", type=int, default=1000, metavar="N")
    gt.add_argument("--muon-lr", type=float, default=0.02, metavar="LR")
    gt.add_argument("--adamw-lr", type=float, default=3e-4, metavar="LR")
    gt.add_argument("--warmup-steps", type=int, default=100, metavar="N")
    gt.add_argument("--checkpoint-format", type=str, default="safetensors",
                   choices=["torch", "safetensors"])
    gt.add_argument("--save-every", type=int, default=500, metavar="N")
    gt.add_argument("--max-checkpoints", type=int, default=3, metavar="N",
                   help="Максимум хранимых чекпоинтов (по умолчанию: 3)")
    gt.add_argument("--initial-active-percent", type=int, default=33, metavar="P",
                   help="Начальный процент активных экспертов (по умолчанию: 33)")
    gt.add_argument("--min-active-percent", type=int, default=15, metavar="P",
                   help="Минимальный процент активных экспертов (по умолчанию: 15)")
    gt.add_argument("--memory-threshold-percent", type=int, default=85, metavar="P",
                   help="Порог использования памяти для адаптации (%%) (по умолчанию: 85)")
    gt.add_argument("--checkpoint-dir", type=str, default="checkpoints", metavar="DIR")
    gt.add_argument("--resume-from", type=str, default=None, metavar="PATH")
    gt.add_argument("--auto-resume", action="store_true")
    
    # Ротация
    gr = parser.add_argument_group("Ротация экспертов")
    gr.add_argument("--rotation-interval", type=int, default=100, metavar="N")
    gr.add_argument("--freeze-after-step", type=int, default=500, metavar="N")
    gr.add_argument("--num-active", type=int, default=1, metavar="N")
    
    # Данные
    gd = parser.add_argument_group("Данные")
    gd.add_argument("--corpus", type=str, default="corpus.tok16", metavar="PATH")
    gd.add_argument("--vocab-size", type=int, default=16384, metavar="N")
    gd.add_argument("--vocab-file", type=str, default="vocab16k.txt", metavar="PATH")
    
    # Инференс
    gi = parser.add_argument_group("Периодический инференс")
    gi.add_argument("--inference-interval", type=int, default=50, metavar="N")
    gi.add_argument("--inference-seed", type=str, default="Привет,", metavar="TEXT")
    gi.add_argument("--inference-tokens", type=int, default=32, metavar="N")
    gi.add_argument("--full-inference", action="store_true")
    
    return parser


# ============================================================================
# КОНФИГУРАЦИИ
# ============================================================================

class ModelConfig:
    """Конфигурация модели."""
    
    def __init__(self, vocab_size=16384, dim=256, num_layers=8, num_heads=4,
                 kv_heads=2, num_experts=16, expert_hidden=512,
                 router_dim=128, num_active=1, max_seq_len=1024):
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.num_experts = num_experts
        self.expert_hidden = expert_hidden
        self.router_dim = router_dim
        self.num_active = num_active
        self.max_seq_len = max_seq_len
    
    def estimate_params(self) -> int:
        embed = self.vocab_size * self.dim
        
        kv_dim = (self.dim // (self.num_heads // self.kv_heads)
                  if self.num_heads > self.kv_heads
                  else self.dim // self.num_heads)
        attn = (self.dim * self.dim + self.dim * kv_dim +
                self.dim * kv_dim + self.dim * self.dim)
        
        experts = self.num_experts * self.dim * self.expert_hidden * 2
        
        router = (self.dim * self.router_dim +
                  self.router_dim * (self.router_dim * 4) +
                  (self.router_dim * 4) * (self.router_dim * 4) +
                  (self.router_dim * 4) * self.num_experts)
        
        residual = self.dim * 4
        
        per_layer = attn + experts + router + residual
        return embed + self.num_layers * per_layer


class TrainingConfig:
    """Конфигурация обучения."""
    
    def __init__(self, batch_size=4, seq_length=256, muon_lr=0.02, adamw_lr=3e-4,
                 warmup_steps=100, max_steps=1000, rotation_interval=100,
                 freeze_after_step=500, target_params_m=77.0,
                 params_tolerance=25.0, memory_limit_mb=2500.0,
                 num_active=1, precision="fp32"):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.muon_lr = muon_lr
        self.adamw_lr = adamw_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.rotation_interval = rotation_interval
        self.freeze_after_step = freeze_after_step
        self.target_params_m = target_params_m
        self.params_tolerance = params_tolerance
        self.memory_limit_mb = memory_limit_mb
        self.num_active = num_active
        self.precision = precision


def estimate_memory(config, training_config, precision='fp32'):
    """Оценка памяти."""
    bytes_per_param = 2 if precision in ('fp16', 'bf16') else 4
    
    total_params = config.estimate_params()
    embed_params = config.vocab_size * config.dim
    
    weights_mb = total_params * bytes_per_param / 1024 / 1024
    gradients_mb = total_params * 4 / 1024 / 1024
    adamw_mb = embed_params * 8 / 1024 / 1024
    activations_mb = (training_config.batch_size * training_config.seq_length *
                      config.dim * config.num_layers * bytes_per_param / 1024 / 1024)
    
    total_mb = weights_mb + gradients_mb + adamw_mb + activations_mb
    
    return {
        'total_params': total_params,
        'total_params_m': total_params / 1e6,
        'weights_mb': weights_mb,
        'gradients_mb': gradients_mb,
        'adamw_mb': adamw_mb,
        'activations_mb': activations_mb,
        'total_mb': total_mb,
        'precision': precision.upper(),
        'bytes_per_param': bytes_per_param,
    }


# ============================================================================
# МОДУЛИ МОДЕЛИ
# ============================================================================

class ZAYA1Router(nn.Module):
    """MLP-роутер с EDA."""
    
    def __init__(self, dim, router_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.w_down = nn.Linear(dim, router_dim, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
        hidden = router_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(router_dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, num_experts, bias=False),
        )
        self.norm = nn.RMSNorm(router_dim)
        self.routing_biases = nn.Parameter(torch.zeros(num_experts))
        # Малая инициализация MLP роутера для стабильности FP16
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.005)  # Было 0.02, стало в 4 раза меньше
        
        # Краткосрочная статистика для PID (сбрасывается каждые 10 шагов)
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_tokens', torch.tensor(0))
        
        # Долгосрочная статистика для умной ротации (сбрасывается при ротации)
        self.register_buffer('long_term_counts', torch.zeros(num_experts))
        self.register_buffer('long_term_tokens', torch.tensor(0))
    

        # Долгосрочная статистика для умной ротации (НЕ сбрасывается PID)
        if not hasattr(self, 'long_term_counts'):
            self.register_buffer('long_term_counts', torch.zeros(num_experts))
        if not hasattr(self, 'long_term_tokens'):
            self.register_buffer('long_term_tokens', torch.tensor(0))
    def forward(self, x, prev_state=None):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        r = self.w_down(x_flat)
        
        if prev_state is not None and prev_state.shape == r.shape:
            r = r + self.gamma * prev_state
        
        r = self.norm(r)
        scores = self.mlp(r)
        
        # ВАЖНО: softmax в FP32 для предотвращения переполнения в FP16
        # exp(12) > 65504 → переполнение в FP16
        scores_fp32 = scores.float()
        # Clamping для дополнительной защиты
        scores_fp32 = torch.clamp(scores_fp32, min=-10.0, max=10.0)
        scores_softmax = F.softmax(scores_fp32, dim=-1)
        # Возвращаем в исходный тип, но значения уже нормализованы [0, 1]
        scores = scores_softmax.to(scores.dtype)
        
        biased = scores + self.routing_biases
        
        top_scores, top_idx = torch.topk(biased, k=1, dim=-1)
        top_weights = F.softmax(top_scores, dim=-1)
        
        if self.training:
            flat = top_idx.reshape(-1)
            unique, counts = torch.unique(flat, return_counts=True)
            for idx, cnt in zip(unique, counts):
                self.expert_counts[idx] += cnt.item()
                self.long_term_counts[idx] += cnt.item()
            self.total_tokens += B * T
            self.long_term_tokens += B * T
        
        return top_idx.view(B, T, -1), top_weights.view(B, T, -1), r
    
    def pid_update(self, lr=0.01):
        total = int(self.total_tokens.item()) if self.total_tokens.numel() > 0 else 0
        if total == 0:
            return
        p = self.expert_counts / self.total_tokens
        grad = p - 1.0 / self.num_experts
        with torch.no_grad():
            self.routing_biases -= lr * grad
        self.expert_counts.zero_()
        self.total_tokens.zero_()

    def get_usage_stats(self):
        """Долгосрочная статистика использования (для умной ротации)."""
        if not hasattr(self, 'long_term_counts') or self.long_term_counts is None:
            # Буфер не существует — создаём
            self.long_term_counts = torch.zeros(self.num_experts, device=self.w_down.weight.device)
        if not hasattr(self, 'long_term_tokens') or self.long_term_tokens is None:
            self.long_term_tokens = torch.tensor(0, device=self.w_down.weight.device)
        
        # Если это не тензор (например int после загрузки старого чекпоинта),
        # возвращаем равномерное распределение
        import torch
        if not isinstance(self.long_term_counts, torch.Tensor):
            return {i: 1.0 / self.num_experts for i in range(self.num_experts)}
        if not isinstance(self.long_term_tokens, torch.Tensor):
            return {i: 1.0 / self.num_experts for i in range(self.num_experts)}
        
        total = int(self.long_term_tokens.item())
        if total == 0:
            return {i: 0.0 for i in range(self.num_experts)}
        
        usage = self.long_term_counts.float() / float(total)
        return {i: float(usage[i].item()) for i in range(self.num_experts)}

    def reset_usage_stats(self):
        """Сброс долгосрочной статистики (вызывается при ротации)."""
        import torch
        device = self.w_down.weight.device
        
        # Сбрасываем long_term (долгосрочные)
        if hasattr(self, 'long_term_counts') and isinstance(self.long_term_counts, torch.Tensor):
            self.long_term_counts.zero_()
        else:
            self.long_term_counts = torch.zeros(self.num_experts, device=device)
        
        if hasattr(self, 'long_term_tokens') and isinstance(self.long_term_tokens, torch.Tensor):
            self.long_term_tokens.zero_()
        else:
            self.long_term_tokens = torch.tensor(0, device=device)

class ExpertFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)
    
    def forward(self, x):
        return self.down(F.gelu(self.up(x)))


class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.frozen_expert_indices = set()
        
        self.router = ZAYA1Router(config.dim, config.router_dim, config.num_experts)
        self.experts = nn.ModuleList([
            ExpertFFN(config.dim, config.expert_hidden)
            for _ in range(config.num_experts)
        ])
    
    def freeze_experts(self, indices):
        for idx in indices:
            if idx < len(self.experts):
                for p in self.experts[idx].parameters():
                    p.requires_grad = False
                self.frozen_expert_indices.add(idx)
    
    def unfreeze_experts(self, indices):
        for idx in indices:
            if idx < len(self.experts):
                for p in self.experts[idx].parameters():
                    p.requires_grad = True
                self.frozen_expert_indices.discard(idx)
    
    def rotate(self, keep_indices, freeze_indices):
        self.unfreeze_experts(keep_indices)
        self.freeze_experts(freeze_indices)
    
    def forward(self, x, prev_state=None):
        B, T, D = x.shape
        top_idx, top_weights, router_state = self.router(x, prev_state)
        
        output = torch.zeros_like(x)
        expert_idx = top_idx[:, :, 0]
        expert_w = top_weights[:, :, 0]
        
        # ВАЖНО: пропускаем замороженных экспертов всегда (и при обучении, и при eval)
        # Это соответствует проверенной реализации в micro_zaya_minimal.py
        for e in range(self.num_experts):
            if e in self.frozen_expert_indices:
                continue
            mask = (expert_idx == e)
            if mask.any():
                out_e = self.experts[e](x[mask])
                output[mask] += expert_w[mask].unsqueeze(-1) * out_e
        
        return output, router_state


class ResidualScaling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        return self.alpha * x + self.beta


class CCA(nn.Module):
    """Compressed Convolutional Attention."""
    
    def __init__(self, dim, num_heads, kv_heads, max_seq_len):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, dim, bias=False)
        
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
    
    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        
        rep = self.num_heads // self.kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        
        q = self.q_norm(q)
        k = self.k_norm(k)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MicroZAYABlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = CCA(config.dim, config.num_heads, config.kv_heads, config.max_seq_len)
        self.attn_scale = ResidualScaling(config.dim)
        
        self.moe_norm = nn.RMSNorm(config.dim)
        self.moe = MoELayer(config)
        self.moe_scale = ResidualScaling(config.dim)
    

    def forward(self, x, prev_state=None):
        res = x
        x = self.attn(self.attn_norm(x))
        x = self.attn_scale(x)
        x = res + x
        
        res = x
        x, state = self.moe(self.moe_norm(x), prev_state)
        x = self.moe_scale(x)
        x = res + x
        return x, state


class MicroZAYA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([MicroZAYABlock(config) for _ in range(config.num_layers)])
        self.norm_f = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.apply(self._init_weights)
        
        # === ПУЛ МИКРОЭКСПЕРТОВ ДЛЯ СТАБИЛИЗАЦИИ ===
        if HAS_MICRO_EXPERTS:
            self.micro_expert_pool = MicroExpertPool(
                dim=config.dim,
                num_micro_experts=8,
                micro_hidden=32
            )
            self.stabilization_manager = AdaptiveStabilizationManager(warmup_steps=200)
            micro_mem = self.micro_expert_pool.get_memory_usage()
            print(f"   🛡️ Микроэксперты: {micro_mem:.2f} МБ (стабилизаторы)")
        else:
            self.micro_expert_pool = None
            self.stabilization_manager = None
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        
        # === РЕЖИМ СТАБИЛИЗАЦИИ ===
        use_micro = False
        micro_weight = 0.0
        big_weight = 1.0
        
        if self.micro_expert_pool is not None and self.stabilization_manager is not None:
            use_micro = self.stabilization_manager.should_use_micro_experts()
            micro_weight = self.stabilization_manager.get_micro_weight()
            big_weight = self.stabilization_manager.get_big_weight()
        
        prev_router_state = None
        for block in self.blocks:
            x_input = x
            
            # Основной forward через большие эксперты
            x_big, prev_router_state = block(x_input, prev_router_state)
            
            # Примешиваем микроэксперты при нестабильности
            if use_micro:
                x_micro = self.micro_expert_pool(x_input, top_k=2)
                x = big_weight * x_big + micro_weight * x_micro
            else:
                x = x_big
        
        x = self.norm_f(x)
        return self.lm_head(x)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def rotate_all_experts(self, keep_indices, freeze_indices):
        for block in self.blocks:
            block.moe.rotate(keep_indices, freeze_indices)
    
    def get_expert_status(self):
        status = []
        for i, block in enumerate(self.blocks):
            active = [j for j in range(len(block.moe.experts))
                      if j not in block.moe.frozen_expert_indices]
            frozen = list(block.moe.frozen_expert_indices)
            status.append({'layer': i, 'active': active, 'frozen': frozen})
        return status
    
    def get_expert_usage_stats(self):
        """Возвращает агрегированную статистику использования экспертов.
        
        Усредняет статистику по всем слоям для глобального решения о ротации.
        
        Returns:
            dict: {expert_idx: avg_usage} где avg_usage в [0, 1]
        """
        if not self.blocks:
            return {}
        
        num_experts = self.blocks[0].moe.num_experts
        total_usage = {i: 0.0 for i in range(num_experts)}
        
        for block in self.blocks:
            stats = block.moe.router.get_usage_stats()
            for expert_idx, usage in stats.items():
                total_usage[expert_idx] += usage
        
        # Усредняем по слоям
        num_layers = len(self.blocks)
        avg_usage = {i: total_usage[i] / num_layers for i in range(num_experts)}
        
        return avg_usage
    
    def reset_expert_usage_stats(self):
        """Сбрасывает статистику использования экспертов во всех слоях."""
        for block in self.blocks:
            block.moe.router.reset_usage_stats()
    
    def pid_update_all(self, lr=0.01):
        for block in self.blocks:
            block.moe.router.pid_update(lr)


# ============================================================================
# OPTUNA
# ============================================================================

def create_objective(model_config_args, training_config, precision):
    """Создаёт objective с явной передачей precision."""
    
    dim_opts = [512, 1024]  # Производительные размеры (из бенчмарков)
    layers_min = model_config_args['layers_min']
    layers_max = model_config_args['layers_max']
    heads_opts = model_config_args['heads_options']
    exp_min = 12  # Минимум с учётом заморозки 1/3-1/2
    exp_max = 24  # Максимум с учётом памяти устройства
    hidden_opts = [256, 512]  # Оптимальные скрытые размеры
    
    def objective(trial):
        dim = trial.suggest_categorical('dim', dim_opts)
        num_layers = trial.suggest_int('num_layers', layers_min, layers_max)
        num_heads = trial.suggest_categorical('num_heads', heads_opts)
        num_experts = trial.suggest_int('num_experts', exp_min, exp_max)
        expert_hidden = trial.suggest_categorical('expert_hidden', hidden_opts)
        
        if dim % num_heads != 0:
            trial.set_user_attr('rejection_reason', 'dim % num_heads != 0')
            return float('inf')
        
        config = ModelConfig(
            dim=dim, num_layers=num_layers, num_heads=num_heads,
            kv_heads=max(1, num_heads // 2),
            num_experts=num_experts, expert_hidden=expert_hidden,
            num_active=training_config.num_active,
        )
        
        mem = estimate_memory(config, training_config, precision)
        
        target = training_config.target_params_m
        tol = training_config.params_tolerance / 100.0
        dev = abs(mem['total_params_m'] - target) / target
        
        if dev > tol:
            trial.set_user_attr('rejection_reason', f'params dev {dev:.1%}')
            trial.set_user_attr('total_params_m', mem['total_params_m'])
            return float('inf')
        
        if mem['total_mb'] > training_config.memory_limit_mb:
            trial.set_user_attr('rejection_reason', f'memory {mem["total_mb"]:.0f}MB')
            trial.set_user_attr('total_params_m', mem['total_params_m'])
            trial.set_user_attr('total_memory_mb', mem['total_mb'])
            return float('inf')
        
        try:
            model = MicroZAYA(config)
    
    # Используем BF16 для хранения на старте, конвертируем в FP16 после стабилизации
            model.eval()
            inp = torch.randint(0, config.vocab_size,
                               (training_config.batch_size, training_config.seq_length))
            with torch.no_grad():
                for _ in range(3):
                    _ = model(inp)
            
            times = []
            with torch.no_grad():
                for _ in range(10):
                    t0 = time.perf_counter()
                    _ = model(inp)
                    times.append(time.perf_counter() - t0)
            
            med = np.median(times)
            trial.set_user_attr('total_params_m', mem['total_params_m'])
            trial.set_user_attr('total_memory_mb', mem['total_mb'])
            trial.set_user_attr('median_time_ms', med * 1000)
            trial.set_user_attr('rejection_reason', 'PASSED')
            # === МНОЖИТЕЛЬ ПРОИЗВОДИТЕЛЬНОСТИ ===
            try:
                dim_val = trial.params.get('dim', 512)
                hidden_val = trial.params.get('expert_hidden', 256)
                num_experts_val = trial.params.get('num_experts', 12)
            
                if dim_val >= 1024 and hidden_val >= 512:
                    size_mult = 5.0
                elif dim_val >= 512 and hidden_val >= 256:
                    size_mult = 3.0
                else:
                    size_mult = 1.0
            
                active_experts = int(num_experts_val * 0.4)
                if 4 <= active_experts <= 8:
                    experts_mult = 1.5
                elif 2 <= active_experts <= 12:
                    experts_mult = 1.0
                else:
                    experts_mult = 0.5
            
                performance_bonus = size_mult * experts_mult
                trial.set_user_attr('performance_bonus', performance_bonus)
                trial.set_user_attr('size_mult', size_mult)
                trial.set_user_attr('experts_mult', experts_mult)
            except Exception:
                performance_bonus = 1.0

            return med / performance_bonus
        except Exception as e:
            trial.set_user_attr('rejection_reason', f'exception: {str(e)[:50]}')
            return float('inf')
        finally:
            gc.collect()
    
    return objective


def calculate_search_space(args, training_config, precision):
    """Подсчёт пространства поиска."""
    dim_opts = [512, 1024]  # Производительные размеры
    layers_range = range(args.layers_min, args.layers_max + 1)
    heads_opts = args.heads_options
    exp_range = range(args.experts_min, args.experts_max + 1)
    hidden_opts = [256, 512]  # Оптимальные скрытые размеры
    
    total = (len(dim_opts) * len(layers_range) * len(heads_opts) *
             len(exp_range) * len(hidden_opts))
    
    valid_div = 0
    for d in dim_opts:
        for h in heads_opts:
            if d % h == 0:
                valid_div += len(layers_range) * len(exp_range) * len(hidden_opts)
    
    target = training_config.target_params_m
    tol = training_config.params_tolerance / 100.0
    
    passed = 0
    for dim in dim_opts:
        for num_layers in layers_range:
            for num_heads in heads_opts:
                if dim % num_heads != 0:
                    continue
                for num_experts in exp_range:
                    for expert_hidden in hidden_opts:
                        cfg = ModelConfig(
                            dim=dim, num_layers=num_layers, num_heads=num_heads,
                            num_experts=num_experts, expert_hidden=expert_hidden,
                        )
                        mem = estimate_memory(cfg, training_config, precision)
                        dev = abs(mem['total_params_m'] - target) / target
                        if dev <= tol and mem['total_mb'] <= training_config.memory_limit_mb:
                            passed += 1
    
    return total, valid_div, passed


# ============================================================================
# ЧЕКПОИНТЫ
# ============================================================================

def auto_detect_config_from_checkpoint(checkpoint_path):
    """Определяет конфигурацию модели из метаданных чекпоинта.
    
    Возвращает словарь с гиперпараметрами или None.
    """
    meta_path = checkpoint_path.replace('.safetensors', '_meta.json')
    if not os.path.exists(meta_path):
        return None
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        config = {}
        for field in ['vocab_size', 'dim', 'num_layers', 'num_heads', 
                      'num_experts', 'expert_hidden']:
            if field in meta:
                config[field] = int(meta[field])
        
        if config:
            print(f"\n📋 Конфигурация из чекпоинта:")
            for k, v in config.items():
                print(f"   {k}: {v}")
        
        return config
    except Exception as e:
        print(f"⚠️ Не удалось прочитать метаданные: {e}")
        return None


def validate_checkpoint_by_sizes(checkpoint_path, model_config):
    """Проверяет совместимость чекпоинта по размерам весов.
    
    Используется когда метаданные в старом формате без конфигурации.
    """
    if not HAS_SAFETENSORS:
        return False
    
    try:
        from safetensors import safe_open
        
        with safe_open(checkpoint_path, framework="pt") as f:
            keys = list(f.keys())
            
            # Проверяем размерность через несколько ключевых тензоров
            
            # 1. embedding.weight: [vocab_size, dim]
            if 'embedding.weight' in keys:
                shape = f.get_slice('embedding.weight').get_shape()
                if shape[1] != model_config.dim:
                    return False
            
            # 2. q_norm.weight: [head_dim] где head_dim = dim // num_heads
            head_dim_expected = model_config.dim // model_config.num_heads
            for key in keys:
                if 'attn.q_norm.weight' in key:
                    shape = f.get_slice(key).get_shape()
                    if shape[0] != head_dim_expected:
                        return False
                    break
            
            # 3. routing_biases: [num_experts]
            for key in keys:
                if 'moe.router.routing_biases' in key:
                    shape = f.get_slice(key).get_shape()
                    if shape[0] != model_config.num_experts:
                        return False
                    break
            
            # 4. experts.0.up.weight: [expert_hidden, dim]
            for key in keys:
                if 'moe.experts.0.up.weight' in key:
                    shape = f.get_slice(key).get_shape()
                    if shape[0] != model_config.expert_hidden:
                        return False
                    break
            
            # 5. Число блоков = num_layers
            block_indices = set()
            for key in keys:
                import re
                m = re.match(r'blocks\.(\d+)\.', key)
                if m:
                    block_indices.add(int(m.group(1)))
            if len(block_indices) != model_config.num_layers:
                return False
            
            return True
            
    except Exception as e:
        # Если не удалось проверить, считаем несовместимым
        return False


def find_compatible_checkpoint(checkpoint_dir, model_config):
    """Находит последний чекпоинт, совместимый с заданной архитектурой.
    
    Проверяет метаданные каждого чекпоинта и выбирает последний,
    соответствующий заданным параметрам модели.
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    patterns = [
        os.path.join(checkpoint_dir, "micro_zaya_step*.safetensors"),
        os.path.join(checkpoint_dir, "micro_zaya_step*.pt"),
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    if not files:
        return None
    
    # Целевая архитектура
    target = {
        'dim': str(model_config.dim),
        'num_layers': str(model_config.num_layers),
        'num_heads': str(model_config.num_heads),
        'num_experts': str(model_config.num_experts),
        'expert_hidden': str(model_config.expert_hidden),
    }
    
    # Проверяем каждый чекпоинт
    compatible = []
    incompatible = []
    
    for f in files:
        m = re.search(r'step(\d+)', f)
        if not m:
            continue
        step = int(m.group(1))
        
        # Читаем метаданные
        meta = read_checkpoint_metadata(f)
        if meta is None:
            incompatible.append((step, f, "нет метаданных"))
            continue
        
        # Проверяем соответствие
        # ВАЖНО: все поля должны присутствовать в метаданных,
        # иначе чекпоинт считается несовместимым (старый формат)
        mismatches = []
        missing_fields = []
        
        for key, value in target.items():
            if key not in meta:
                missing_fields.append(key)
            elif str(meta[key]) != value:
                mismatches.append(f"{key}: {meta[key]} vs {value}")
        
        if missing_fields:
            # Старый формат метаданных - проверяем через размеры весов
            size_check = validate_checkpoint_by_sizes(f, model_config)
            if not size_check:
                incompatible.append((step, f, f"старый формат, размеры не совпадают"))
            else:
                compatible.append((step, f))
        elif mismatches:
            incompatible.append((step, f, "; ".join(mismatches)))
        else:
            compatible.append((step, f))
    
    # Логируем несовместимые
    if incompatible:
        print(f"\n⚠️ Найдено {len(incompatible)} несовместимых чекпоинтов:")
        for step, f, reason in incompatible[:5]:
            print(f"   Шаг {step}: {reason}")
        if len(incompatible) > 5:
            print(f"   ... и ещё {len(incompatible) - 5}")
    
    if not compatible:
        print(f"\n❌ Не найдено совместимых чекпоинтов")
        return None
    
    # Выбираем последний совместимый
    compatible.sort()
    best_step, best_path = compatible[-1]
    print(f"\n✅ Найден совместимый чекпоинт: шаг {best_step}")
    return best_path


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Находит последний чекпоинт по номеру шага."""
    if not os.path.exists(checkpoint_dir):
        return None
    
    patterns = [
        os.path.join(checkpoint_dir, "micro_zaya_step*.safetensors"),
        os.path.join(checkpoint_dir, "micro_zaya_step*.pt"),
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    if not files:
        return None
    
    steps = []
    for f in files:
        m = re.search(r'step(\d+)', f)
        if m:
            steps.append((int(m.group(1)), f))
    
    if not steps:
        return None
    
    steps.sort()
    return steps[-1][1]


def read_checkpoint_metadata(path):
    """Читает метаданные чекпоинта."""
    meta_path = path.replace('.safetensors', '_meta.json')
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def validate_checkpoint_config(model, meta_path):
    """Проверяет соответствие конфигурации чекпоинта и модели.
    
    Returns:
        True если совместимо, False если нет
    """
    meta = read_checkpoint_metadata(meta_path)
    if meta is None:
        print("   ⚠️ Нет метаданных для валидации")
        return True  # Не блокируем загрузку
    
    # Поля для проверки
    fields = ['vocab_size', 'dim', 'num_layers', 'num_heads', 
              'num_experts', 'expert_hidden']
    
    mismatches = []
    for field in fields:
        if field not in meta:
            continue
        ckpt_value = int(meta[field])
        model_value = getattr(model.config, field, None)
        if model_value is not None and ckpt_value != model_value:
            mismatches.append(f"{field}: {ckpt_value} (ckpt) vs {model_value} (model)")
    
    if mismatches:
        print("   ❌ НЕСООТВЕТСТВИЕ АРХИТЕКТУРЫ:")
        for m in mismatches:
            print(f"      {m}")
        print("   Чекпоинт не будет загружен, обучение начнётся с нуля.")
        return False
    
    return True


def load_checkpoint_state(model, optimizer, path):
    """Загружает состояние модели и оптимизатора с проверкой совместимости."""
    
    # ПРОВЕРКА СОВМЕСТИМОСТИ АРХИТЕКТУРЫ
    if not validate_checkpoint_config(model, path):
        print("   ⚠️ Пропускаем загрузку чекпоинта из-за несоответствия архитектуры")
        return 0
    
    try:
        if path.endswith('.safetensors') and HAS_SAFETENSORS:
            state_dict = st_load(path)
            filtered_state = {k: v for k, v in state_dict.items() if k != 'lm_head.weight'}
            model.load_state_dict(filtered_state, strict=False)
        else:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if optimizer is not None and 'optimizer_state_dict' in ckpt:
                try:
                    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                except Exception as e:
                    print(f"   ⚠️ Не восстановлен оптимизатор: {e}")
        
        # Восстанавливаем weight tying
        model.lm_head.weight = model.embedding.weight
        
        m = re.search(r'step(\d+)', path)
        step = int(m.group(1)) if m else 0
        
        print(f"   ✅ Загружен чекпоинт: {os.path.basename(path)}")
        print(f"   Продолжаем с шага: {step}")
        print(f"   ✅ Weight tying восстановлен (lm_head ↔ embedding)")
        return step
    except Exception as e:
        print(f"   ⚠️ Ошибка загрузки: {e}")
        return 0


def get_free_space_mb(path="."):
    """Возвращает свободное место на диске в МБ."""
    try:
        stat = shutil.disk_usage(path)
        return stat.free / (1024 * 1024)
    except Exception:
        # Если не удалось определить, возвращаем большое число
        return float('inf')


def cleanup_old_checkpoints(checkpoint_dir, keep_latest=3):
    """Удаляет старые чекпоинты, оставляя N последних.
    
    Возвращает число удалённых файлов и освобождённое место в МБ.
    """
    if not os.path.exists(checkpoint_dir):
        return 0, 0
    
    # Ищем все чекпоинты
    patterns = [
        os.path.join(checkpoint_dir, "micro_zaya_step*.safetensors"),
        os.path.join(checkpoint_dir, "micro_zaya_step*.pt"),
    ]
    
    files = []
    for p in patterns:
        files.extend(glob.glob(p))
    
    if not files:
        return 0, 0
    
    # Сортируем по номеру шага
    steps = []
    for f in files:
        m = re.search(r'step(\d+)', f)
        if m:
            steps.append((int(m.group(1)), f))
    
    steps.sort()
    
    if len(steps) <= keep_latest:
        return 0, 0
    
    to_delete = steps[:-keep_latest]
    deleted_count = 0
    freed_mb = 0
    
    for step, path in to_delete:
        try:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            os.remove(path)
            
            # Также удаляем метафайл
            meta = path.replace('.safetensors', '_meta.json')
            if os.path.exists(meta):
                os.remove(meta)
            
            deleted_count += 1
            freed_mb += size_mb
        except Exception:
            pass
    
    return deleted_count, freed_mb


def estimate_checkpoint_size_mb(model):
    """Оценивает размер чекпоинта в МБ."""
    params = model.get_num_params()
    # FP32: 4 байта на параметр
    # + копия lm_head.weight (из-за обхода shared memory)
    extra_mb = (model.config.vocab_size * model.config.dim) * 4 / (1024 * 1024)
    return params * 4 / (1024 * 1024) + extra_mb


def save_checkpoint(model, step, losses, path_prefix, keep_latest=3):
    """Сохраняет чекпоинт с обработкой ошибок записи.
    
    Возвращает путь к сохранённому файлу или None при ошибке.
    Обучение НЕ прерывается при ошибке сохранения.
    """
    dir_path = os.path.dirname(path_prefix) or "."
    
    # === ШАГ 1: Проверяем свободное место ===
    estimated_mb = estimate_checkpoint_size_mb(model)
    free_mb = get_free_space_mb(dir_path)
    
    # Если места мало, пробуем очистить старые чекпоинты
    if free_mb < estimated_mb * 1.5:
        print(f"\n⚠️ Мало места: {free_mb:.0f} МБ свободно, нужно ~{estimated_mb:.0f} МБ")
        deleted, freed = cleanup_old_checkpoints(dir_path, keep_latest=keep_latest)
        if deleted > 0:
            free_mb = get_free_space_mb(dir_path)
            print(f"   🗑️ Удалено {deleted} старых чекпоинтов, освобождено ~{freed:.0f} МБ")
            print(f"   Теперь свободно: {free_mb:.0f} МБ")
    
    # Если всё ещё мало, пропускаем сохранение
    if free_mb < estimated_mb * 1.2:
        print(f"\n⚠️{'='*60}")
        print(f"⚠️ НЕ ХВАТАЕТ МЕСТА ДЛЯ ЧЕКПОИНТА")
        print(f"⚠️{'='*60}")
        print(f"   Нужно: ~{estimated_mb:.0f} МБ")
        print(f"   Свободно: {free_mb:.0f} МБ")
        print(f"   Освободите место на устройстве:")
        print(f"     - Удалите ненужные файлы")
        print(f"     - Переместите старые чекпоинты")
        print(f"     - Очистите кэш приложений")
        print(f"   Обучение продолжается без сохранения.")
        print(f"⚠️{'='*60}\n")
        return None
    
    # === ШАГ 2: Подготавливаем данные ===
    try:
        os.makedirs(dir_path, exist_ok=True)
        
        # Собираем информацию о замороженных экспертах
        frozen_info = []
        for block in model.blocks:
            if hasattr(block, 'moe'):
                frozen_info.append(sorted(list(block.moe.frozen_expert_indices)))
        
        # Собираем информацию о конфигурации модели
        config_info = {
            'vocab_size': str(model.config.vocab_size),
            'dim': str(model.config.dim),
            'num_layers': str(model.config.num_layers),
            'num_heads': str(model.config.num_heads),
            'kv_heads': str(model.config.kv_heads),
            'num_experts': str(model.config.num_experts),
            'expert_hidden': str(model.config.expert_hidden),
        }
        
        metadata = {
            'step': str(step),
            'final_loss': str(losses[-1]) if losses else '0',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            **config_info,
        }
        
        # Развязываем веса для совместимости с safetensors
        state_dict = {}
        for k, v in model.state_dict().items():
            if k == 'lm_head.weight':
                state_dict[k] = v.detach().cpu().clone()
            else:
                state_dict[k] = v.detach().cpu()
        
        # === ШАГ 3: Сохраняем с обработкой ошибок ===
        path = None
        try:
            if HAS_SAFETENSORS:
                path = f"{path_prefix}_step{step}.safetensors"
                st_save(state_dict, path, metadata=metadata)
                
                meta_path = f"{path_prefix}_step{step}_meta.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        'step': step,
                        'losses_last_10': losses[-10:] if len(losses) >= 10 else losses,
                        'timestamp': metadata['timestamp'],
                        'frozen_experts': frozen_info,
                    }, f, indent=2)
            else:
                path = f"{path_prefix}_step{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': state_dict,
                    'losses': losses,
                    'frozen_experts': frozen_info,
                }, path)
            
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"   💾 Чекпоинт: {path} ({size_mb:.1f} МБ)")
            return path
            
        except OSError as e:
            # Обработка ошибок записи (включая "нет места")
            if e.errno == 28:  # ENOSPC: No space left on device
                print(f"\n⚠️{'='*60}")
                print(f"⚠️ НЕ ХВАТАЕТ МЕСТА ПРИ ЗАПИСИ ЧЕКПОИНТА")
                print(f"⚠️{'='*60}")
                print(f"   Освободите место на устройстве.")
                print(f"   Обучение продолжается без сохранения.")
                print(f"⚠️{'='*60}\n")
                
                # Удаляем частичный файл, если он был создан
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        print(f"   🗑️ Удалён частичный файл: {path}")
                    except Exception:
                        pass
                
                return None
            else:
                raise
        
    except Exception as e:
        print(f"\n⚠️ Ошибка сохранения чекпоинта: {e}")
        print(f"   Обучение продолжается без сохранения.")
        return None


# ============================================================================
# ОБУЧЕНИЕ
# ============================================================================

def get_memory_info():
    """Возвращает информацию о памяти в МБ."""
    try:
        mem = psutil.virtual_memory()
        return {
            'total_mb': mem.total / 1024 / 1024,
            'available_mb': mem.available / 1024 / 1024,
            'used_mb': mem.used / 1024 / 1024,
            'percent': mem.percent,
        }
    except Exception:
        return {'total_mb': 0, 'available_mb': 0, 'used_mb': 0, 'percent': 0}


def load_memory_state(checkpoint_dir):
    """Загружает сохранённое состояние памяти (% активных экспертов)."""
    state_path = os.path.join(checkpoint_dir, 'memory_state.json')
    if os.path.exists(state_path):
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            print(f"\n📊 Загружено состояние памяти: {state.get('active_percent', 33)}% активных экспертов")
            return state
        except Exception as e:
            print(f"⚠️ Не удалось загрузить состояние памяти: {e}")
    return {'active_percent': 33, 'oom_count': 0}


def save_memory_state(checkpoint_dir, active_percent, oom_count=0, reason=''):
    """Сохраняет состояние памяти для следующего запуска."""
    state_path = os.path.join(checkpoint_dir, 'memory_state.json')
    try:
        os.makedirs(checkpoint_dir, exist_ok=True)
        state = {
            'active_percent': active_percent,
            'oom_count': oom_count,
            'reason': reason,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'memory_info': get_memory_info(),
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"   💾 Состояние памяти сохранено: {active_percent}% активных")
    except Exception as e:
        print(f"⚠️ Не удалось сохранить состояние памяти: {e}")


def check_memory_pressure(threshold_percent=85):
    """Проверяет, есть ли давление на память.
    
    Returns:
        'ok' — память в норме
        'warning' — память выше порога, нужно уменьшить активных экспертов
        'critical' — память критична, нужно сохранить и выйти
    """
    mem = get_memory_info()
    if mem['percent'] >= 95:
        return 'critical', mem
    elif mem['percent'] >= threshold_percent:
        return 'warning', mem
    else:
        return 'ok', mem


def adjust_router_bias_towards_underused(model, sorted_experts, num_freeze, lr=0.2):
    """Корректирует смещение роутера для отстающих экспертов."""
    underused = [e for e, _ in sorted_experts[:-num_freeze]]
    
    if not underused:
        return
    
    for block in model.blocks:
        if not hasattr(block, 'moe'):
            continue
        
        router = block.moe.router
        with torch.no_grad():
            # Увеличиваем bias для отстающих
            for expert_idx in underused:
                if expert_idx < len(router.routing_biases):
                    router.routing_biases[expert_idx] += lr
            
            # Уменьшаем bias для замороженных
            frozen = [e for e, _ in sorted_experts[-num_freeze:]]
            for expert_idx in frozen:
                if expert_idx < len(router.routing_biases):
                    router.routing_biases[expert_idx] -= lr * 0.5


def train_micro_zaya(config, training_config, corpus_path="corpus.tok16",
                     resume_from=None, args=None):
    """Основная функция обучения."""
    
    print("=" * 80)
    print("🚀 ОБУЧЕНИЕ MICRO-ZAYA")
    print("=" * 80)
    
    mem = estimate_memory(config, training_config, training_config.precision)
    print(f"\n📊 ОЦЕНКА ПАМЯТИ:")
    print(f"   Точность: {mem['precision']} ({mem['bytes_per_param']} байт/параметр)")
    print(f"   Всего параметров: {mem['total_params_m']:.1f}M")
    print(f"   Веса: {mem['weights_mb']:.1f} МБ")
    print(f"   Градиенты: {mem['gradients_mb']:.1f} МБ")
    print(f"   AdamW: {mem['adamw_mb']:.1f} МБ")
    print(f"   Активации: {mem['activations_mb']:.1f} МБ")
    print(f"   ИТОГО: {mem['total_mb']:.1f} МБ")
    
    if mem['total_mb'] > training_config.memory_limit_mb:
        print(f"\n⚠️ Память превышает лимит {training_config.memory_limit_mb} МБ")
        return None, []
    
    model = MicroZAYA(config)
    print(f"\nМодель создана: {model.get_num_params()/1e6:.1f}M параметров")
    
    # Приведение к точности
    precision = training_config.precision
    if precision == 'fp16':
        model = model.half()
        print("   ✅ Модель в FP16")
    elif precision == 'bf16':
        model = model.to(torch.bfloat16)
        print("   ✅ Модель в BF16")
    else:
        print("   ✅ Модель в FP32")
    
    # ВАЖНО: разрешаем градиентам быть любого типа (для свежих версий PyTorch)
    grad_dtype_set = 0
    for p in model.parameters():
        if hasattr(p, 'grad_dtype'):
            p.grad_dtype = None
            grad_dtype_set += 1
    if grad_dtype_set > 0:
        print(f"   ✅ grad_dtype=None установлен для {grad_dtype_set} параметров")
    
    # Оптимизатор
    if create_muon_optimizer is not None:
        optimizer, mc, ac = create_muon_optimizer(
            model, muon_lr=training_config.muon_lr, adamw_lr=training_config.adamw_lr
        )
        print(f"Оптимизатор: Muon ({mc}), AdamW ({ac})")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.adamw_lr)
        print("Оптимизатор: AdamW (Muon недоступен)")
    
    # Загрузка чекпоинта
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        print(f"\n📂 Продолжение из: {resume_from}")
        
        # Валидация архитектуры ПЕРЕД загрузкой
        if not validate_checkpoint_config(model, resume_from):
            print("   ⚠️ Чекпоинт не соответствует архитектуре модели")
            print("   Начинаем обучение с нуля")
            resume_from = None
        else:
            start_step = load_checkpoint_state(model, optimizer, resume_from)
        
        # Восстановление заморозки экспертов
        meta_path = resume_path_to_meta(resume_from)
        if meta_path and os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                frozen_info = meta.get('frozen_experts', [])
                if frozen_info and len(frozen_info) == len(model.blocks):
                    for i, block in enumerate(model.blocks):
                        frozen = frozen_info[i]
                        keep = list(set(range(config.num_experts)) - set(frozen))
                        block.moe.rotate(keep, frozen)
                    print("   ✅ Состояние заморозки восстановлено")
            except Exception as e:
                print(f"   ⚠️ Не восстановлена заморозка: {e}")
    
    # Корпус
    corpus = None
    if os.path.exists(corpus_path):
        corpus = np.memmap(corpus_path, dtype=np.uint16, mode='r')
        print(f"Корпус: {len(corpus):,} токенов")
    else:
        print(f"⚠️ Корпус не найден: {corpus_path}, случайные данные")
    
    # Инференс-семплер
    inference_sampler = None
    if HAS_INFERENCE_SAMPLER and args is not None:
        inference_sampler = InferenceSampler(
            model,
            vocab_path=args.vocab_file,
            seed=args.inference_seed,
            max_gen_tokens=args.inference_tokens,
        )
        print("✅ Семплер инференса активен")
    
    # Трекер ротации
    rotation_tracker = None
    if HAS_ROTATION_TRACKER:
        rotation_tracker = RotationTracker(model)
        print("✅ Трекер ротации активен")
    
    # Обучение
    # === ИНИЦИАЛИЗАЦИЯ АДАПТИВНОЙ РОТАЦИИ И СТАБИЛИЗАЦИИ ===
    # ВАЖНО: инициализируем ДО цикла, чтобы первый шаг не падал
    
    # 1. Процент активных экспертов
    if not hasattr(train_micro_zaya, '_active_percent'):
        memory_state = load_memory_state(args.checkpoint_dir)
        train_micro_zaya._active_percent = memory_state.get(
            'active_percent', args.initial_active_percent)
    
    # 2. Менеджер микроэкспертов
    
    # 3. Счётчик NaN
    train_micro_zaya._consecutive_nan = 0
    train_micro_zaya._last_valid_step = 0
    
    # 4. Начальный процент активных
    print(f"   📊 Начальный % активных экспертов: {train_micro_zaya._active_percent}%")
    
    # === ПРЕДЗАГРУЗКА ПЕРВОГО БАТЧА (ускорение старта) ===
    print(f"   📥 Предзагрузка первого батча из корпуса...")
    import time as _time
    _t0 = _time.perf_counter()
    
    # Читаем первый батч, чтобы прогреть файловый кэш
    try:
        with open(corpus_path, 'rb') as f:
            # Читаем небольшой объём для прогрева кэша
            _ = f.read(1024 * 1024)  # 1 МБ
        print(f"   ✅ Корпус прогрет за {_time.perf_counter()-_t0:.2f}с")
    except Exception as e:
        print(f"   ⚠️ Не удалось прогреть корпус: {e}")
    
    # === ПЛАНИРОВЩИК ТОЧНОСТИ ===
    # Старт в BF16 для стабилизации, конвертация в FP16 после
    precision_scheduler = PrecisionScheduler(
        stabilization_steps=200,
        good_steps_threshold=100,
        loss_threshold=9.0
    )
    
    # === УСТАНОВКА НАЧАЛЬНОГО ШАГА ПЛАНИРОВЩИКА ===
    # Критично для возобновления с чекпоинта
    if start_step > 0:
        if hasattr(precision_scheduler, 'current_step'):
            precision_scheduler.current_step = start_step
            print(f"   ✅ Планировщик: установлен шаг {start_step}")
        if hasattr(precision_scheduler, 'step'):
            precision_scheduler.step = start_step
            print(f"   ✅ Планировщик: шаг установлен в {start_step}")
        
        # Если шаг больше 200 (фаза стабилизации завершена), устанавливаем фазу обучения
        if start_step >= 200:
            if hasattr(precision_scheduler, 'phase'):
                precision_scheduler.phase = 'training'
            if hasattr(precision_scheduler, 'current_phase'):
                precision_scheduler.current_phase = 'training'
            print(f"   ✅ Фаза установлена: 'training' (шаг {start_step} >= 200)")

    
    print(f"\n🏃 Начало обучения ({training_config.max_steps} шагов)...")
    model.train()
    losses = []
    step_times = []
    baseline_loss = None
    
    pbar_range = range(start_step, training_config.max_steps)
    if HAS_TQDM:
        pbar = tqdm(pbar_range, desc="Обучение", unit="шаг", dynamic_ncols=True,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    else:
        pbar = pbar_range
    
    for step in pbar:
        t0 = time.perf_counter()
        
        # Батч
        if corpus is not None:
            seq_len = training_config.seq_length
            bs = training_config.batch_size
            inp = np.zeros((bs, seq_len), dtype=np.int64)
            for i in range(bs):
                s = np.random.randint(0, max(1, len(corpus) - seq_len - 1))
                inp[i] = corpus[s:s + seq_len]
            input_ids = torch.from_numpy(inp)
        else:
            input_ids = torch.randint(0, config.vocab_size,
                                     (training_config.batch_size, training_config.seq_length))
        
        labels = input_ids.clone()
        
        optimizer.zero_grad()
        logits = model(input_ids)
        # === ЗАЩИТА ОТ ПЕРЕПОЛНЕНИЯ В ЛОССЕ ===
        # Приводим логиты к FP32 и ограничиваем диапазон
        logits_for_loss = logits[:, :-1, :].reshape(-1, config.vocab_size).float()
        
        # Clamping для предотвращения переполнения в softmax
        logits_for_loss = torch.clamp(logits_for_loss, min=-20.0, max=20.0)
        
        # Замена существующих NaN/Inf на нули
        if torch.isnan(logits_for_loss).any() or torch.isinf(logits_for_loss).any():
            logits_for_loss = torch.nan_to_num(logits_for_loss, nan=0.0, posinf=20.0, neginf=-20.0)
        
        loss = F.cross_entropy(
            logits_for_loss,
            labels[:, 1:].reshape(-1),
            label_smoothing=0.1
        )
        
        if torch.isnan(loss) or torch.isinf(loss):
            # === СЧЁТЧИК NaN ПОДРЯД ===
            if not hasattr(train_micro_zaya, '_consecutive_nan'):
                train_micro_zaya._consecutive_nan = 0
            train_micro_zaya._consecutive_nan += 1

            # === ПЕРЕИНИЦИАЛИЗАЦИЯ МИКРОЭКСПЕРТОВ ПРИ NaN ===
            if HAS_MICRO_EXPERTS and hasattr(model, 'micro_expert_pool') and model.micro_expert_pool is not None:
                for expert in model.micro_expert_pool.experts:
                    expert.reinit_weights()
                print(f"   🔄 Микроэксперты переинициализированы")

            if HAS_MICRO_EXPERTS and hasattr(model, 'stabilization_manager') and model.stabilization_manager is not None:
                model.stabilization_manager.mode = 'micro_only'
                model.stabilization_manager.stability_score = 0.0
                print(f"   🟡 Режим: микро + большие (стабилизация)")
            print(f"⚠️ NaN/Inf на шаге {step} (подряд: {train_micro_zaya._consecutive_nan})")


            # === ОБРАБОТКА NaN С МИКРОЭКСПЕРТАМИ ===
            print(f"   🔄 Переинициализация микроэкспертов")
            
            
            # Обнуляем градиенты
            optimizer.zero_grad()

            # Если слишком много NaN подряд — сохраняем и выходим
            if train_micro_zaya._consecutive_nan >= 10:
                print(f"\n💾 {train_micro_zaya._consecutive_nan} NaN подряд, сохраняем чекпоинт")
                break

            # Пропускаем шаг, продолжаем обучение
            continue
        
        # Сбрасываем счётчик при валидном шаге
        if hasattr(train_micro_zaya, '_consecutive_nan'):
            train_micro_zaya._consecutive_nan = 0
        
        # === ЗАПИСЬ В ПЛАНИРОВЩИК ТОЧНОСТИ ===
        precision_scheduler.record_step(loss.item())
        
        # === ПРОВЕРКА КОНВЕРТАЦИИ В FP16 ===
        if precision_scheduler.should_convert():
            model = precision_scheduler.convert_to_fp16(model)
            
            # Обновляем оптимизатор для новой модели
            # Используем значение по умолчанию, если атрибут отсутствует
            weight_decay_value = getattr(training_config, 'weight_decay', 0.01)
            optimizer, _, _ = create_muon_optimizer(
                model,
                muon_lr=training_config.muon_lr,
                adamw_lr=training_config.adamw_lr,
                weight_decay=weight_decay_value
            )
            
            # Обновляем флаг точности
            training_config.precision = 'fp16'
            print(f"   📊 Точность переключена на {training_config.precision}")

        loss.backward()

        # ВАЖНО: для FP16/BF16 приводим градиенты к типу параметров
        precision = training_config.precision
        if precision in ('fp16', 'bf16'):
            target_dtype = torch.float16 if precision == 'fp16' else torch.bfloat16
            for p in model.parameters():
                if p.grad is not None and p.grad.dtype != target_dtype:
                    try:
                        if hasattr(p, 'grad_dtype'):
                            p.grad_dtype = None
                        p.grad = p.grad.to(target_dtype)
                    except Exception:
                        pass
        
        # Примечание: градиенты уже приведены к типу параметров выше.
        # Дополнительное приведение к FP32 убрано для избежания конфликта типов.
        # Muon работает с тем типом, который есть у параметров.
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # === ЗАПИСЬ В МЕНЕДЖЕР СТАБИЛИЗАЦИИ ===
        if hasattr(model, 'stabilization_manager') and model.stabilization_manager is not None:
            # Вычисляем grad norm для оценки стабильности
            grad_norm_record = 0.0
            for p_rec in model.parameters():
                if p_rec.grad is not None:
                    grad_norm_record += p_rec.grad.norm().item() ** 2
            grad_norm_record = grad_norm_record ** 0.5
            model.stabilization_manager.record_step(loss.item(), grad_norm_record)
            
            # Логируем статус каждые 25 шагов
            if step % 25 == 0:
                status = model.stabilization_manager.get_status()
                mode_emoji = {'micro_only': '🟡', 'micro_plus_big': '🟠', 'big_only': '🟢'}
                print(f"   {mode_emoji.get(status['mode'], '⚪')} Режим: {status['mode']}, "
                      f"стабильность: {status['stability']:.2f}")
        
        if step % 10 == 0:
            model.pid_update_all(lr=0.01)
        
        # Ротация экспертов
        # Ротация экспертов
        # Ротация экспертов
        if step > 0 and step % training_config.rotation_interval == 0:
            print(f"\n🔄 Ротация на шаге {step}")
            
            # Стратегия: инвертированная умная ротация
            # ЗАМОРАЖИВАЕМ наиболее используемые (они уже обучены, знания сохраняются)
            # ДООБУЧАЕМ наименее используемые (им нужен шанс)
            usage_stats = model.get_expert_usage_stats()
            
            # Проверяем, есть ли данные
            total_usage = sum(usage_stats.values())
            if total_usage < 0.01:
                # Первая ротация после загрузки: используем случайную
                print(f"   ℹ️ Первая ротация: используем случайную стратегию")
                num_keep = max(1, config.num_experts // 3)
                all_idx = list(range(config.num_experts))
                np.random.shuffle(all_idx)
                keep = all_idx[:num_keep]
                freeze = all_idx[num_keep:]
                model.rotate_all_experts(keep, freeze)
                
                for status in model.get_expert_status():
                    print(f"   Слой {status['layer']}: активны {status['active']}, "
                          f"заморожены {status['frozen']}")
                
                # ВАЖНО: инициализируем долгосрочную статистику для следующей ротации
                # (чтобы на следующем шаге была реальная статистика)
                continue
            
            # Сортируем экспертов по использованию (от меньшего к большему)
            sorted_experts = sorted(usage_stats.items(), key=lambda x: x[1])
            
            # === АДАПТИВНЫЙ РАСЧЁТ ЧИСЛА ЗАМОРОЖЕННЫХ ЭКСПЕРТОВ ===
            # Читаем текущий процент активных из состояния памяти
            if not hasattr(train_micro_zaya, '_active_percent'):
                train_micro_zaya._active_percent = getattr(args, 'initial_active_percent', 33)
            
            active_percent = train_micro_zaya._active_percent
            num_active = max(1, int(config.num_experts * active_percent / 100))
            num_freeze = config.num_experts - num_active
            
            # Проверяем давление памяти перед ротацией
            mem_status, mem_info = check_memory_pressure(
                threshold_percent=getattr(args, 'memory_threshold_percent', 85))
            
            if mem_status == 'warning':
                # Уменьшаем процент активных экспертов на 10
                new_percent = max(getattr(args, 'min_active_percent', 15),
                                  active_percent - 10)
                if new_percent < active_percent:
                    print(f"\n⚠️ Давление на память: {mem_info['percent']:.0f}%")
                    print(f"   Уменьшаем активных экспертов: {active_percent}% → {new_percent}%")
                    train_micro_zaya._active_percent = new_percent
                    active_percent = new_percent
                    num_active = max(1, int(config.num_experts * active_percent / 100))
                    num_freeze = config.num_experts - num_active
            
            elif mem_status == 'critical':
                # Критическая нехватка памяти: сохраняем и выходим
                print(f"\n🚨 КРИТИЧЕСКАЯ НЕХВАТКА ПАМЯТИ: {mem_info['percent']:.0f}%")
                print(f"   Свободно: {mem_info['available_mb']:.0f} МБ")
                
                # Сохраняем чекпоинт
                save_checkpoint(model, step, losses,
                               path_prefix=f"{args.checkpoint_dir}/micro_zaya",
                               keep_latest=args.max_checkpoints)
                
                # Сохраняем состояние с уменьшенным процентом
                new_percent = max(getattr(args, 'min_active_percent', 15),
                                  active_percent - 15)
                save_memory_state(args.checkpoint_dir, new_percent,
                                  oom_count=getattr(train_micro_zaya, '_oom_count', 0) + 1,
                                  reason='critical_memory')
                
                print(f"   Выход для перезапуска с {new_percent}% активных экспертов")
                sys.exit(42)  # Специальный код для перезапуска
            
            # ЗАМОРАЖИВАЕМ наиболее используемые (1/3)
            freeze = [e for e, _ in sorted_experts[-num_freeze:]]
            
            # ДООБУЧАЕМ наименее используемые (2/3)
            keep = [e for e, _ in sorted_experts[:-num_freeze]]
            
            # Логируем статистику
            print(f"   Использование экспертов:")
            print(f"   🔒 Заморожены (наиболее используемые, знания сохранены):")
            for e, usage in sorted_experts[-num_freeze:]:
                print(f"      Эксперт {e}: {usage*100:.1f}%")
            print(f"   📚 Активны (наименее используемые, дообучение):")
            for e, usage in sorted_experts[:min(5, len(sorted_experts)-num_freeze)]:
                print(f"      Эксперт {e}: {usage*100:.1f}%")
            
            model.rotate_all_experts(keep, freeze)
            
            # Логируем состояние после ротации
            total_active = 0
            total_frozen = 0
            for status in model.get_expert_status():
                total_active += len(status['active'])
                total_frozen += len(status['frozen'])
                print(f"   Слой {status['layer']}: активны {status['active']}, "
                      f"заморожены {status['frozen']}")
            
            print(f"   Итого: {total_active} активных, {total_frozen} замороженных")
            
            # === ЭКОНОМИЯ ПАМЯТИ ОТ ЗАМОРОЗКИ ===
            # Это главная цель ротации: больше моделей влезает в память смартфона
            frozen_params = 0
            for block in model.blocks:
                if hasattr(block, 'moe'):
                    for idx in block.moe.frozen_expert_indices:
                        if idx < len(block.moe.experts):
                            frozen_params += sum(p.numel() for p in block.moe.experts[idx].parameters())
            
            # Состояние оптимизатора: 
            # - Adam: 8 байт/параметр (моментум + дисперсия)
            # - Полный чекпоинт: 12 байт/параметр (веса + моментум + дисперсия)
            memory_saved_optim_mb = frozen_params * 8 / 1024 / 1024
            memory_saved_full_mb = frozen_params * 12 / 1024 / 1024
            
            # Процент от общего числа параметров
            total_params = model.get_num_params()
            frozen_percent = frozen_params / total_params * 100
            
            print(f"   💾 ЭКОНОМИЯ ПАМЯТИ:")
            print(f"      Заморожено параметров: {frozen_params/1e6:.1f}M ({frozen_percent:.0f}% от модели)")
            print(f"      Состояние оптимизатора: ~{memory_saved_optim_mb:.1f} МБ")
            print(f"      Полный чекпоинт: ~{memory_saved_full_mb:.1f} МБ")
            
            # Корректируем роутер для отстающих экспертов
            adjust_router_bias_towards_underused(model, sorted_experts, num_freeze)
            
            # Сбрасываем статистику для следующего интервала
            model.reset_expert_usage_stats()
            print(f"   Статистика сброшена, роутер скорректирован")
            
        losses.append(loss.item())
        
        # === ПЕРИДИЧЕСКАЯ ПРОВЕРКА ПАМЯТИ (каждые 10 шагов) ===
        if step % 10 == 0:
            mem_status, mem_info = check_memory_pressure(
                threshold_percent=getattr(args, 'memory_threshold_percent', 85))
            
            if mem_status == 'critical':
                print(f"\n🚨 КРИТИЧЕСКАЯ НЕХВАТКА ПАМЯТИ на шаге {step}")
                print(f"   Использование: {mem_info['percent']:.0f}%")
                print(f"   Свободно: {mem_info['available_mb']:.0f} МБ")
                
                # Сохраняем чекпоинт
                save_checkpoint(model, step, losses,
                               path_prefix=f"{args.checkpoint_dir}/micro_zaya",
                               keep_latest=args.max_checkpoints)
                
                # Сохраняем состояние с уменьшенным процентом
                current_percent = getattr(train_micro_zaya, '_active_percent', 33)
                new_percent = max(getattr(args, 'min_active_percent', 20),
                                  current_percent - 15)
                save_memory_state(args.checkpoint_dir, new_percent,
                                  oom_count=getattr(train_micro_zaya, '_oom_count', 0) + 1,
                                  reason='critical_memory_during_training')
                
                print(f"   Выход для перезапуска с {new_percent}% активных")
                sys.exit(42)
        dt = time.perf_counter() - t0
        step_times.append(dt)
        
        if baseline_loss is None and len(losses) >= 10:
            baseline_loss = np.mean(losses[:10])
        
        # Прогресс-бар
        if HAS_TQDM and hasattr(pbar, 'set_postfix'):
            avg = np.mean(losses[-10:]) if losses else loss.item()
            pbar.set_postfix({'loss': f"{avg:.4f}"})
        
        # Логирование каждые 50 шагов
        if step % 50 == 0:
            avg = np.mean(losses[-50:]) if len(losses) >= 50 else loss.item()
            avg_time = np.mean(step_times[-50:]) * 1000 if len(step_times) >= 50 else dt * 1000
            msg = f"Шаг {step:5d} | Loss: {avg:.4f} | {avg_time:.1f} мс"
            if HAS_TQDM:
                tqdm.write(f"\n   {msg}")
            else:
                print(f"   {msg}")
        
        # Инференс
        if (args is not None and inference_sampler is not None and
            step > 0 and args.inference_interval > 0 and
            step % args.inference_interval == 0):
            if HAS_TQDM:
                tqdm.write(f"\n   📝 Инференс на шаге {step}")
            else:
                print(f"\n   📝 Инференс на шаге {step}")
            if args.full_inference:
                inference_sampler.sample_full(step)
            else:
                inference_sampler.sample_all(step)
        
        # Чекпоинт
        if (args is not None and args.save_every > 0 and
            step > 0 and step % args.save_every == 0):
            save_checkpoint(model, step, losses,
                           path_prefix=f"{args.checkpoint_dir}/micro_zaya",
                           keep_latest=args.max_checkpoints)
    
    if HAS_TQDM and hasattr(pbar, 'close'):
        pbar.close()
    
    # Финальный чекпоинт
    if args is not None and losses:
        save_checkpoint(model, training_config.max_steps, losses,
                       path_prefix=f"{args.checkpoint_dir}/micro_zaya_final",
                       keep_latest=args.max_checkpoints)
    
    # Финальный отчёт о ротации
    print(f"\n📊 ОТЧЁТ О РОТАЦИИ ЭКСПЕРТОВ")
    print("-" * 80)
    
    total_active = 0
    total_frozen = 0
    for status in model.get_expert_status():
        total_active += len(status['active'])
        total_frozen += len(status['frozen'])
        print(f"   Слой {status['layer']}: активны {status['active']}, "
              f"заморожены {status['frozen']}")
    
    print(f"\n   Итого: {total_active} активных, {total_frozen} замороженных")
    
    # Экономия памяти
    frozen_params = 0
    for block in model.blocks:
        if hasattr(block, 'moe'):
            for idx in block.moe.frozen_expert_indices:
                if idx < len(block.moe.experts):
                    frozen_params += sum(p.numel() for p in block.moe.experts[idx].parameters())
    memory_saved_mb = frozen_params * 8 / 1024 / 1024
    print(f"   Экономия памяти: ~{memory_saved_mb:.1f} МБ")
    
    # Сохраняем историю через rotation_tracker
    if rotation_tracker:
        rotation_tracker.save_history()
        rotation_tracker.print_summary()
    
    print(f"\n{'='*80}")
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'='*80}")
    if losses:
        print(f"   Финальный loss: {losses[-1]:.4f}")
        print(f"   Среднее время: {np.mean(step_times)*1000:.1f} мс")
        if baseline_loss:
            print(f"   Улучшение: {baseline_loss - losses[-1]:.4f}")
    
    return model, losses


def resume_path_to_meta(path):
    """Возвращает путь к meta.json для чекпоинта."""
    if path.endswith('.safetensors'):
        return path.replace('.safetensors', '_meta.json')
    return None


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = build_parser()
    args = parser.parse_args()
    
    # === АВТООПРЕДЕЛЕНИЕ КОНФИГУРАЦИИ ИЗ ЧЕКПОИНТА ===
    # Должно выполняться ДО любых проверок и создания ModelConfig
    
    # Определяем доступность модуля автодетекции ПРЯМО ЗДЕСЬ
    try:
        from checkpoint_autodetect import (
            detect_config_from_checkpoint,
            find_latest_checkpoint,
            save_config_to_checkpoint,
            load_config_from_checkpoint,
        )
        HAS_AUTODETECT = True
    except ImportError:
        HAS_AUTODETECT = False
    
    detected_config = None
    if HAS_AUTODETECT and args.checkpoint_dir:
        import os
        if os.path.exists(args.checkpoint_dir):
            latest_ckpt = find_latest_checkpoint(args.checkpoint_dir)
            if latest_ckpt:
                print(f"\n🔍 Найден чекпоинт: {latest_ckpt}")
                
                # Сначала пробуем загрузить сохранённую конфигурацию
                detected_config = load_config_from_checkpoint(latest_ckpt)
                
                if not detected_config:
                    # Если нет сохранённой, детектим из тензоров
                    detected_config = detect_config_from_checkpoint(latest_ckpt)
                
                if detected_config:
                    print(f"   📊 Автоопределённая конфигурация:")
                    for k, v in sorted(detected_config.items()):
                        if not k.startswith('_'):
                            print(f"      {k}: {v}")
                    
                    # Применяем детектированные значения к аргументам
                    # Только если аргументы не были явно заданы
                    # При авто-резюме всегда используем параметры из чекпоинта
                    if 'dim' in detected_config:
                        args.dim = detected_config['dim']
                        print(f"   ✅ args.dim = {args.dim}")
                    
                    if 'num_layers' in detected_config:
                        args.num_layers = detected_config['num_layers']
                        print(f"   ✅ args.num_layers = {args.num_layers}")
                    
                    if 'num_heads' in detected_config:
                        args.num_heads = detected_config['num_heads']
                        print(f"   ✅ args.num_heads = {args.num_heads}")
                    
                    if 'num_experts' in detected_config:
                        args.num_experts = detected_config['num_experts']
                        print(f"   ✅ args.num_experts = {args.num_experts}")
                    
                    if 'expert_hidden' in detected_config:
                        args.expert_hidden = detected_config['expert_hidden']
                        print(f"   ✅ args.expert_hidden = {args.expert_hidden}")
                    
                    print(f"   ✅ Параметры обновлены из чекпоинта")
                else:
                    print(f"   ⚠️ Не удалось определить конфигурацию из чекпоинта")
    
    print("=" * 80)
    print("🎯 MICRO-ZAYA: ПОИСК И ОБУЧЕНИЕ")
    print("=" * 80)
    
    # Настройка диапазона экспертов
    experts_min = args.experts_min
    experts_max = args.experts_max
    if args.num_experts is not None:
        experts_min = max(4, args.num_experts - 8)
        experts_max = args.num_experts + 8
        print(f"\n📌 Задано экспертов: {args.num_experts} (диапазон [{experts_min}, {experts_max}])")
    
    model_config_args = {
        'dim_options': args.dim_options,
        'layers_min': args.layers_min,
        'layers_max': args.layers_max,
        'heads_options': args.heads_options,
        'experts_min': experts_min,
        'experts_max': experts_max,
        'expert_hidden_options': args.expert_hidden_options,
    }
    
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        muon_lr=args.muon_lr,
        adamw_lr=args.adamw_lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        rotation_interval=args.rotation_interval,
        freeze_after_step=args.freeze_after_step,
        target_params_m=args.target_params,
        params_tolerance=args.params_tolerance,
        memory_limit_mb=args.memory_limit,
        num_active=args.num_active,
        precision=args.precision,
    )
    
    # Определение пути для продолжения
    # Сначала создаём предварительную конфигурацию для проверки совместимости
    temp_config = ModelConfig(
        vocab_size=args.vocab_size,
        dim=args.dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts or 16,
        expert_hidden=args.expert_hidden,
    )
    
    resume_path = None
    if args.auto_resume:
        resume_path = find_compatible_checkpoint(args.checkpoint_dir, temp_config)
        if resume_path:
            print(f"\n📂 Авто-продолжение: {resume_path}")
        else:
            print(f"\n⚠️ Совместимый чекпоинт не найден, начинаем с нуля")
    elif args.resume_from:
        resume_path = args.resume_from
    
    # === Ветвь 1: skip-search ===
    if args.skip_search:
        print(f"\n[РЕЖИМ] Пропуск поиска, обучение с заданной конфигурацией")
        
        # Если --auto-resume, пытаемся определить конфигурацию из чекпоинта
        detected_config = None
        if args.auto_resume and resume_path:
            detected_config = auto_detect_config_from_checkpoint(resume_path)
        
        # Определяем параметры (из detected_config или аргументов)
        def get_param(name, default):
            if detected_config and name in detected_config:
                return detected_config[name]
            return getattr(args, name, default)
        
        final_dim = get_param('dim', args.dim)
        final_layers = get_param('num_layers', args.num_layers)
        final_heads = get_param('num_heads', args.num_heads)
        final_experts = get_param('num_experts', args.num_experts or 16)
        final_hidden = get_param('expert_hidden', args.expert_hidden)
        final_vocab = get_param('vocab_size', args.vocab_size)
        
        config = ModelConfig(
            vocab_size=final_vocab,
            dim=final_dim,
            num_layers=final_layers,
            num_heads=final_heads,
            kv_heads=max(1, final_heads // 2),
            num_experts=final_experts,
            expert_hidden=final_hidden,
            num_active=args.num_active,
        )
        
        model, losses = train_micro_zaya(
            config, training_config, args.corpus,
            resume_from=resume_path, args=args,
        )
        return
    
    # === Ветвь 2: Optuna-поиск ===
    print(f"\n📋 РАСЧЁТ ПРОСТРАНСТВА КОНФИГУРАЦИЙ")
    print("-" * 70)
    
    total, valid_div, passed = calculate_search_space(args, training_config, args.precision)
    
    print(f"Всего комбинаций: {total:,}")
    print(f"Валидных: {valid_div:,}")
    print(f"Проходящих фильтры: {passed:,}")
    print(f"Целевой размер: {args.target_params}M ±{args.params_tolerance}%")
    print(f"Лимит памяти: {args.memory_limit} МБ")
    
    if passed == 0:
        print(f"\n❌ Нет конфигураций, проходящих фильтры")
        return
    
    # Настройка Optuna
    study_name = f"micro_zaya_{args.target_params:.0f}M_v2"
    db_path = f"{study_name}.db"
    storage = f"sqlite:///{db_path}"
    
    if args.fresh_start and os.path.exists(db_path):
        os.remove(db_path)
        print(f"\n🗑️ Удалена база: {db_path}")
    
    load_if_exists = args.resume or not args.fresh_start
    
    warnings.filterwarnings('ignore', category=optuna.exceptions.ExperimentalWarning)
    
    sampler = optuna.samplers.TPESampler(
        seed=42,
        n_startup_trials=args.startup_trials,
        multivariate=True,
        group=True,
    )
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
        direction="minimize",
        sampler=sampler,
    )
    
    prev_completed = len([t for t in study.trials if t.state == TrialState.COMPLETE])
    prev_passed = len([t for t in study.trials
                       if t.state == TrialState.COMPLETE and t.value != float('inf')])
    
    print(f"\n📊 БАЗА: {db_path}")
    print(f"   Завершено: {prev_completed}")
    print(f"   Прошедших фильтры: {prev_passed}")
    
    print(f"\n🚀 НАЧАЛО ПОИСКА ({args.n_trials} триалов)")
    print("-" * 80)
    
    objective = create_objective(model_config_args, training_config, args.precision)
    
    t_start = time.time()
    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)
    except KeyboardInterrupt:
        print("\n⚠️ Прервано")
    
    elapsed = time.time() - t_start
    
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    passed_trials = [t for t in completed if t.value != float('inf')]
    
    print(f"\n{'='*80}")
    print(f"📈 СТАТИСТИКА ПОИСКА")
    print(f"{'='*80}")
    print(f"   Всего: {len(completed)}")
    print(f"   Прошли: {len(passed_trials)}")
    print(f"   Время: {elapsed:.1f} с")
    
    if passed_trials:
        print(f"\n🏆 ТОП-10:")
        print(f"{'#':<3}{'Парам.':<10}{'Память':<10}{'Время':<10}{'Эксп.':<8}Конфигурация")
        print("-" * 80)
        
        for i, trial in enumerate(sorted(passed_trials, key=lambda t: t.value)[:10]):
            pm = trial.user_attrs.get('total_params_m', 0)
            mm = trial.user_attrs.get('total_memory_mb', 0)
            ms = trial.user_attrs.get('median_time_ms', 0)
            cfg = (f"dim={trial.params['dim']}, layers={trial.params['num_layers']}, "
                   f"experts={trial.params['num_experts']}, hidden={trial.params['expert_hidden']}")
            print(f"{i:<3}{pm:<10.1f}{mm:<10.0f}{ms:<10.1f}"
                  f"{trial.params['num_experts']:<8}{cfg}")
        
        best = min(passed_trials, key=lambda t: t.value)
        print(f"\n🏆 ЛУЧШАЯ:")
        print(f"   Параметры: {best.user_attrs.get('total_params_m', 0):.1f}M")
        print(f"   Память: {best.user_attrs.get('total_memory_mb', 0):.0f} МБ")
        print(f"   Время шага: {best.user_attrs.get('median_time_ms', 0):.1f} мс")
        for k, v in best.params.items():
            print(f"   {k}: {v}")
        
        if args.save_config:
            with open(args.save_config, 'w') as f:
                json.dump({
                    'target_params_m': args.target_params,
                    'params': best.params,
                    'user_attrs': best.user_attrs,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                }, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Сохранено: {args.save_config}")
        
        print(f"\n[2/2] Обучение с лучшей конфигурацией...")
        
        config = ModelConfig(
            vocab_size=args.vocab_size,
            dim=best.params['dim'],
            num_layers=best.params['num_layers'],
            num_heads=best.params['num_heads'],
            kv_heads=max(1, best.params['num_heads'] // 2),
            num_experts=best.params['num_experts'],
            expert_hidden=best.params['expert_hidden'],
            num_active=args.num_active,
        )
        
        train_micro_zaya(
            config, training_config, args.corpus,
            resume_from=resume_path, args=args,
        )
    else:
        print(f"\n❌ Нет подходящих конфигураций")
        print(f"   --params-tolerance {args.params_tolerance * 1.5:.0f}")
        print(f"   --memory-limit {args.memory_limit * 1.2:.0f}")


if __name__ == '__main__':
    main()
