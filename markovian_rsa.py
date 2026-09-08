"""
Markovian RSA (Reasoning Step Attention) для ZAYA1-8B.

Позволяет одновременно обрабатывать несколько цепочек рассуждений
через марковскую передачу состояния между слоями.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkovianRSA(nn.Module):
    """
    Markovian Reasoning Step Attention.
    
    Передаёт состояние роутера между слоями через марковскую цепь,
    позволяя модели поддерживать несколько цепочек рассуждений.
    """
    
    def __init__(self, dim, num_paths=3, max_history=4):
        """
        Args:
            dim: размерность состояния
            num_paths: число параллельных цепочек рассуждений
            max_history: максимальная длина истории состояний
        """
        super().__init__()
        self.dim = dim
        self.num_paths = num_paths
        self.max_history = max_history
        
        # Параметры для каждой цепочки
        self.path_gammas = nn.Parameter(torch.ones(num_paths) * 0.1)
        
        # Внимание к истории состояний
        self.history_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=4,
            batch_first=True,
        )
        
        # Проекция для агрегации цепочек
        self.path_projection = nn.Linear(dim * num_paths, dim, bias=False)
        
        # Нормализация
        self.norm = nn.RMSNorm(dim)
    
    def forward(self, current_state, history_states=None):
        """
        Применяет Markovian RSA к текущему состоянию.
        
        Args:
            current_state: [B*T, dim] текущее состояние роутера
            history_states: список предыдущих состояний (до max_history)
        
        Returns:
            updated_state: [B*T, dim] обновлённое состояние
            new_history: обновлённая история состояний
        """
        if history_states is None:
            history_states = []
        
        # Если истории нет, возвращаем текущее состояние
        if not history_states:
            return current_state, [current_state.detach()]
        
        # Ограничиваем длину истории
        history_states = history_states[-self.max_history:]
        
        # Применяем внимание к истории
        # history_states: список [B*T, dim]
        history_stack = torch.stack(history_states, dim=1)  # [B*T, H, dim]
        current_expanded = current_state.unsqueeze(1)  # [B*T, 1, dim]
        
        # Attention: query=current, keys/values=history
        attended, _ = self.history_attention(
            current_expanded,
            history_stack,
            history_stack,
        )  # [B*T, 1, dim]
        
        attended = attended.squeeze(1)  # [B*T, dim]
        
        # Комбинируем текущее состояние с attended history
        # через обучаемый gamma
        gamma = torch.sigmoid(self.path_gammas[0])  # Используем первый путь
        updated_state = current_state + gamma * attended
        
        # Нормализация
        updated_state = self.norm(updated_state)
        
        # Обновляем историю
        new_history = history_states + [updated_state.detach()]
        if len(new_history) > self.max_history:
            new_history = new_history[-self.max_history:]
        
        return updated_state, new_history


class SimpleMarkovianRSA(nn.Module):
    """
    Упрощённая версия Markovian RSA (как в текущей реализации).
    
    Использует только один путь и фиксированный gamma.
    """
    
    def __init__(self, dim, gamma_init=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma_init))
    
    def forward(self, current_state, prev_state=None):
        """
        Применяет простую марковскую передачу состояния.
        
        Args:
            current_state: [B*T, dim] текущее состояние
            prev_state: [B*T, dim] предыдущее состояние (или None)
        
        Returns:
            updated_state: [B*T, dim]
        """
        if prev_state is None or prev_state.shape != current_state.shape:
            return current_state
        
        # Марковское обновление: current + gamma * prev
        return current_state + self.gamma * prev_state
