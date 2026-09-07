"""
Модуль отслеживания ротации экспертов.
Работает с любой структурой MoE-слоя через duck typing.
"""
import json
import time
import numpy as np


class RotationTracker:
    """Отслеживает ротацию экспертов и их эффективность."""
    
    def __init__(self, model, log_file="rotation_history.json"):
        self.model = model
        self.log_file = log_file
        self.history = []
        self.baseline_loss = None
        self._baseline_set_at = 0
    
    def _get_moe_layers(self):
        """Находит все MoE-слои в модели (через обход блоков)."""
        moe_layers = []
        # Пробуем разные варианты доступа к блокам
        blocks = getattr(self.model, 'blocks', None)
        if blocks is None:
            blocks = getattr(self.model, 'layers', None)
        if blocks is None:
            return moe_layers
        
        for idx, block in enumerate(blocks):
            moe = getattr(block, 'moe', None)
            if moe is not None:
                moe_layers.append((idx, moe))
        
        return moe_layers
    
    def _get_expert_state(self, moe):
        """Определяет активные и замороженные эксперты через requires_grad."""
        active = []
        frozen = []
        
        experts = getattr(moe, 'experts', [])
        for i, expert in enumerate(experts):
            # Проверяем, обучается ли эксперт
            params = list(expert.parameters())
            if params and params[0].requires_grad:
                active.append(i)
            else:
                frozen.append(i)
        
        return active, frozen
    
    def _count_frozen_params(self):
        """Подсчитывает параметры замороженных экспертов."""
        total = 0
        for layer_idx, moe in self._get_moe_layers():
            active, frozen = self._get_expert_state(moe)
            experts = getattr(moe, 'experts', [])
            for idx in frozen:
                if idx < len(experts):
                    total += sum(p.numel() for p in experts[idx].parameters())
        return total
    
    def log_state(self, step, verbose=True, loss=None):
        """Логирует текущее состояние экспертов."""
        state = {
            'step': step,
            'layers': [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        total_active = 0
        total_frozen = 0
        
        for layer_idx, moe in self._get_moe_layers():
            active, frozen = self._get_expert_state(moe)
            
            layer_info = {
                'layer': layer_idx,
                'active': active,
                'frozen': frozen,
                'num_active': len(active),
                'num_frozen': len(frozen)
            }
            state['layers'].append(layer_info)
            
            total_active += len(active)
            total_frozen += len(frozen)
            
            if verbose:
                print(f"      Слой {layer_idx}: активны {active}, "
                      f"заморожены {len(frozen)}")
        
        state['total_active'] = total_active
        state['total_frozen'] = total_frozen
        
        # Экономия памяти (8 байт/параметр для состояния оптимизатора)
        frozen_params = self._count_frozen_params()
        memory_saved_mb = frozen_params * 8 / 1024 / 1024
        state['frozen_params'] = frozen_params
        state['memory_saved_mb'] = memory_saved_mb
        
        if loss is not None:
            state['loss'] = loss
        
        if verbose:
            print(f"      Итого: {total_active} активных, {total_frozen} замороженных")
            print(f"      Экономия: {memory_saved_mb:.1f} МБ")
        
        return state
    
    def set_baseline(self, loss, step):
        """Устанавливает базовый лосс для сравнения."""
        if self.baseline_loss is None:
            self.baseline_loss = loss
            self._baseline_set_at = step
            return True
        return False
    
    def measure_effectiveness(self, current_loss):
        """Измеряет эффективность относительно baseline."""
        if self.baseline_loss is None:
            return None
        
        improvement = self.baseline_loss - current_loss
        return {
            'baseline_loss': self.baseline_loss,
            'current_loss': current_loss,
            'improvement': improvement,
            'improvement_percent': (improvement / self.baseline_loss) * 100
        }
    
    def record_rotation(self, step, loss, verbose=True):
        """Записывает событие ротации."""
        state = self.log_state(step, verbose=verbose, loss=loss)
        
        event = {
            'step': step,
            'state': state,
            'loss': loss
        }
        
        if self.baseline_loss is not None:
            event['effectiveness'] = self.measure_effectiveness(loss)
        
        self.history.append(event)
        return event
    
    def save_history(self):
        """Сохраняет историю в JSON."""
        data = {
            'history': self.history,
            'baseline_loss': self.baseline_loss,
            'baseline_set_at': self._baseline_set_at,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        try:
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"   ⚠️ Ошибка сохранения истории: {e}")
            return False
    
    def print_summary(self):
        """Печатает итоговую сводку."""
        print(f"\n📊 ИТОГОВАЯ СВОДКА О РОТАЦИИ")
        print("="*70)
        print(f"   Всего событий ротации: {len(self.history)}")
        
        if self.baseline_loss is not None:
            print(f"   Baseline loss: {self.baseline_loss:.4f}")
        
        if self.history:
            last = self.history[-1]
            state = last['state']
            print(f"\n   Финальное состояние:")
            print(f"     Активных экспертов: {state['total_active']}")
            print(f"     Замороженных экспертов: {state['total_frozen']}")
            print(f"     Экономия памяти: {state['memory_saved_mb']:.1f} МБ")
        
        # Анализ эффективности
        effective_rotations = [
            e for e in self.history 
            if 'effectiveness' in e and e['effectiveness']['improvement'] > 0
        ]
        
        if effective_rotations:
            print(f"\n   Успешных ротаций: {len(effective_rotations)}/{len(self.history)}")
            avg_improvement = np.mean([
                e['effectiveness']['improvement_percent'] 
                for e in effective_rotations
            ])
            print(f"   Среднее улучшение: {avg_improvement:+.1f}%")
