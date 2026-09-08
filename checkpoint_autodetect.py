"""
Автоопределение архитектуры модели из safetensors чекпоинта.

Извлекает все необходимые параметры из размеров тензоров:
- vocab_size, dim из embedding.weight
- num_layers из blocks.*
- num_experts, expert_hidden из moe.experts.*
- num_heads, kv_heads из attn.*
"""
import os
import re
import json
import glob
from typing import Optional, Dict, Any


def detect_config_from_checkpoint(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Определяет конфигурацию модели из чекпоинта.
    
    Args:
        ckpt_path: путь к safetensors файлу
        
    Returns:
        Словарь с параметрами конфигурации или None
    """
    try:
        # Пытаемся использовать safetensors
        try:
            from safetensors import safe_open
        except ImportError:
            print("⚠️ safetensors не установлен, пробуем альтернативу")
            return _detect_from_filename(ckpt_path)
        
        config = {}
        tensor_shapes = {}
        
        # Читаем только метаданные (быстро, без загрузки тензоров)
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                shape = f.get_slice(key).get_shape()
                tensor_shapes[key] = shape
        
        # === ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ ===
        
        # vocab_size и dim из embedding
        if 'embedding.weight' in tensor_shapes:
            vocab_size, dim = tensor_shapes['embedding.weight']
            config['vocab_size'] = vocab_size
            config['dim'] = dim
        
        # num_layers из blocks
        block_indices = set()
        for key in tensor_shapes:
            match = re.match(r'blocks\.(\d+)\.', key)
            if match:
                block_indices.add(int(match.group(1)))
        
        if block_indices:
            config['num_layers'] = max(block_indices) + 1
        
        # num_experts и expert_hidden из moe.experts
        expert_indices = set()
        for key in tensor_shapes:
            match = re.match(r'blocks\.\d+\.moe\.experts\.(\d+)\.up\.weight', key)
            if match:
                expert_indices.add(int(match.group(1)))
                # expert_hidden из формы [expert_hidden, dim]
                expert_hidden, _ = tensor_shapes[key]
                config['expert_hidden'] = expert_hidden
        
        if expert_indices:
            config['num_experts'] = max(expert_indices) + 1
        
        # num_heads и kv_heads из attn
        for key in tensor_shapes:
            # q_norm.weight [head_dim] → head_dim
            match = re.match(r'blocks\.\d+\.attn\.q_norm\.weight', key)
            if match:
                head_dim = tensor_shapes[key][0]
                if 'dim' in config:
                    config['num_heads'] = config['dim'] // head_dim
            
            # k_proj.weight [kv_dim, dim] → kv_heads
            match = re.match(r'blocks\.\d+\.attn\.k_proj\.weight', key)
            if match:
                kv_dim = tensor_shapes[key][0]
                if 'dim' in config and 'num_heads' in config:
                    head_dim = config['dim'] // config['num_heads']
                    config['kv_heads'] = kv_dim // head_dim
        
        # Микроэксперты
        micro_expert_indices = set()
        for key in tensor_shapes:
            match = re.match(r'micro_expert_pool\.experts\.(\d+)\.', key)
            if match:
                micro_expert_indices.add(int(match.group(1)))
                # micro_hidden из up_weight_f32 [micro_hidden, micro_dim]
                if 'up_weight_f32' in key:
                    micro_hidden, micro_dim = tensor_shapes[key]
                    config['micro_hidden'] = micro_hidden
                    config['micro_dim'] = micro_dim
        
        if micro_expert_indices:
            config['num_micro_experts'] = max(micro_expert_indices) + 1
        
        # Сохраняем полную информацию для отладки
        config['_tensor_count'] = len(tensor_shapes)
        config['_ckpt_path'] = ckpt_path
        
        return config
        
    except Exception as e:
        print(f"❌ Ошибка детекции из {ckpt_path}: {e}")
        return None


def _detect_from_filename(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Fallback: попытка извлечь шаг из имени файла."""
    match = re.search(r'step(\d+)', os.path.basename(ckpt_path))
    if match:
        return {'_step': int(match.group(1))}
    return None


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Находит последний чекпоинт в директории."""
    patterns = [
        os.path.join(checkpoint_dir, '**', '*.safetensors'),
        os.path.join(checkpoint_dir, '*.safetensors'),
        os.path.join(checkpoint_dir, '**', 'step_*.pt'),
        os.path.join(checkpoint_dir, '*.pt'),
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(pattern, recursive=True))
    
    if not all_files:
        return None
    
    # Сортируем по номеру шага
    def get_step(path):
        match = re.search(r'step[_]?(\d+)', os.path.basename(path))
        return int(match.group(1)) if match else 0
    
    all_files.sort(key=get_step, reverse=True)
    return all_files[0]


def save_config_to_checkpoint(ckpt_path: str, config: Dict[str, Any]):
    """Сохраняет конфигурацию рядом с чекпоинтом."""
    config_path = ckpt_path.replace('.safetensors', '_config.json')
    config_path = config_path.replace('.pt', '_config.json')
    
    # Убираем служебные поля
    clean_config = {k: v for k, v in config.items() if not k.startswith('_')}
    
    with open(config_path, 'w') as f:
        json.dump(clean_config, f, indent=2)
    
    return config_path


def load_config_from_checkpoint(ckpt_path: str) -> Optional[Dict[str, Any]]:
    """Загружает сохранённую конфигурацию, если есть."""
    config_path = ckpt_path.replace('.safetensors', '_config.json')
    config_path = config_path.replace('.pt', '_config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    
    return None


if __name__ == '__main__':
    # Тест
    import sys
    
    if len(sys.argv) < 2:
        print("Использование: python checkpoint_autodetect.py <путь_к_чекпоинту>")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    
    if not os.path.exists(ckpt_path):
        # Возможно, это директория
        if os.path.isdir(ckpt_path):
            ckpt_path = find_latest_checkpoint(ckpt_path)
            if not ckpt_path:
                print(f"❌ Чекпоинты не найдены в {sys.argv[1]}")
                sys.exit(1)
            print(f"📁 Найден последний чекпоинт: {ckpt_path}")
        else:
            print(f"❌ Файл не найден: {ckpt_path}")
            sys.exit(1)
    
    config = detect_config_from_checkpoint(ckpt_path)
    
    if config:
        print("\n" + "="*70)
        print("🔍 ОПРЕДЕЛЁННАЯ КОНФИГУРАЦИЯ")
        print("="*70)
        
        for key, value in sorted(config.items()):
            if not key.startswith('_'):
                print(f"   {key}: {value}")
        
        print(f"\n📊 Всего тензоров: {config.get('_tensor_count', 0)}")
        
        # Сохраняем конфигурацию
        saved_path = save_config_to_checkpoint(ckpt_path, config)
        print(f"\n💾 Конфигурация сохранена: {saved_path}")
        
        print("="*70)
    else:
        print("❌ Не удалось определить конфигурацию")
