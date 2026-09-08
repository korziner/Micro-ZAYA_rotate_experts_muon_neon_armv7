"""
Модуль периодического инференса с разными температурами и русскими затравками.
Использует словарь vocab16k.txt для токенизации/детокенизации.
"""
import torch
import numpy as np
import os


# Русские затравки для тестирования генерации
RUSSIAN_SEEDS = [
    "Привет",
    "Жил-был",
    "Однажды",
    "В далёком",
    "Было это",
    "Сказал мудрец",
    "Солнце взошло",
]

# Температуры для тестирования
TEMPERATURES = [0.7, 1.0, 1.3]


class InferenceSampler:
    """Периодический инференс с разными параметрами."""
    
    def __init__(self, model, vocab_path="vocab16k.txt", 
                 seed="Привет", max_gen_tokens=32):
        self.model = model
        self.vocab_path = vocab_path
        self.seed = seed
        self.max_gen_tokens = max_gen_tokens
        
        # Загружаем словарь
        self.vocab = self._load_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for i, token in enumerate(self.vocab)}
    
    def _load_vocab(self):
        """Загружает словарь из файла."""
        if not os.path.exists(self.vocab_path):
            print(f"   ⚠️ Словарь {self.vocab_path} не найден, "
                  f"используются токены по индексам")
            return [str(i) for i in range(16384)]
        
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            vocab = [line.rstrip('\n') for line in f]
        
        return vocab
    
    def tokenize(self, text):
        """Токенизирует текст в список индексов."""
        tokens = []
        # Простая токенизация: разбиваем по пробелам и ищем в словаре
        words = text.split()
        for word in words:
            if word in self.token_to_id:
                tokens.append(self.token_to_id[word])
            else:
                # Пробуем найти похожий токен
                # Если не нашли, используем токен <UNK> или первый
                tokens.append(0)
        return tokens
    
    def detokenize(self, token_ids):
        """Конвертирует список индексов в текст."""
        tokens = []
        for idx in token_ids:
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                tokens.append(f"<{idx}>")
        return ' '.join(tokens)
    
    @torch.no_grad()
    @torch.no_grad()
    def generate(self, seed_text="", max_length=100, temperature=1.0, top_k=50, top_p=0.9):
        """Генерирует текст с защитой от экстремальных значений."""
        import torch
        import torch.nn.functional as F
        
        # Сохраняем режим модели
        was_training = self.model.training
        self.model.eval()
        
        # Токенизируем seed
        if seed_text:
            context = self.tokenize(seed_text)
        else:
            # Начинаем с случайного токена
            context = [torch.randint(0, self.model.config.vocab_size, (1,)).item()]
        
        generated = []
        current_tokens = context.copy()
        
        device = next(self.model.parameters()).device
        
        for _ in range(max_length):
            # Ограничиваем контекст
            context = current_tokens[-512:]  # максимальный контекст 512
            
            # Создаём входной тензор
            input_ids = torch.tensor([context], dtype=torch.long, device=device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Берём логиты последнего токена
            next_logits = logits[0, -1, :]  # [vocab_size]
            
            # Применяем температуру
            if temperature != 1.0:
                next_logits = next_logits / temperature
            
            # === ЗАЩИТА ОТ ПЕРЕПОЛНЕНИЯ ЛОГИТОВ ===
            next_logits = torch.clamp(next_logits, min=-50.0, max=50.0)
            
            # === ЗАЩИТА ОТ NaN/Inf ===
            if torch.isnan(next_logits).any() or torch.isinf(next_logits).any():
                next_logits = torch.nan_to_num(next_logits, nan=0.0, posinf=10.0, neginf=-10.0)
            
            # Softmax и выборка
            probs = torch.softmax(next_logits, dim=-1)
            
            # === ЗАЩИТА ОТ ЭКСТРЕМАЛЬНЫХ ВЕРОЯТНОСТЕЙ ===
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Удаляем отрицательные значения
            probs = torch.clamp(probs, min=0.0)
            
            # Если все вероятности нулевые, используем равномерное распределение
            if probs.sum() < 1e-10:
                probs = torch.ones_like(probs) / probs.numel()
            else:
                # Нормализуем
                probs = probs / probs.sum()
            
            # Выборка токена
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            current_tokens.append(next_token)
            generated.append(next_token)
            
            # Останавливаемся на специальном токене (если есть)
            if next_token >= self.model.config.vocab_size:
                break
        
        # Возвращаем модель в исходный режим
        if was_training:
            self.model.train()
        
        # Детокенизируем
        generated_text = self.detokenize(generated)
        
        return {
            'text': generated_text,
            'tokens': generated,
        }

    def sample_all(self, step, verbose=True):
        """Прогоняет инференс со всеми температурами и затравками."""
        results = []
        
        if verbose:
            print(f"\n   📝 ИНФЕРЕНС НА ШАГЕ {step}")
            print("   " + "-"*66)
        
        # Используем первую затравку для краткости вывода
        seed_text = self.seed
        
        for temp in TEMPERATURES:
            result = self.generate(seed_text=seed_text, temperature=temp)
            results.append(result)
            
            if verbose:
                print(f"   [t={temp:.1f}] {result.get('generated', result.get('text', '[нет результата]'))[:80]}")
        
        if verbose:
            print()
        
        return {
            'step': step,
            'results': results
        }
    
    def sample_full(self, step, verbose=True):
        """Полный прогон со всеми затравками и температурами."""
        all_results = []
        
        if verbose:
            print(f"\n   📝 ПОЛНЫЙ ИНФЕРЕНС НА ШАГЕ {step}")
            print("   " + "="*66)
        
        for seed_text in RUSSIAN_SEEDS:
            for temp in TEMPERATURES:
                result = self.generate(seed_text=seed_text, temperature=temp)
                result['seed_used'] = seed_text
                all_results.append(result)
                
                if verbose:
                    print(f"   [{seed_text[:10]}] t={temp:.1f}: "
                          f"{result.get('generated', result.get('text', '[нет результата]'))[:60]}")
        
        if verbose:
            print()
        
        return {
            'step': step,
            'results': all_results
        }
