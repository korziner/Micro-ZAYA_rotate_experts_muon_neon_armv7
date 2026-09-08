# Micro-ZAYA_rotate_experts_muon_neon_armv7
The best GPT $100-phone can pretrain from 0

```
clang version 21.1.8
Target: armv7a-unknown-linux-android24   

cargo 1.98.0 (797e8a9bc 2026-08-05) (built from a source tarball)
~/deb/nocuda $ python -V     
Python 3.14.6

pip freeze|rg 'muon|torch|rotat|former|tok|nump|blas'
expert_rotator @ file:///data/data/com.termux/files/home/deb/nocuda/expert_rotator
muon==0.1.9
muon_neon @ file:///data/data/com.termux/files/home/deb/nocuda/muon_neon
muon_neon_optimized @ file:///data/data/com.termux/files/home/deb/nocuda/muon_neon_optimized
numpy==2.4.6
tokenizers @ file:///data/data/com.termux/files/home/deb/tokenizers/bindings/python/target/wheels/tokenizers-0.23.2.dev0-cp310-abi3-android_24_armeabi_v7a.whl#sha256=5297919565604591022dc001efd773099dc421ce70e0383a61bb017c2a8d227c
torch @ file:///home/builder/.termux-build/python-torch/src
torchvision @ file:///home/builder/.termux-build/python-torchvision/src
transformers==5.16.1

dpkg -l|rg 'muon|torch|rotat|former|tok|nump|blas'
ii  blas-openblas                    0.3.34                          arm          OpenBLAS symlinks for BLAS/CBLAS/LAPACK/LAPACKE
ii  clblast                          1.7.0                           arm          Tuned OpenCL BLAS
 
ii  libopenblas                      0.3.34                          arm          An optimized BLAS library based on GotoBLAS2 1.13 BSD
ii  libopenblas-static               0.3.34                          arm          Static libraries for libopenblas
ii  python-numpy                     2.4.4-1                         arm          The fundamental package for scientific computing with Python
ii  python-numpy-static              2.4.4-1                         arm          Static libraries for python-numpy
ii  python-torch                     2.11.0-2                        arm          Tensors and Dynamic neural networks in Python
```

```
~/deb/nocuda $ python Micro-ZAYA.rotate_experts_rust.py     --skip-search     --max-steps 1000     --rotation-interval 300     --inference-interval 10     --save-every 5     --max-checkpoints 3     --checkpoint-dir checkpoints/39     --corpus corpus.tok16     --precision fp32     --auto-resume

✅ Микроэксперты доступны
[Muon] ✅ Режим data_ptr доступен (быстрый путь через ctypes)

🔍 Найден чекпоинт: checkpoints/39/micro_zaya_step195.safetensors
   📊 Автоопределённая конфигурация:
      dim: 512
      expert_hidden: 256
      kv_heads: 4
      micro_dim: 32
      micro_hidden: 32
      num_experts: 12
      num_heads: 8
      num_layers: 10
      num_micro_experts: 8
      vocab_size: 16384
   ✅ args.dim = 512
   ✅ args.num_layers = 10
   ✅ args.num_heads = 8
   ✅ args.num_experts = 12
   ✅ args.expert_hidden = 256
   ✅ Параметры обновлены из чекпоинта
================================================================================
🎯 MICRO-ZAYA: ПОИСК И ОБУЧЕНИЕ
================================================================================

📌 Задано экспертов: 12 (диапазон [4, 20])

✅ Найден совместимый чекпоинт: шаг 195

📂 Авто-продолжение: checkpoints/39/micro_zaya_step195.safetensors

[РЕЖИМ] Пропуск поиска, обучение с заданной конфигурацией
================================================================================
🚀 ОБУЧЕНИЕ MICRO-ZAYA
================================================================================

📊 ОЦЕНКА ПАМЯТИ:
   Точность: FP32 (4 байт/параметр)
   Всего параметров: 51.7M
   Веса: 197.3 МБ
   Градиенты: 197.3 МБ
   AdamW: 64.0 МБ
   Активации: 20.0 МБ
   ИТОГО: 478.6 МБ
   🛡️ Микроэксперты: 1.08 МБ (стабилизаторы)

Модель создана: 52.0M параметров
   ✅ Модель в FP32
   ✅ grad_dtype=None установлен для 458 параметров
Оптимизатор: Muon (457), AdamW (1)

📂 Продолжение из: checkpoints/39/micro_zaya_step195.safetensors
   ✅ Загружен чекпоинт: micro_zaya_step195.safetensors
   Продолжаем с шага: 195
   ✅ Weight tying восстановлен (lm_head ↔ embedding)
   ✅ Состояние заморозки восстановлено
Корпус: 239,580,724 токенов
✅ Семплер инференса активен
✅ Трекер ротации активен
   📊 Начальный % активных экспертов: 33%
   📥 Предзагрузка первого батча из корпуса...
   ✅ Корпус прогрет за 0.01с
   🎯 Планировщик точности:
      Фаза стабилизации: 200 шагов (BF16)
      Фаза обучения: после конвертации в FP16
      Порог лосса: 9.0
   ✅ Планировщик: установлен шаг 195

🏃 Начало обучения (1000 шагов)...
Обучение:   0%|                                                                                                           | 0/805 [01:20<?]
💾 Чекпоинт: checkpoints/39/micro_zaya_step195.safetensors (230.5 МБ)
Обучение:   1%|▌                                                                                                   | 5/805 [06:47<17:42:56]
🟡 Режим: micro_only, стабильность: 0.12
                                                                                                                                           
   Шаг   200 | Loss: 4.4249 | 77751.6 мс
                                                                                                                                           
   📝 Инференс на шаге 200
Обучение:   1%|▌                                                                                                   | 5/805 [08:05<17:42:56]
   📝 ИНФЕРЕНС НА ШАГЕ 200
   ------------------------------------------------------------------
   [t=0.7] божественный действие кай испорченность сне растерянность трое духоборческий бог
   [t=1.0] зависимость неупорядоченность бесстрастный необходимый молитва высокопоставленны
   [t=1.3] насильственно заключаться в то Иисус Христос он самонадеянность свидетельствоват

   💾 Чекпоинт: checkpoints/39/micro_zaya_step200.safetensors (230.5 МБ)
Обучение:   1%|█▏                                                                                                 | 10/805 [23:48<26:39:47]
⚠️ Мало места: 204 МБ свободно, нужно ~230 МБ
   🗑️ Удалено 5 старых чекпоинтов, освобождено ~1152 МБ
   Теперь свободно: 1357 МБ
   💾 Чекпоинт: checkpoints/39/micro_zaya_step205.safetensors (230.5 МБ)
                                                                                                                                           
   📝 Инференс на шаге 210
Обучение:   2%|█▊                                                                                                 | 15/805 [30:49<19:36:33]
   📝 ИНФЕРЕНС НА ШАГЕ 210
   ------------------------------------------------------------------
   [t=0.7] вдохновитель православный восточный наставление самоуверенный  бесплодность сос
   [t=1.0] разнообразно пятдесятница естественный созерцание православный богословский обяз
   [t=1.3] александрийский школа богодухновенный писание постепенный непонятный расстрелива

   💾 Чекпоинт: checkpoints/39/micro_zaya_step210.safetensors (230.5 МБ)
Обучение:   2%|██▍                                                                                                | 20/805 [46:45<26:44:34]
💾 Чекпоинт: checkpoints/39/micro_zaya_step215.safetensors (230.5 МБ)
                                                                                                                                           
   📝 Инференс на шаге 220
Обучение:   3%|███                                                                                                | 25/805 [53:41<19:21:18]
   📝 ИНФЕРЕНС НА ШАГЕ 220
   ------------------------------------------------------------------
   [t=0.7] положительно переставлять срать продемонстрировать состояние возможность другой 
   [t=1.0] монументальный вслушиваться раскапывать ладан южнорусский штундист видеть зачина
   [t=1.3] невозможно представлять плотоугодие церковный приходский школа министерство наш 

   💾 Чекпоинт: checkpoints/39/micro_zaya_step220.safetensors (230.5 МБ)
Обучение:   4%|███▌                                                                                             | 30/805 [1:08:30<27:03:57]
🟡 Режим: micro_only, стабильность: 0.62
Обучение:   4%|███▌                                                                                             | 30/805 [1:09:55<27:03:57]
💾 Чекпоинт: checkpoints/39/micro_zaya_step225.safetensors (230.5 МБ)
                                                                                                                                           
   📝 Инференс на шаге 230
Обучение:   4%|████▏                                                                                            | 35/805 [1:17:03<19:39:07]
   📝 ИНФЕРЕНС НА ШАГЕ 230
   ------------------------------------------------------------------
   [t=0.7] культоцентрический самоуслаждение быть широко распространять Ѳ макарий египетски
   [t=1.0] рукопись синайский библиотека приближение неокончательность оборачиваться коммун
   [t=1.3] благовонный необходимо чтобы существовать инок представитель центральный градона


⚠️ Мало места: 192 МБ свободно, нужно ~230 МБ
   🗑️ Удалено 5 старых чекпоинтов, освобождено ~1152 МБ
   Теперь свободно: 1345 МБ
   💾 Чекпоинт: checkpoints/39/micro_zaya_step230.safetensors (230.5 МБ)
Обучение:   5%|████▊                                                                                            | 40/805 [1:33:09<28:04

```

```
📊 Анализ метафайлов чекпоинтов:
Чекпоинт                                      Лосс (последний) Статус
--------------------------------------------------------------------------------
checkpoints/39/micro_zaya_step165.safetensors 4.3809          ✅ Хороший
checkpoints/39/micro_zaya_step170.safetensors 4.3117          ✅ Хороший
checkpoints/39/micro_zaya_step175.safetensors 4.2800          ✅ Хороший
checkpoints/39/micro_zaya_step180.safetensors 4.4369          ✅ Хороший
checkpoints/39/micro_zaya_step185.safetensors 4.3534          ✅ Хороший
checkpoints/39/micro_zaya_step190.safetensors 4.3392          ✅ Хороший
checkpoints/39/micro_zaya_step195.safetensors 4.3567          ✅ Хороший
```
