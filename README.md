# C++ Diffusion XL

Высокопроизводительная консольная утилита для генерации изображений SDXL на C++ с оптимизациями из ComfyUI.

## Особенности

- **Оптимизированное управление памятью**: Умная загрузка моделей в VRAM и RAM
- **cuDNN ускорение**: Использование ядер NVIDIA cuDNN для максимальной производительности
- **Поддержка SDXL**: Полная поддержка архитектуры Stable Diffusion XL
- **Euler A сэмплер**: Реализация Euler Ancestral сэмплера
- **SGM Uniform планировщик**: Оптимизированный планировщик шума
- **Консольный интерфейс**: Простой в использовании CLI

## Требования

- CUDA 11.8+ с cuDNN 8.6+
- OpenCV 4.5+
- CMake 3.18+
- C++17 компилятор (GCC 9+, Clang 10+, MSVC 2019+)
- NVIDIA GPU с поддержкой CUDA

## Сборка

### Linux/macOS

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

### Windows

```cmd
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

## Использование

```bash
./cpp_diffusion_xl --model /path/to/sdxl/model --prompt "beautiful landscape" --width 1024 --height 1024 --steps 20 --cfg 7.0 --sampler euler_a --scheduler sgm_uniform
```

### Параметры

- `--model <path>` - Путь к модели SDXL (обязательно)
- `--prompt <text>` - Положительный промпт (обязательно)
- `--negative <text>` - Негативный промпт (опционально)
- `--width <int>` - Ширина изображения (по умолчанию: 1024)
- `--height <int>` - Высота изображения (по умолчанию: 1024)
- `--steps <int>` - Количество шагов сэмплинга (по умолчанию: 20)
- `--cfg <float>` - CFG scale (по умолчанию: 7.0)
- `--sampler <name>` - Тип сэмплера (по умолчанию: euler_a)
- `--scheduler <name>` - Тип планировщика (по умолчанию: sgm_uniform)
- `--seed <int>` - Случайное зерно (по умолчанию: 0)
- `--output <path>` - Папка для сохранения (по умолчанию: ./results)

### Поддерживаемые сэмплеры

- `euler_a` - Euler Ancestral (рекомендуется)
- `euler` - Euler
- `heun` - Heun
- `dpm_2` - DPM-2
- `dpm_2_ancestral` - DPM-2 Ancestral
- `lms` - LMS
- `dpm_fast` - DPM Fast
- `dpm_adaptive` - DPM Adaptive
- `dpmpp_2s_ancestral` - DPM++ 2S Ancestral
- `dpmpp_sde` - DPM++ SDE
- `dpmpp_2m` - DPM++ 2M
- `dpmpp_2m_sde` - DPM++ 2M SDE
- `ddpm` - DDPM
- `lcm` - LCM

### Поддерживаемые планировщики

- `sgm_uniform` - SGM Uniform (рекомендуется)
- `normal` - Normal
- `karras` - Karras
- `exponential` - Exponential
- `simple` - Simple
- `ddim_uniform` - DDIM Uniform
- `beta` - Beta
- `linear_quadratic` - Linear Quadratic
- `kl_optimal` - KL Optimal

## Архитектура

Проект использует оптимизации из ComfyUI:

### Управление памятью
- Умная загрузка моделей в VRAM/RAM
- Автоматическое определение доступной памяти
- Оптимизация для различных конфигураций GPU

### cuDNN интеграция
- Использование cuDNN для операций свертки
- Оптимизированные ядра для attention механизмов
- Поддержка mixed precision (FP16/BF16)

### Сэмплеры
- Реализация Euler A на основе ComfyUI
- Поддержка ancestral sampling
- Оптимизированные математические операции

### Планировщики
- SGM Uniform планировщик из ComfyUI
- Поддержка различных стратегий шума
- Оптимизированные вычисления

## Структура проекта

```
cpp-diffusion-xl/
├── include/           # Заголовочные файлы
│   ├── sdxl_model.h
│   ├── memory_manager.h
│   ├── sampler.h
│   ├── scheduler.h
│   ├── clip_encoder.h
│   ├── vae_decoder.h
│   ├── unet_model.h
│   └── image_utils.h
├── src/              # Исходный код
│   ├── main.cpp
│   ├── sdxl_model.cpp
│   ├── memory_manager.cpp
│   ├── sampler.cpp
│   ├── scheduler.cpp
│   ├── clip_encoder.cpp
│   ├── vae_decoder.cpp
│   ├── unet_model.cpp
│   └── image_utils.cpp
├── CMakeLists.txt    # Конфигурация сборки
└── README.md         # Документация
```

## Производительность

Утилита оптимизирована для максимальной производительности:

- Использование cuDNN для ускорения операций
- Умное управление памятью для больших моделей
- Оптимизированные алгоритмы сэмплинга
- Поддержка mixed precision вычислений

## Лицензия

MIT License
