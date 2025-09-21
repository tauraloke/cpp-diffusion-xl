# Установка C++ Diffusion XL

## Системные требования

- **ОС**: Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **GPU**: NVIDIA GPU с поддержкой CUDA Compute Capability 6.0+
- **RAM**: Минимум 8GB, рекомендуется 16GB+
- **VRAM**: Минимум 4GB, рекомендуется 8GB+

## Установка зависимостей

### Windows

1. **CUDA Toolkit 11.8+**
   - Скачайте с [NVIDIA Developer](https://developer.nvidia.com/cuda-downloads)
   - Установите CUDA Toolkit
   - Добавьте CUDA в PATH

2. **cuDNN 8.6+**
   - Скачайте с [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Распакуйте в папку CUDA (обычно `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`)

3. **OpenCV 4.5+**
   - Скачайте с [OpenCV Releases](https://opencv.org/releases/)
   - Или используйте vcpkg: `vcpkg install opencv4`

4. **Visual Studio 2019+**
   - Установите Visual Studio Community с C++ поддержкой
   - Или установите Build Tools for Visual Studio

5. **CMake 3.18+**
   - Скачайте с [CMake Downloads](https://cmake.org/download/)
   - Добавьте CMake в PATH

### Ubuntu/Debian

```bash
# Обновить пакеты
sudo apt update

# Установить CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Установить cuDNN
sudo apt-get install libcudnn8 libcudnn8-dev

# Установить OpenCV
sudo apt-get install libopencv-dev

# Установить CMake и компилятор
sudo apt-get install cmake build-essential

# Добавить CUDA в PATH
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### macOS

```bash
# Установить Homebrew (если не установлен)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Установить зависимости
brew install cmake opencv

# Установить CUDA (если доступно)
# Скачайте с NVIDIA Developer сайта
```

## Сборка проекта

### Windows

```cmd
# Клонировать репозиторий
git clone <repository-url>
cd cpp-diffusion-xl

# Собрать проект
build.bat

# Или вручную
mkdir build
cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
```

### Linux/macOS

```bash
# Клонировать репозиторий
git clone <repository-url>
cd cpp-diffusion-xl

# Сделать скрипт исполняемым
chmod +x build.sh

# Собрать проект
./build.sh

# Или вручную
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Проверка установки

```bash
# Проверить CUDA
nvcc --version

# Проверить cuDNN
cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2

# Проверить OpenCV
pkg-config --modversion opencv4
```

## Настройка модели SDXL

1. Скачайте модель SDXL с [Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
2. Распакуйте в папку с моделями
3. Структура должна быть:
   ```
   sdxl_model/
   ├── unet/
   ├── vae/
   ├── text_encoder/
   └── scheduler/
   ```

## Использование

```bash
# Базовое использование
./build/cpp_diffusion_xl --model /path/to/sdxl/model --prompt "beautiful landscape"

# С дополнительными параметрами
./build/cpp_diffusion_xl \
  --model /path/to/sdxl/model \
  --prompt "a detailed portrait of a woman" \
  --negative "blurry, low quality" \
  --width 1024 \
  --height 1024 \
  --steps 20 \
  --cfg 7.0 \
  --sampler euler_a \
  --scheduler sgm_uniform \
  --seed 42
```

## Устранение неполадок

### Ошибка "CUDA not found"
- Убедитесь, что CUDA установлена и добавлена в PATH
- Проверьте переменную окружения `CUDA_PATH`

### Ошибка "cuDNN not found"
- Убедитесь, что cuDNN установлен в папку CUDA
- Проверьте переменную окружения `CUDNN_PATH`

### Ошибка "OpenCV not found"
- Убедитесь, что OpenCV установлен
- На Linux: `sudo apt-get install libopencv-dev`
- На Windows: используйте vcpkg или установите вручную

### Ошибка компиляции
- Убедитесь, что используется C++17 компилятор
- Проверьте версию CMake (должна быть 3.18+)
- Убедитесь, что все зависимости установлены

### Ошибка "Out of memory"
- Уменьшите размер изображения (--width, --height)
- Уменьшите количество шагов (--steps)
- Закройте другие приложения, использующие GPU

## Производительность

Для максимальной производительности:

1. Используйте GPU с большим количеством VRAM
2. Включите mixed precision (автоматически)
3. Используйте быстрые сэмплеры (euler_a, dpm_fast)
4. Оптимизируйте размер изображения под вашу VRAM

## Поддержка

При возникновении проблем:

1. Проверьте системные требования
2. Убедитесь, что все зависимости установлены
3. Проверьте логи сборки и выполнения
4. Создайте issue в репозитории с подробным описанием проблемы
