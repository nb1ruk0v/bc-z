# BC-Z Implementation

Обучающий проект для улучшения навыков Claude Code и реализации BC-Z (Behavioral Cloning with Zero-shot task generalization) для робототехники.

**Статус:** Week 2 Complete ✅ - Data Pipeline + Model (ResNet + FiLM) реализованы (55/55 тестов)

## О проекте

BC-Z - это система имитационного обучения для робототехники с zero-shot генерализацией на новые задачи. Проект основан на исследовательской работе Google Research по multi-task imitation learning.

**Ключевые особенности:**
- Language/video-conditioned policies с предобученными эмбеддингами
- ResNet18/50 + FiLM архитектура для task-conditioned управления
- Предсказание multi-waypoint траекторий (10 waypoints)
- Представление ориентации через axis-angle
- Residual actions относительно текущего состояния

## Структура проекта

```
bc-z/
├── src/                          # Основной код реализации
│   ├── data/
│   │   └── dataset.py           # ✅ BCZDataset для TFRecord данных
│   ├── models/                   # ✅ ResNet + FiLM + BCZPolicy
│   │   ├── film.py              # ✅ FiLMLayer (task conditioning)
│   │   ├── backbone.py          # ✅ ResNetBackbone (resnet18/50)
│   │   └── policy.py            # ✅ BCZPolicy
│   ├── training/                 # 🔜 Training loop и trainer
│   └── utils/
│       └── metrics.py           # ✅ Метрики для оценки (13 тестов)
├── tests/
│   ├── fixtures/
│   │   ├── test_data.tfrecord   # Тестовые данные (5 сэмплов, 154 KB)
│   │   └── create_test_data.py  # Скрипт для создания тестовых данных
│   ├── test_dataset.py          # ✅ 29 тестов для BCZDataset
│   ├── test_metrics.py          # ✅ 13 тестов для метрик
│   └── test_models.py           # ✅ 13 тестов для моделей
├── examples/
│   └── dataloader.py            # ✅ Пример использования dataset
├── docs/
│   └── dataset_usage.md         # Документация по датасету
├── data/                         # Реальные TFRecord данные (321,970 сэмплов)
├── bc-z/                         # Оригинальная реализация Google
├── .plan.md                      # Детальный план реализации
└── CLAUDE.md                     # Инструкции для Claude Code

✅ = Готово    🔜 = В планах
```

## Установка

Проект использует **uv** для управления зависимостями (Python 3.12+):

```bash
# Установить зависимости
uv sync

# Активировать виртуальное окружение
source .venv/bin/activate
```

**Основные зависимости:**
- PyTorch - для нейронных сетей
- TensorFlow - для чтения TFRecord файлов
- Pillow - для обработки изображений
- pytest - для тестирования
- ruff - для линтинга и форматирования

## Использование Dataset

### Базовый пример

```python
from pathlib import Path
from src.data.dataset import BCZDataset
from torch.utils.data import DataLoader

# Создать dataset
dataset = BCZDataset(
    data_path=Path("data/bcz-21task_v9.0.1.tfrecord"),
    image_size=(100, 100),
    mode="train",
    num_waypoints=10,
)

# Получить один сэмпл
sample = dataset[0]
print(f"Image shape: {sample['image'].shape}")              # (3, 100, 100)
print(f"Embedding: {sample['sentence_embedding'].shape}")   # (512,)
print(f"Position: {sample['present_xyz'].shape}")           # (3,)
print(f"Actions: {sample['future_xyz_residual'].shape}")    # (10, 3)

# Использовать с DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for batch in dataloader:
    images = batch["image"]              # (32, 3, 100, 100)
    embeddings = batch["sentence_embedding"]  # (32, 512)
    # ... training code
```

### Структура данных

Каждый сэмпл содержит:

| Ключ | Shape | Описание |
|------|-------|----------|
| `image` | (3, H, W) | RGB изображение (нормализовано в [0, 1]) |
| `sentence_embedding` | (512,) | Task embedding от предобученной модели |
| `present_xyz` | (3,) | Текущая позиция end-effector (x, y, z) |
| `present_axis_angle` | (3,) | Текущая ориентация (axis-angle) |
| `present_gripper` | (1,) | Текущее состояние gripper [0=open, 1=closed] |
| `future_xyz_residual` | (10, 3) | Residual действия для позиции |
| `future_axis_angle_residual` | (10, 3) | Residual действия для ориентации |
| `future_target_close` | (10, 1) | Целевые состояния gripper |

**Больше примеров:** см. `examples/dataloader.py` и `docs/dataset_usage.md`

## Тестирование

Запустить все тесты:

```bash
# Все тесты
uv run pytest tests/ -v

# Только dataset тесты
uv run pytest tests/test_dataset.py -v

# Только metrics тесты
uv run pytest tests/test_metrics.py -v
```

**Текущий статус:** 55/55 тестов проходит ✅
- Dataset: 29 тестов
- Metrics: 13 тестов
- Models: 13 тестов

Тесты используют самодостаточные fixture данные в `tests/fixtures/` и не зависят от внешних больших датасетов.

## Линтинг и форматирование

```bash
# Автоматическое форматирование
uv run ruff format .

# Проверка и автофикс
uv run ruff check --fix .

# Запустить все проверки
uv run ruff format . && uv run ruff check --fix . && uv run pytest tests/
```

## Разработка

### Стиль кода

- **Форматтер:** ruff (line length: 100)
- **Type hints:** обязательны для всех функций
- **Docstrings:** Google-style для всех публичных функций
- **Параметры:** каждый на новой строке (включая `self`)

```python
def example_function(
    self,
    param1: str,
    param2: int = 10,
) -> dict[str, Tensor]:
    """
    Short description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description
    """
```

### Workflow

Проект следует двухфазному workflow:

1. **Design** - Исследование и планирование
2. **Build** - Реализация с коммитами и тестированием

**Рекомендуемые паттерны:**
- Explore → Plan → Code → Commit
- Write tests, commit; code, iterate, commit
- Write code, screenshot result, iterate

Подробнее см. секции ниже.

## Текущий прогресс

**Week 1: Foundation + Data Pipeline** ✅ ЗАВЕРШЕНО
- [x] Структура проекта
- [x] Настройка зависимостей (uv, tensorflow)
- [x] Линтинг (ruff) и pre-commit hooks
- [x] BCZDataset для TFRecord формата
- [x] Метрики для оценки
- [x] 42 unit теста (все проходят)
- [x] Документация и примеры

**Week 2: Model Implementation** ✅ ЗАВЕРШЕНО
- [x] ResNet18/50 backbone (`src/models/backbone.py`)
- [x] FiLM layers для task conditioning (`src/models/film.py`)
- [x] BCZPolicy: stem → 4×(stage+FiLM) → GAP → concat(state) → MLP head
- [x] Single MLP head (TODO: разделение на 3 головы)
- [x] 13 model тестов (shape, affine-identity, gradient flow, batch independence)

**Week 3: Training Loop** 🔜 В ПЛАНАХ
- [ ] Trainer class
- [ ] Loss functions
- [ ] Optimizer и scheduler
- [ ] Wandb integration

См. `.plan.md` для детального плана.

## Документация

- **`.plan.md`** - Детальный 9-фазный план реализации
- **`CLAUDE.md`** - Инструкции для разработки и AI assistant
- **`docs/dataset_usage.md`** - Подробное руководство по датасету
- **`bc-z/`** - Оригинальная реализация Google Research
