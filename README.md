# BC-Z Implementation

Обучающий проект для улучшения навыков Claude Code и реализации BC-Z (Behavioral Cloning with Zero-shot task generalization) для робототехники.

**Статус:** Week 1 Complete ✅ - Data Pipeline реализован и протестирован (29/29 тестов)

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
│   ├── models/                   # 🔜 Нейронные сети (ResNet + FiLM)
│   ├── training/                 # 🔜 Training loop и trainer
│   └── utils/
│       └── metrics.py           # ✅ Метрики для оценки (13 тестов)
├── tests/
│   ├── fixtures/
│   │   ├── test_data.tfrecord   # Тестовые данные (5 сэмплов, 154 KB)
│   │   └── create_test_data.py  # Скрипт для создания тестовых данных
│   ├── test_dataset.py          # ✅ 29 тестов для BCZDataset
│   └── test_metrics.py          # ✅ 13 тестов для метрик
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

**Текущий статус:** 42/42 тестов проходит ✅
- Dataset: 29 тестов
- Metrics: 13 тестов

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

**Week 2: Model Implementation** 🔜 В ПЛАНАХ
- [ ] ResNet18/50 backbone
- [ ] FiLM layers для task conditioning
- [ ] Multi-head prediction для actions
- [ ] Model tests

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

## Try common workflows
Claude Code doesn’t impose a specific workflow, giving you the flexibility to use it how you want. Within the space this flexibility affords, several successful patterns for effectively using Claude Code have emerged across our community of users:

a. Explore, plan, code, commit
This versatile workflow suits many problems:

Ask Claude to read relevant files, images, or URLs, providing either general pointers ("read the file that handles logging") or specific filenames ("read logging.py"), but explicitly tell it not to write any code just yet.
This is the part of the workflow where you should consider strong use of subagents, especially for complex problems. Telling Claude to use subagents to verify details or investigate particular questions it might have, especially early on in a conversation or task, tends to preserve context availability without much downside in terms of lost efficiency.
Ask Claude to make a plan for how to approach a specific problem. We recommend using the word "think" to trigger extended thinking mode, which gives Claude additional computation time to evaluate alternatives more thoroughly. These specific phrases are mapped directly to increasing levels of thinking budget in the system: "think" < "think hard" < "think harder" < "ultrathink." Each level allocates progressively more thinking budget for Claude to use.
If the results of this step seem reasonable, you can have Claude create a document or a GitHub issue with its plan so that you can reset to this spot if the implementation (step 3) isn’t what you want.
Ask Claude to implement its solution in code. This is also a good place to ask it to explicitly verify the reasonableness of its solution as it implements pieces of the solution.
Ask Claude to commit the result and create a pull request. If relevant, this is also a good time to have Claude update any READMEs or changelogs with an explanation of what it just did.
Steps #1-#2 are crucial—without them, Claude tends to jump straight to coding a solution. While sometimes that's what you want, asking Claude to research and plan first significantly improves performance for problems requiring deeper thinking upfront.

b. Write tests, commit; code, iterate, commit
This is an Anthropic-favorite workflow for changes that are easily verifiable with unit, integration, or end-to-end tests. Test-driven development (TDD) becomes even more powerful with agentic coding:

Ask Claude to write tests based on expected input/output pairs. Be explicit about the fact that you’re doing test-driven development so that it avoids creating mock implementations, even for functionality that doesn’t exist yet in the codebase.
Tell Claude to run the tests and confirm they fail. Explicitly telling it not to write any implementation code at this stage is often helpful.
Ask Claude to commit the tests when you’re satisfied with them.
Ask Claude to write code that passes the tests, instructing it not to modify the tests. Tell Claude to keep going until all tests pass. It will usually take a few iterations for Claude to write code, run the tests, adjust the code, and run the tests again.
At this stage, it can help to ask it to verify with independent subagents that the implementation isn’t overfitting to the tests
Ask Claude to commit the code once you’re satisfied with the changes.
Claude performs best when it has a clear target to iterate against—a visual mock, a test case, or another kind of output. By providing expected outputs like tests, Claude can make changes, evaluate results, and incrementally improve until it succeeds.

c. Write code, screenshot result, iterate
Similar to the testing workflow, you can provide Claude with visual targets:

Give Claude a way to take browser screenshots (e.g., with the Puppeteer MCP server, an iOS simulator MCP server, or manually copy / paste screenshots into Claude).
Give Claude a visual mock by copying / pasting or drag-dropping an image, or giving Claude the image file path.
Ask Claude to implement the design in code, take screenshots of the result, and iterate until its result matches the mock.
Ask Claude to commit when you're satisfied.
Like humans, Claude's outputs tend to improve significantly with iteration. While the first version might be good, after 2-3 iterations it will typically look much better. Give Claude the tools to see its outputs for best results.