# Система детекции детей без сопровождения взрослых

Прототип системы компьютерного зрения: на видео с камеры наблюдения находит людей, отличает детей от взрослых и поднимает тревогу, если ребёнок остаётся без взрослого рядом дольше заданного порога.

Репозиторий: https://github.com/Knexy04/VKR

## Требования

- Python 3.10–3.12 (рекомендуется)
- 4+ ГБ оперативной памяти
- Веса моделей уже лежат в репозитории, отдельно скачивать ничего не нужно

Первая установка `requirements.txt` тяжёлая: вместе с `ultralytics` подтянется PyTorch.

## Быстрый старт

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8501
```

Откройте в браузере http://127.0.0.1:8501

1. В поле «путь на сервере» оставьте `data/2.mp4` (демо уже в комплекте).
2. При желании уменьшите «Порог (сек)» до 3, чтобы тревога сработала быстрее.
3. Нажмите **Старт**.
4. На кадре: зелёный — взрослый, жёлтый — ребёнок, красный — тревога. Справа счётчики и журнал тревог.

Остановка: кнопка **Стоп** или `Ctrl+C` в терминале.

Первый запуск загружает модели 10–30 секунд. На CPU обработка медленнее реального времени — это нормально. На Mac с Apple Silicon используется MPS, на NVIDIA — CUDA.

Чтобы ускорить работу на слабом компьютере, в `config.py` поставьте `YOLO_IMGSZ = 640`.

## Другие способы запуска

Консоль:

```bash
python main.py --source data/2.mp4
python main.py --source data/2.mp4 --threshold 3.0 --radius 150
python main.py --source 0
```

Streamlit-интерфейс:

```bash
streamlit run web_app.py
```

Docker (АРМ оператора на FastAPI):

```bash
docker compose up --build
```

Приложение будет на http://localhost:8501. Папки `data/` и `models/` монтируются в контейнер.

## Зависимости

Список в `requirements.txt`:

- ultralytics
- opencv-python
- numpy
- supervision
- onnxruntime
- fastapi, uvicorn, python-multipart
- streamlit
- requests
- pytest

Обученные веса:

- `yolo26n-pose.pt` — детекция людей и 17 ключевых точек скелета
- `models/yolo_child_detector.pt` — детектор классов adult / child
- `models/age_classifier_v2.onnx` — классификатор лица (MobileNetV2)

## Как это работает

```
видео → YOLO26-pose + YOLO26s (adult/child) + BoT-SORT
     → ансамбль возраста (до 6 оценщиков)
     → конечный автомат сопровождения
     → тепловая карта + АРМ оператора
```

Самая сложная часть — `age_classifier.py`: класс `EnsembleAgeClassifier` собирает голоса обученных моделей и геометрических эвристик, сглаживает решение по `track_id` и не даёт метке мигать.

Логика тревоги — `alert_logic.py`: если ближайший взрослый дальше `PROXIMITY_RADIUS_PX` дольше `ALERT_THRESHOLD_SEC`, формируется событие.

## Структура

```
├── server.py              # АРМ оператора (FastAPI + MJPEG)
├── web_app.py             # альтернативный интерфейс на Streamlit
├── main.py                # консольный запуск
├── config.py              # все параметры системы
├── detection.py           # детекция и трекинг
├── age_classifier.py      # ансамбль классификации возраста
├── alert_logic.py         # конечный автомат тревог
├── heatmap.py             # тепловая карта зон риска
├── visualization.py       # рамки, подписи, баннер
├── utils.py               # FPS, webhook
├── models.py              # структура детекции человека
├── botsort_reid.yaml      # конфиг трекера
├── train_local.py         # обучение классификатора лица
├── train_yolo_child.py    # обучение детектора adult/child
├── models/                # веса .pt и .onnx
├── data/2.mp4             # демо-видео
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Основные параметры (`config.py`)

| Параметр | Смысл | По умолчанию |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | порог детекции | `0.25` |
| `PROXIMITY_RADIUS_PX` | радиус сопровождения, px | `200` |
| `ALERT_THRESHOLD_SEC` | секунд без взрослого до тревоги | `5.0` |
| `AGE_CLASSIFIER` | режим классификации | `ensemble` |
| `YOLO_IMGSZ` | разрешение инференса | `1280` |
| `AGE_CHILD_THRESHOLD` | порог решения child / adult | `0.55` |

## Обучение моделей

Не нужно для запуска прототипа. Скрипты оставлены как часть исходников ВКР:

```bash
python train_yolo_child.py
python train_local.py
```

Для обучения нужны отдельные датасеты (Adult-Child, UTKFace, FGNET) — в этот архив они не входят.

## Автор

Маркин Никита Владимирович  
Выпускная квалификационная работа, Московский политех, 2026  
https://github.com/Knexy04/VKR
