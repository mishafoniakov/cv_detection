# Computer Vision

Классификация цветов и детекция животных.

Репозиторий собран как портфолио-проект: два самостоятельных Jupyter-ноутбука, воспроизводимый запуск и явные инструкции по данным. Код экспериментов не переписывался — изменена только обвязка репозитория.

| Задача | Модель | Данные | Результат |
| --- | --- | --- | --- |
| Классификация цветов | MobileNetV2 + свой классификатор (Keras) | TensorFlow Flowers (`flower_photos`) | **85.56%** accuracy на тесте |
| Детекция животных | YOLOv8n (Ultralytics) | [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) | bounding boxes + подсчёт объектов |

Python **3.11**.

---

## Содержание

- [Что внутри](#что-внутри)
- [Структура](#структура)
- [Быстрый старт](#быстрый-старт)
- [Ноутбук 1. Классификация цветов](#ноутбук-1-классификация-цветов)
- [Ноутбук 2. Детекция животных](#ноутбук-2-детекция-животных)
- [Зависимости](#зависимости)

## Что внутри

**Классификация.** Transfer learning на MobileNetV2: веса backbone заморожены, сверху — полносвязная голова. Датасет делится 80/20, обучение 20 эпох, затем оценка и инференс на отдельных фото.

**Детекция.** Предобученный YOLOv8n на изображении из Animals-10: визуализация боксов, подсчёт объектов по классам и обёртка `ImageDetectionYOLO` для прогона по папке.

## Структура

```text
cv_detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/workflows/validate.yml
└── notebooks/
    ├── cv_flowers.ipynb      # классификация цветов
    ├── cv_detection.ipynb    # детекция животных
    ├── flower_photos/        # датасет цветов (скачать отдельно)
    ├── model/                # веса MobileNetV2 (скачать отдельно)
    └── test_images/          # test_1.jpg … test_3.jpg для инференса
```

Пути в ноутбуках относительные (`./flower_photos/`, `./model/`, `./test_images`). Запускайте ячейки из каталога `notebooks/`, чтобы они совпали с кодом.

## Быстрый старт

```bash
git clone https://github.com/mishafoniakov/cv_detection.git
cd cv_detection

python3.11 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/
```

### Данные для классификации цветов

Ноутбук `cv_flowers.ipynb` ожидает три локальных пути рядом с собой:

1. Датасет — см. [`notebooks/flower_photos/README.md`](notebooks/flower_photos/README.md)
2. Веса MobileNetV2 — см. [`notebooks/model/README.md`](notebooks/model/README.md)
3. Три тестовых JPEG — см. [`notebooks/test_images/README.md`](notebooks/test_images/README.md)

Кратко из корня репозитория:

```bash
curl -L http://download.tensorflow.org/example_images/flower_photos.tgz -o /tmp/flower_photos.tgz
tar -xzf /tmp/flower_photos.tgz -C notebooks

mkdir -p notebooks/model
curl -L "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5" \
  -o notebooks/model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
```

Затем положите `test_1.jpg`, `test_2.jpg`, `test_3.jpg` в `notebooks/test_images/`.

### Данные для детекции

Animals-10 скачивается из ячейки через `kagglehub` (`alessiocorrado99/animals10`). Веса `yolov8n.pt` подтягивает Ultralytics при первом запуске. Для Kaggle может понадобиться [API-токен](https://github.com/Kaggle/kagglehub).

## Ноутбук 1. Классификация цветов

Файл: [`notebooks/cv_flowers.ipynb`](notebooks/cv_flowers.ipynb)

Пайплайн:

1. Загрузка изображений по папкам-классам
2. Нормализация пикселей в `[0, 1]`, split 80/20 (`random_state=42`)
3. MobileNetV2 без головы, `trainable=False`
4. Sequential: Flatten → Dense(64) → Dense(128) → Dense(64) → Dropout(0.5) → softmax
5. Adam, `sparse_categorical_crossentropy`, 20 эпох, `batch_size=32`, вход 128×128
6. Оценка на тесте, графики accuracy/loss, инференс на `test_images`

Зафиксированный результат прогона: **точность на тестовой выборке 85.56%**.

## Ноутбук 2. Детекция животных

Файл: [`notebooks/cv_detection.ipynb`](notebooks/cv_detection.ipynb)

Пайплайн:

1. YOLOv8n (`yolov8n.pt`)
2. Пример из `raw-img/elefante`
3. Детекция, отрисовка боксов, подсчёт объектов по классам COCO
4. Класс `ImageDetectionYOLO`: подготовка кадра, инференс, визуализация
5. Прогон по первым 10 изображениям папки

## Зависимости

Список пакетов — в [`requirements.txt`](requirements.txt). Основные:

- классификация: TensorFlow / Keras, scikit-learn, NumPy, pandas, Matplotlib
- детекция: Ultralytics, OpenCV, kagglehub

Датасеты и веса в git не входят (см. [`.gitignore`](.gitignore)).

## Лицензия

[MIT](LICENSE)
