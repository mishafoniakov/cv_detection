# Computer Vision

Классификация цветов, классификация овощей, детекция животных и детекция объектов.

Репозиторий собран как портфолио-проект: самостоятельные Jupyter-ноутбуки, воспроизводимый запуск и явные инструкции по данным.

| Задача | Модель | Данные | Результат |
| --- | --- | --- | --- |
| Классификация цветов | MobileNetV2 + свой классификатор (Keras) | TensorFlow Flowers (`flower_photos`) | **85.56%** accuracy на тесте |
| Классификация овощей | CNN (Conv2D 32-64-128, Keras) | [Vegetable Image Dataset](https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset) | 15 классов, обучение с диска через kagglehub |
| Детекция животных | YOLOv8n (Ultralytics) | [Animals-10](https://www.kaggle.com/datasets/alessiocorrado99/animals10) | bounding boxes + подсчёт объектов |
| Детекция объектов | Haar Cascade (OpenCV) | номера авто, морды котов, настенные часы | bounding boxes на тестовых фото |

Python **3.11**.

---

## Содержание

- [Что внутри](#что-внутри)
- [Структура](#структура)
- [Быстрый старт](#быстрый-старт)
- [Ноутбук 1. Классификация цветов](#ноутбук-1-классификация-цветов)
- [Ноутбук 2. Детекция животных](#ноутбук-2-детекция-животных)
- [Ноутбук 3. Классификация овощей](#ноутбук-3-классификация-овощей)
- [Ноутбук 4. Детекция объектов](#ноутбук-4-детекция-объектов)
- [Зависимости](#зависимости)

## Что внутри

**Классификация цветов.** Transfer learning на MobileNetV2: веса backbone заморожены, сверху — полносвязная голова. Датасет делится 80/20, обучение 20 эпох, затем оценка и инференс на отдельных фото.

**Классификация овощей.** Своя CNN на Keras: 15 классов Vegetable Image Dataset, 200 фото на класс, аугментация в модели, обучение 10 эпох.

**Детекция животных.** Предобученный YOLOv8n на изображении из Animals-10: визуализация боксов, подсчёт объектов по классам и обёртка `ImageDetectionYOLO` для прогона по папке.

**Детекция объектов.** Классические Haar-каскады OpenCV: автомобильные номера, морды котов и настенные часы.

## Структура

```text
cv_detection/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── .github/workflows/validate.yml
├── images/                   # тестовые JPEG (номера, коты, часы; test_1…3 для цветов)
├── xml/                      # Haar-каскады OpenCV
├── flower_photos/            # датасет цветов (скачать отдельно)
├── model/                    # веса MobileNetV2 (скачать отдельно)
└── notebooks/
    ├── cv_flowers.ipynb
    ├── cv_detection.ipynb
    ├── detect_goods.ipynb
    └── object_detection.ipynb
```

В `notebooks/` лежат только `.ipynb`. Запускайте ячейки из каталога `notebooks/`: пути к данным идут на уровень выше (`../images`, `../xml`, `../flower_photos`, `../model`).

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

Ноутбук `cv_flowers.ipynb` читает три локальных пути из корня репозитория:

1. Датасет TensorFlow Flowers — `flower_photos/<класс>/*.jpg` (daisy, dandelion, roses, sunflowers, tulips)
2. Веса MobileNetV2 — `model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5`
3. Три тестовых JPEG — `images/test_1.jpg`, `images/test_2.jpg`, `images/test_3.jpg`

Из корня репозитория:

```bash
curl -L http://download.tensorflow.org/example_images/flower_photos.tgz -o /tmp/flower_photos.tgz
tar -xzf /tmp/flower_photos.tgz -C .

mkdir -p model
curl -L "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5" \
  -o model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
```

Затем положите `test_1.jpg`, `test_2.jpg`, `test_3.jpg` в `images/`. Сами датасет и веса `.h5` в git не входят.

### Данные для детекции животных

Animals-10 скачивается из ячейки через `kagglehub` (`alessiocorrado99/animals10`). Веса `yolov8n.pt` подтягивает Ultralytics при первом запуске. Для Kaggle может понадобиться [API-токен](https://github.com/Kaggle/kagglehub).

### Данные для классификации овощей

Vegetable Image Dataset скачивается из ячейки `detect_goods.ipynb` через `kagglehub` (`misrakahmed/vegetable-image-dataset`).

### Данные для детекции объектов

Для `object_detection.ipynb` уже лежат:

- фото в [`images/`](images/) — `number_*.jpg`, `cat_*.jpg`, `clock*.jpg`
- каскады в [`xml/`](xml/) — `haarcascade_russian_plate_number.xml`, `haarcascade_frontalcatface.xml`, `haarcascade_wallclock.xml`

Дополнительно ничего скачивать не нужно.

## Ноутбук 1. Классификация цветов

Файл: [`notebooks/cv_flowers.ipynb`](notebooks/cv_flowers.ipynb)

Пайплайн:

1. Загрузка изображений по папкам-классам из `../flower_photos`
2. Нормализация пикселей в `[0, 1]`, split 80/20 (`random_state=42`)
3. MobileNetV2 без головы, `trainable=False`, веса из `../model`
4. Sequential: Flatten → Dense(64) → Dense(128) → Dense(64) → Dropout(0.5) → softmax
5. Adam, `sparse_categorical_crossentropy`, 20 эпох, `batch_size=32`, вход 128×128
6. Оценка на тесте, графики accuracy/loss, инференс на `../images/test_1.jpg` … `test_3.jpg`

Зафиксированный результат прогона: **точность на тестовой выборке 85.56%**.

## Ноутбук 2. Детекция животных

Файл: [`notebooks/cv_detection.ipynb`](notebooks/cv_detection.ipynb)

Пайплайн:

1. YOLOv8n (`yolov8n.pt`)
2. Пример из `raw-img/elefante`
3. Детекция, отрисовка боксов, подсчёт объектов по классам COCO
4. Класс `ImageDetectionYOLO`: подготовка кадра, инференс, визуализация
5. Прогон по первым 10 изображениям папки

## Ноутбук 3. Классификация овощей

Файл: [`notebooks/detect_goods.ipynb`](notebooks/detect_goods.ipynb)

Пайплайн:

1. Скачивание Vegetable Image Dataset через `kagglehub`
2. Загрузка до 200 фото на класс, ресайз 128×128, нормализация в `float32`
3. Split 80/20 (`random_state=42`, `stratify`)
4. CNN: Conv2D 32-64-128 + MaxPool, Dense(128), Dropout, softmax на 15 классов
5. Аугментация слоями Keras (`RandomFlip`, `RandomRotation`, `RandomZoom`)
6. Adam, `sparse_categorical_crossentropy`, 10 эпох, оценка и инференс на тесте

На Python 3.14 ноутбук идёт через Keras 3 + PyTorch: пакет TensorFlow для этой версии интерпретатора недоступен.

## Ноутбук 4. Детекция объектов

Файл: [`notebooks/object_detection.ipynb`](notebooks/object_detection.ipynb)

Пайплайн:

1. Haar Cascade для российских автомобильных номеров
2. Haar Cascade для морды кота
3. Предобработка кадра и детекция настенных часов
4. Отрисовка bounding boxes на фото из `../images`, каскады из `../xml`

## Зависимости

Список пакетов — в [`requirements.txt`](requirements.txt). Основные:

- классификация цветов: TensorFlow / Keras, scikit-learn, NumPy, pandas, Matplotlib
- классификация овощей: Keras, PyTorch, scikit-learn, kagglehub, Pillow
- детекция животных: Ultralytics, OpenCV, kagglehub
- детекция объектов: OpenCV, Matplotlib

Датасеты и веса в git не входят (см. [`.gitignore`](.gitignore)). Фото для Haar-детекции и XML-каскады — входят.

## Лицензия

[MIT](LICENSE)
