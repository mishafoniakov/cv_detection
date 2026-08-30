# Датасет цветов

Ноутбук `cv_flowers.ipynb` читает изображения из этой папки:

```text
notebooks/flower_photos/<класс>/*.jpg
```

Ожидается набор [TensorFlow Flowers](http://download.tensorflow.org/example_images/flower_photos.tgz) (daisy, dandelion, roses, sunflowers, tulips).

## Как положить данные сюда

Из корня репозитория:

```bash
curl -L http://download.tensorflow.org/example_images/flower_photos.tgz -o /tmp/flower_photos.tgz
tar -xzf /tmp/flower_photos.tgz -C notebooks --strip-components=0
```

После распаковки структура должна выглядеть так:

```text
notebooks/flower_photos/daisy/
notebooks/flower_photos/dandelion/
notebooks/flower_photos/roses/
notebooks/flower_photos/sunflowers/
notebooks/flower_photos/tulips/
```

Сами изображения в git не коммитятся.
