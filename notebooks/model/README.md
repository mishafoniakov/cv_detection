# Веса MobileNetV2

Ноутбук `cv_flowers.ipynb` загружает backbone из файла:

```text
notebooks/model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
```

Это стандартные веса Keras MobileNetV2 (`alpha=1.0`, вход 128×128, без классификационной головы).

## Как скачать

```bash
mkdir -p notebooks/model
curl -L "https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5" \
  -o notebooks/model/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5
```

Файл `.h5` в git не коммитится.
