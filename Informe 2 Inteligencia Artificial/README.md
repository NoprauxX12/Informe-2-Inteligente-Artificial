# Informe 2 — Inteligencia Artificial
## Integrantes
- Juan Sebastián Lizcano Urrea
- Jose Andrés Mendoza Hernandez

Proyecto compuesto por dos casos de estudio en aprendizaje automático: uno aborda **clasificación binaria** sobre señales electromiográficas (EMG) y el otro **regresión** sobre imágenes faciales. Ambos siguen un pipeline completo que va desde la carga y exploración de datos hasta el entrenamiento, la evaluación y la comparación de modelos.

---

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `fatiga_muscular.ipynb` | Clasificación de fatiga muscular a partir de señales EMG durante ciclismo |
| `age_regression_dataloader.ipynb` | Estimación de edad a partir de imágenes faciales mediante redes neuronales convolucionales (CNN) |

---

## 1. Clasificación de fatiga muscular (`fatiga_muscular.ipynb`)

### Problema
Determinar si un sujeto se encuentra en estado de **fatiga** o **no fatiga** durante una sesión de ciclismo, a partir de señales EMG registradas en ocho músculos de las extremidades inferiores.

### Dataset
- **Fuente:** [YominE/Muscle_Fatigue_Cycling](https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling) (Hugging Face)
- **Tamaño:** ~3 000 000 de muestras temporales, 8 canales EMG + variable objetivo binaria


### Pipeline
1. Carga y exploración del dataset (valores nulos, distribución de clases, tipos de variables)
2. Ventaneo temporal de la señal cruda (windowing)
3. Extracción de características estadísticas y espectrales (media, varianza, skewness, kurtosis, densidad espectral de potencia vía Welch)
4. División train/test y estandarización
5. Entrenamiento y ajuste de hiperparámetros con `RandomizedSearchCV` para cinco clasificadores:
   - K-Nearest Neighbors (KNN)
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - MLP (red neuronal densa)
6. Evaluación con accuracy, F1-score, matrices de confusión y curvas de aprendizaje
7. Comparación final de modelos — **mejor resultado: MLP (DNN) con ~89 % de accuracy en test**

### Librerías principales
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `scipy`, `datasets`

---

## 2. Estimación de edad (`age_regression_dataloader.ipynb`)

### Problema
Predecir la **edad** de una persona a partir de una fotografía facial, formulado como un problema de regresión.

### Dataset
- ~24 000 imágenes faciales divididas en train / val / test
- La etiqueta de edad se extrae del nombre de archivo (formato `[age]_[gender]_[race]_[datetime].jpg`)

### Pipeline
1. Implementación de un `Dataset` personalizado en PyTorch con carga perezosa desde disco
2. Transformaciones de preprocesamiento y data augmentation (resize, flip, color jitter, rotación, normalización ImageNet)
3. Construcción de `DataLoader` con batching, shuffle y lectura en paralelo
4. Arquitectura CNN con transfer learning basada en **ResNet18** pre-entrenada y capas de Dropout
5. Entrenamiento con `MSELoss` como función de pérdida y **MAE** como métrica de evaluación
6. Guardado del mejor modelo según validación — **mejor Val MAE: ~6.32 años**

### Librerías principales
`torch`, `torchvision`, `PIL`, `numpy`, `matplotlib`

---

## Requisitos

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
datasets
torch
torchvision
Pillow
```

## Ejecución

Cada notebook es autocontenido y puede ejecutarse de forma independiente en cualquier entorno compatible con Jupyter (JupyterLab, VS Code, Google Colab, etc.). El notebook de fatiga muscular descarga el dataset automáticamente desde Hugging Face; el de estimación de edad espera una carpeta `dataset/` con las imágenes organizadas en subdirectorios `train/`, `val/` y `test/`.
