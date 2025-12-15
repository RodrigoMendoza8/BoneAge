# Predicción de Edad Ósea en Manos con Deep Learning
Este repositorio contiene el proyecto final para la materia de Aprendizaje Automático II, enfocado en la estimación de la edad ósea a partir de radiografías de manos utilizando redes neuronales convolucionales (CNN),

# Descripción del Proyecto
El objetivo principal de este proyecto es desarrollar un modelo de regresión capaz de predecir la edad ósea (en meses) de un paciente basándose en una radiografía de su mano. Se implementan y comparan arquitecturas de Deep Learning pre-entrenadas (VGG16 y ResNet50) para realizar la extracción de características y la predicción final.

Además, se integra la técnica Grad-CAM (Gradient-weighted Class Activation Mapping) para proporcionar interpretabilidad al modelo, generando mapas de calor que resaltan las regiones de la radiografía más relevantes para la decisión de la red.

# Contexto Académico
Institución: Universidad Iberoamericana León

Materia: Aprendizaje Automático II

Profesor: Mariano José Juan Rivera Meraz

Fecha de Entrega: 4 de Julio de 2025

# Dataset
El proyecto utiliza el conjunto de datos RSNA Bone Age, el cual es descargado directamente desde Kaggle utilizando la librería kagglehub.

Fuente: RSNA Bone Age en Kaggle

Estructura: Imágenes de radiografías de manos y un archivo CSV con las etiquetas de edad ósea (en meses) y género.

# Metodología
Preprocesamiento de Datos:

Carga de imágenes y redimensionamiento a 224x224 píxeles.

Conversión a formato RGB.

Uso de preprocess_input específico para cada modelo (VGG16/ResNet50).

Implementación de un generador de datos personalizado BoneAgeDataGenerator basado en tf.keras.utils.Sequence para el manejo eficiente de lotes durante el entrenamiento.

# Entrenamiento:

División del dataset en conjuntos de entrenamiento (80%) y validación (20%).

Uso de Transfer Learning con pesos de ImageNet.

Congelamiento inicial de las capas base y entrenamiento de las capas densas personalizadas.

Fine-tuning opcional descongelando los últimos bloques convolucionales (block5 para VGG16) para ajustar el modelo a la tarea específica.

# Modelos Utilizados
VGG16: Modelo base seguido de capas Flatten y Dense (con activación ReLU).

ResNet50: Modelo base seguido de capas Flatten y Dense.

Métricas de Evaluación:

Función de pérdida: mean_squared_error (MSE).

Métrica de desempeño: mean_absolute_error (MAE).

# Resultados y Visualización
El proyecto incluye una sección de Interpretabilidad donde se visualizan los resultados cualitativos:

Comparación entre la Edad Real vs Edad Predicha.

Generación de mapas de calor con Grad-CAM superpuestos a la imagen original para identificar qué zonas óseas (falanges, muñeca, etc.) está "mirando" el modelo para determinar la edad.

# Requisitos
El código está desarrollado en Python y requiere las siguientes librerías principales:

tensorflow / keras

numpy

pandas

matplotlib

Pillow (PIL)

scikit-learn

scikit-image

kagglehub

Para ejecutarlo, asegúrate de instalar las dependencias:

Bash

pip install tensorflow numpy pandas matplotlib pillow scikit-learn scikit-image kagglehub

# Autores
Rodrigo Mendoza Rodríguez - 192462-2

Fernando Leon Franco - 192488-7