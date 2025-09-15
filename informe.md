# Informe detallado del modelo LSTM

## 1. Descripción de características adicionales y su relevancia

Además de la secuencia de palabras (críticas de películas) representadas como índices enteros, el modelo incorpora dos características adicionales para cada muestra:

- **Logaritmo de la longitud de la crítica**: Se calcula como $\log(1 + \text{longitud})$. Esta característica ayuda a capturar la variabilidad en la extensión de las críticas, que puede estar relacionada con la polaridad del sentimiento.
- **Longitud normalizada**: Es la longitud de la crítica dividida por la longitud máxima permitida ($\text{longitud} / \text{MAX\_LEN}$). Esto permite al modelo tener información sobre si la crítica fue truncada o si es especialmente corta, lo que puede influir en la interpretación del sentimiento.

Ambas características se agregan como un vector de entrada adicional al modelo, permitiendo que la red tenga contexto sobre la estructura de cada crítica, no solo su contenido textual.

## 2. Explicación de la arquitectura del modelo

La arquitectura del modelo es la siguiente:

- **Entrada de secuencia**: Una secuencia de longitud fija (`MAX_LEN=331`), donde cada elemento es el índice de una palabra.
- **Capa de Embedding**: Convierte los índices de palabras en vectores densos de dimensión 128 (`UNITS=128`), permitiendo que el modelo aprenda representaciones semánticas de las palabras.
- **SpatialDropout1D**: Aplica dropout a los vectores de embedding para mejorar la generalización.
- **Capa LSTM**: Procesa la secuencia de embeddings, capturando dependencias temporales y contextuales en el texto.
- **GlobalMaxPooling1D y GlobalAveragePooling1D**: Extraen representaciones globales de la secuencia, resumiendo la información más relevante.
- **Concatenación de features**: Se concatenan las salidas de pooling y las características adicionales (log-longitud y longitud normalizada), permitiendo que el modelo utilice tanto información textual como estructural.
- **Normalización y Densas**: Se normalizan las características adicionales y se combinan con la representación textual, pasando por una capa densa con activación ReLU y regularización L2, seguida de dropout.
- **Salida**: Una neurona con activación sigmoide para clasificación binaria (sentimiento positivo/negativo).

## 3. Presentación y análisis de resultados

El modelo fue entrenado usando early stopping y reducción de tasa de aprendizaje basada en la métrica de AUPRC (área bajo la curva de precisión-recall en validación). Durante el entrenamiento, se observó lo siguiente:

- El modelo alcanzó un punto donde no mejoraba la métrica de validación, activando el early stopping.
- Se detectó cierto overfitting, ya que los cambios en dropout, número de unidades o características adicionales no lograron mejorar el desempeño.
- Los resultados finales muestran que el modelo logra una buena capacidad de ajuste, pero su generalización es limitada por la complejidad del problema y la posible falta de regularización adicional.

**Conclusiones**:  
- El uso de características adicionales como la longitud de la crítica aporta información relevante al modelo, permitiendo que la red tenga contexto más allá del contenido textual. 
- La arquitectura propuesta es robusta y moderna, combinando técnicas de regularización y procesamiento secuencial. No obstante, el modelo muestra signos de sobreajuste, por lo que podría explorarse técnicas adicionales como mayor regularización, data augmentation o arquitecturas alternativas para mejorar la generalización.
- El uso de LSTM es bastante efectivo para catalogar reseñas como negativas o positivas, pero 
existen mejores modelos para generalizaciónes más grandes. 

