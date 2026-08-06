# Taller de Aprendizaje Automático No Supervisado

Entrega individual del Proyecto 7 del Bootcamp de IA 

Autora: Gisella Cuesta


## Descripción del proyecto

Este repositorio contiene un taller práctico de machine learning no supervisado que cubre reducción de dimensionalidad (PCA, t-SNE), clustering (K-Means, Agglomerative Clustering, GMM, DBSCAN) y detección de anomalías (Isolation Forest).

El taller se divide en dos notebooks complementarios, cada uno con su propio dataset, elegidos para contrastar cómo se comporta el mismo conjunto de técnicas según el tipo de datos y según si se dispone o no de una etiqueta de referencia.

## Estructura del repositorio

| Notebook | Dataset | Tipo de datos | Etiqueta disponible |
|---|---|---|---|
| [`workshop-clustering-Mushrooms.ipynb`](workshop-clustering-Mushrooms.ipynb) | [`data/mushrooms.csv`](data/mushrooms.csv) | Categóricos | Sí — `class` (usada solo para validación) |
| [`workshop-clustering-creditcard.ipynb`](workshop-clustering-creditcard.ipynb) | [`data/credit_card.csv`](data/credit_card.csv) | Numéricos | No — segmentación no supervisada real |

Ambos notebooks deben revisarse juntos: la Parte 2 se apoya en los conceptos introducidos en la Parte 1.

## Parte 1 — Setas (datos categóricos, con etiqueta)

Dataset: [Mushroom Dataset (Kaggle)](https://www.kaggle.com/uciml/mushroom-classification) / [UCI](https://archive.ics.uci.edu/ml/datasets/Mushroom). Cada fila describe una seta mediante aproximadamente 22 variables categóricas. La variable `class` es binaria (`e` = comestible, `p` = venenosa) y se utiliza únicamente para validar los resultados del clustering a posteriori, nunca como variable de entrada.

Trabajo realizado:

- Carga de datos, EDA, detección de un valor nulo encubierto (`'?'` en `stalk-root`) y de una columna constante (`veil-type`).
- Imputación con la moda y One-Hot Encoding.
- PCA y t-SNE para visualizar en 2D un dataset con más de 100 dimensiones.
- Random Forest como baseline supervisado, junto con un estudio de cuántos componentes de PCA son necesarios para mantener su accuracy.
- Clustering con K-Means (codo + silhouette), Agglomerative Clustering (con dendrograma), GMM y DBSCAN.
- Comparativa de DBSCAN entre distancia euclídea y distancia de Jaccard, mostrando que la elección de la métrica de distancia importa en datos categóricos codificados con One-Hot.
- Validación frente a la etiqueta real con Adjusted Rand Index (ARI) y Normalized Mutual Information (NMI).
- Isolation Forest para detección de anomalías.

Resultados clave:

- Accuracy del Random Forest baseline (todas las features): 100%.
- K-Means (k=2) sobre una proyección PCA de 10 componentes: ARI = 0.62, NMI = 0.57.
- DBSCAN con distancia euclídea sobre la proyección PCA colapsa en un único cluster, mientras que DBSCAN con distancia de Jaccard sobre los datos One-Hot originales recupera estructura significativa (ARI = 0.29) — una ilustración directa de por qué la métrica de distancia importa con datos categóricos.
- Isolation Forest marcó un 5% de las muestras como anómalas.

## Parte 2 — Clientes de tarjeta de crédito (datos numéricos, sin etiqueta)

Dataset: [Credit Card Dataset for Clustering (Kaggle)](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata). Comportamiento de uso de unos 9.000 titulares de tarjeta de crédito durante 6 meses, descrito con 17 variables numéricas. No hay etiqueta: se trata de aprendizaje no supervisado en sentido estricto, donde el éxito se mide con métricas internas y, sobre todo, con la interpretabilidad de los segmentos resultantes. El objetivo es segmentar clientes para una estrategia de marketing.

Trabajo realizado:

- Carga de datos, EDA y tratamiento de valores nulos (imputación con la mediana, robusta frente a datos sesgados).
- Observación del sesgo típico en variables financieras.
- Estandarización (`StandardScaler`), necesaria dada la gran diferencia de escalas entre variables.
- PCA: scree plot de varianza explicada y proyección en 2D (los datos forman una nube continua, no grupos separados).
- Clustering con K-Means (codo + silhouette), Agglomerative Clustering (dendrograma), GMM y DBSCAN.
- Validación sin etiqueta mediante silhouette, Davies-Bouldin y Calinski-Harabasz.
- Visualización de los segmentos con t-SNE.
- Perfilado de segmentos (valor medio por cluster, heatmap estandarizado) e interpretación de negocio.
- Isolation Forest para marcar clientes atípicos.

Resultados clave:

- 7 componentes de PCA retienen aproximadamente el 80% de la varianza.
- Mejor k según silhouette score: 3.
- Segmentos resultantes:
  - Cluster 0 (1.568 clientes, ~17,5%) — Riesgo / uso intensivo de crédito: alto uso de cash advance, baja frecuencia de compra. Candidatos a revisión de límite o a ofertas de refinanciación.
  - Cluster 1 (6.132 clientes, ~68,5%) — Bajo perfil / actividad limitada: el grupo mayoritario, balance bajo, uso mínimo de cash advance. Objetivo de campañas de activación.
  - Cluster 2 (1.250 clientes, ~14%) — VIP / alto valor: volumen y frecuencia de compra muy altos. Objetivo de programas de fidelización y beneficios premium.
- Isolation Forest marcó un 5% de los clientes como atípicos.

## Por qué dos datasets

| | Setas | Tarjetas de crédito |
|---|---|---|
| Variables | Categóricas | Numéricas |
| Preprocesamiento clave | One-Hot Encoding | Escalado / imputación |
| Etiqueta | Sí (solo para validación) | No |
| Enfoque de validación | ARI / NMI frente a etiqueta | Métricas internas + interpretabilidad |
| Estructura | Grupos separables | Nube continua |
| Mejor distancia | Jaccard mejor que euclídea | Euclídea sobre datos escalados |

Conclusión transversal: ningún algoritmo o métrica es superior en todos los casos. La elección correcta de preprocesamiento, métrica de distancia y estrategia de validación depende del tipo de datos y del problema concreto.

## Tecnologías

- Python, Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn: `PCA`, `TSNE`, `KMeans`, `AgglomerativeClustering`, `GaussianMixture`, `DBSCAN`, `IsolationForest`, `RandomForestClassifier` y métricas de clustering
- SciPy (`linkage`, `dendrogram`, `pdist`)

## Cómo ejecutar

1. Clonar el repositorio.
2. Los datasets están en la carpeta `data/` (`data/mushrooms.csv` y `data/credit_card.csv`); ambos notebooks los leen con ruta relativa, por ejemplo `pd.read_csv("data/mushrooms.csv")`.
3. Abrir cada notebook (Jupyter o VS Code) y ejecutar todas las celdas en orden.
4. Empezar por la Parte 1 (Setas) y continuar después con la Parte 2 (Tarjetas de crédito).

Entorno: Python 3.12.10. Paquetes necesarios: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `scipy`.

## Flujo de trabajo del repositorio

El trabajo se desarrolló en la rama `develop` con commits manuales frecuentes e incrementales, y se fusionó a `main` para la entrega final. Este flujo replica una práctica de Git profesional estándar (rama de desarrollo/feature más una rama estable reservada para las entregas).
