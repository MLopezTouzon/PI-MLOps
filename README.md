<p align=center><img src=https://d31uz8lwfmyn8g.cloudfront.net/Assets/logo-henry-white-lg.png><p>

# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
<img src="https://user-images.githubusercontent.com/67664604/217914153-1eb00e25-ac08-4dfa-aaf8-53c09038f082.png"  height=300>
</p>

## Desarrollo de ETL y EDA

Los DataSets que fueron proporcionados estaban en formatos json los cuales tuvieron que ser desempaquetados para su estraccion de datos; ademas, al momento de transformar los datos, tenia que desanidar ciertas columnas, borrar datos nulos y asi mismo asignarle a algunos datos nulos algunos valores para que no haya problemas en el desarrollo de las funciones.
En la carga de los data frames uno de ellos se transformo en archivos .gz; asi mismo, entre dos data frames se realizo un mapeo para la optimizacion de la memoria.
Por otro lado los datos fueron explorados y en ciertos graficos se muetran el top videojuegos, el analisis de sentimiento y usuarios con mas juegos recomendados, entre otros.  
<p align="center">
<img src="https://github.com/HX-PRomero/PI_ML_OPS/raw/main/src/DiagramaConceptualDelFlujoDeProcesos.png"  height=500>
</p>

## CONTENIDO DE LA API

La API, el cual esta en [Render](https://render.com/docs/free#free-web-services) contiene 5 funciones y un sistema de recomendacion los cuales fueron solicitados, los cuales realizan el funcionamiento de genero mas jugado, horas jugadas por determinados usuarios en dichos generos, año en donde mas juegos se recomendaron, entre otros.
Con respecto a los sistemas de recomendacion nos retornara 5 juegos similares, hice 3 endpoints para recomendar juegos, uno de item-item y dos de user-item hechos de distintas maneras.

## LINKS
 + Repositorio de (GITHUB) : https://github.com/MLopezTouzon/PI-MLOps.git
 + Link de Render : https://pi-mlops-matias-agustin-lopez-touzon.onrender.com/docs
 + Video : 