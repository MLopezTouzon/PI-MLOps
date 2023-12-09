import pandas as pd
from fastapi import FastAPI, HTTPException


app = FastAPI(title='PI MLOps | Realizado por Matias A. Lopez Touzon ')
# Carga los DataFrame desde el archivo CSV
data_reviews = pd.read_csv(
    'G:\\PI MLOps - STEAM\\australian_user_reviews_ok.csv')
data_items = pd.read_csv('G:\\PI MLOps - STEAM\\data_users_items_ok.csv')
data_steam = pd.read_csv('G:\\PI MLOps - STEAM\\steam_games_ok.csv')


@app.get('/')
def Index():
    return {"message": "Hola!!"}


@app.get('/PlayTimeGenre/{genero}')
async def PlayTimeGenre(genero: str):
    try:
        # Filtrar por el género específico
        df_genero = data_steam[data_steam['genres'].apply(
            lambda x: genero in x)]

        if df_genero.empty:
            # Devolver un mensaje si no hay juegos en el género especificado
            raise HTTPException(
                status_code=404, detail=f"No hay datos para el año {genero}")

        # Encontrar el juego con más horas jugadas
        juego_mas_horas = df_genero.loc[df_genero['playtime_forever'].idxmax()]
        # Obtener el año de lanzamiento del juego con más horas jugadas
        año_mas_horas = int(juego_mas_horas['release_date'])

        # Crear el diccionario de retorno
        resultado = {
            "Año de lanzamiento con más horas jugadas para Género " + genero: año_mas_horas}

        return resultado

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}")


@app.get('/UserForGenre/{genero}')
async def UserForGenre(genero: str):
    try:
        # Filtrar por el género específico
        df_genero = data_steam[data_steam['genres'].apply(
            lambda x: genero in x)]
        df_usuario = data_items[data_items['item_name'].apply(
            lambda x: genero in x)]

        if df_genero.empty or df_usuario.empty:
            # Devolver un mensaje si no hay juegos o usuarios en el género especificado
            raise HTTPException(
                status_code=404, detail="No hay juegos o usuarios en el género especificado")

        # Encontrar el usuario con más horas jugadas
        usuario_mas_horas = str(df_usuario.loc[df_usuario['playtime_forever'].idxmax(
        )]['user_id'])
        # Obtener el total de horas jugadas por el usuario
        horas_mas_usuario = str(df_usuario['playtime_forever'].max())
        # Crear una lista de acumulación de horas jugadas por año
        acumulacion_horas_por_año = df_genero.groupby(
            'release_date')['playtime_forever'].sum().reset_index()
        lista_acumulacion = [{"Año": int(año), "Horas": horas} for año, horas in zip(
            acumulacion_horas_por_año['release_date'], acumulacion_horas_por_año['playtime_forever'])]
        # Crear el diccionario de retorno
        resultado = {"Usuario con más horas jugadas para Género " + genero: usuario_mas_horas,
                     "Total de horas jugadas por el usuario": horas_mas_usuario,
                     "Acumulación de horas jugadas por año": lista_acumulacion}

        return resultado
    except Exception as e:
        # Manejar cualquier otra excepción y devolver un mensaje de error
        raise HTTPException(
            status_code=500, detail=f"Error interno del servidor: {str(e)}")


@app.get('/UsersRecommend/{anio}')
async def UsersRecommend(anio: int):
    try:
        # Obtener una muestra aleatoria del DataFrame
        df_sample = data_reviews.sample(frac=0.15, random_state=42)
        # Fusionar los DataFrames para obtener la información relevante
        df_merged = pd.merge(df_sample[['item_id', 'recommend', 'sentiment_analysis', 'año_posted']],
                             data_items[['item_id', 'item_name']],
                             on='item_id',
                             how='inner')
        # Filtrar por el año especificado
        df_año = df_merged[df_merged['año_posted'] == str(anio)]

        if df_año.empty:
            # Devolver un mensaje si no hay datos para el año especificado
            raise HTTPException(status_code=404, detail={
                                "Mensaje": "No hay datos para el año especificado"})

        # Filtrar por recomendaciones positivas/neutrales
        df_recomendados = df_año[(df_año['recommend'] == True) & (
            df_año['sentiment_analysis'].isin([1, 2]))]

        if df_recomendados.empty:
            # Devolver un mensaje si no hay juegos recomendados para el año especificado
            raise HTTPException(status_code=404, detail={
                                "Mensaje": "No hay juegos recomendados para el año especificado"})
        # Contar las recomendaciones por juego
        conteo_recomendaciones = df_recomendados['item_name'].value_counts(
        ).reset_index()
        conteo_recomendaciones.columns = ['Juego', 'Recomendaciones']
        # Obtener el top 3 de juegos recomendados
        top3_recomendados = conteo_recomendaciones.head(3)
        # Crear la lista de retorno
        resultado = [{"Puesto " + str(i + 1): {"Juego": juego, "Recomendaciones": recomendaciones}}
                     for i, (juego, recomendaciones) in enumerate(top3_recomendados.values)]

        return resultado

    except Exception as e:
        # Devolver un mensaje de error en caso de cualquier otra excepción
        raise HTTPException(
            status_code=500, detail={"Mensaje": f"Error interno del servidor: {str(e)}"})


@app.get('/UsersNotRecommend/{anio}')
async def UsersNotRecommend(anio: int):
    try:
        # Obtener una muestra aleatoria del DataFrame
        df_sample = data_reviews.sample(frac=0.15, random_state=42)
        # Fusionar los DataFrames para obtener la información relevante
        df_merged = pd.merge(df_sample[['item_id', 'recommend', 'sentiment_analysis', 'año_posted']],
                             data_items[['item_id', 'item_name']],
                             on='item_id',
                             how='inner')
        # Filtrar por el año especificado
        df_año = df_merged[df_merged['año_posted'] == str(anio)]

        if df_año.empty:
            # Devolver un mensaje si no hay datos para el año especificado
            raise HTTPException(status_code=404, detail={
                                "Mensaje": "No hay datos para el año especificado"})

        # Filtrar por recomendaciones negativas
        df_no_recomendados = df_año[(df_año['recommend'] == False) & (
            df_año['sentiment_analysis'].isin([0]))]

        if df_no_recomendados.empty:
            # Devolver un mensaje si no hay juegos no recomendados para el año especificado
            raise HTTPException(status_code=404, detail={
                                "Mensaje": "No hay juegos no recomendados para el año especificado"})

        # Contar las no recomendaciones por juego
        conteo_no_recomendaciones = df_no_recomendados['item_name'].value_counts(
        ).reset_index()
        conteo_no_recomendaciones.columns = ['Juego', 'Recomendaciones']
        # Obtener el top 3 de juegos recomendados
        top3_no_recomendados = conteo_no_recomendaciones.head(3)
        # Crear la lista de retorno
        resultado = [{"Puesto " + str(i + 1): {"Juego": juego, "No lo recomiendan ": no_recomendaciones}}
                     for i, (juego, no_recomendaciones) in enumerate(top3_no_recomendados.values)]

        return resultado

    except Exception as e:
        # Devolver un mensaje de error en caso de cualquier otra excepción
        raise HTTPException(
            status_code=500, detail={"Mensaje": f"Error interno del servidor: {str(e)}"})


@app.get('/Sentiment_analysis/{anio}')
async def Sentiment_analysis(anio: int):
    try:
        # Fusionar los DataFrames para obtener la información relevante
        df_merged = pd.merge(data_reviews[['sentiment_analysis', 'item_id']],
                             data_steam[['id', 'release_date']],
                             left_on='item_id',
                             right_on='id',
                             how='inner')
        # Filtrar por el año especificado
        df_año = df_merged[(df_merged['release_date']) == anio]

        if df_año.empty:
            # Devolver un mensaje si no hay datos para el año especificado
            raise HTTPException(status_code=404, detail={
                                "Mensaje": "No hay datos para el año especificado"})
        # Contar la cantidad de registros por análisis de sentimiento
        conteo_sentimientos = df_año['sentiment_analysis'].value_counts(
        ).reset_index()
        conteo_sentimientos.columns = ['Sentimiento', 'Cantidad']
        # Mapear los códigos de sentimiento a etiquetas
        sentimiento_labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        conteo_sentimientos['Sentimiento'] = conteo_sentimientos['Sentimiento'].map(
            sentimiento_labels)
        # Crear el diccionario de retorno
        resultado = {row['Sentimiento']: row['Cantidad']
                     for _, row in conteo_sentimientos.iterrows()}

        return resultado

    except Exception as e:
        # Devolver un mensaje de error en caso de cualquier otra excepción
        raise HTTPException(
            status_code=500, detail={"Mensaje": f"Error interno del servidor: {str(e)}"})
