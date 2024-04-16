import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from fastapi import FastAPI
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Importamos los datos que se encuentran en formato parquet para dataframes
df_BestDeveloper = pd.read_parquet("Datasets/df_BestDeveloper.parquet")
df_Developer = pd.read_parquet("Datasets/df_Developer.parquet")
df_DeveloperReviews = pd.read_parquet("Datasets/df_DeveloperReviews.parquet")
df_UserData = pd.read_parquet("Datasets/df_UserData.parquet")
df_UsersForGenre = pd.read_parquet("Datasets/df_UsersForGenre.parquet")


# Primera Funcion : developer

@app.get("/developer")

def developer(desarrollador: str):
    """
    Ingrese desarrollador y devuelve un Diccionario con años como clave y Cantidad de Items y Porcentaje de Contenido Free
    """
    if type(desarrollador) != str:
        return {"Error": "Debes colocar un desarrollador de tipo str, por ejemplo: 'Valve'"}
    if len(desarrollador) == 0:
        return {"Error": "Debes colocar un desarrollador en tipo String"}
    df_desarrollador = df_Developer[df_Developer["developer"] == desarrollador]
    df_desarrollador_subset = df_desarrollador[["release_year", "cantidad_id", "contenido free"]]
    df_desarrollador_subset["contenido_free_porcentaje"] = df_desarrollador_subset["contenido free"].round(2)
    result_dict = {}
    for year, data in df_desarrollador_subset.groupby("release_year"):
        result_dict[year] = {str(year): {'Cantidad de Items': cantidad, 'Contenido Free': porcentaje} for cantidad, porcentaje in data[["cantidad_id", "contenido_free_porcentaje"]].values}
    reformatted_result = {year: values[str(year)] for year, values in result_dict.items()}
    return reformatted_result

# Segunda Funcion : userdata

@app.get("/userdata")

def userdata(user_id: str):
    """
    Ingrese id de usuario y devuelve la cantidad de dinero gastado, el porcentaje de 
    recomendación y cantidad de juegos.
    """
    user_data = df_UserData[df_UserData['user_id'] == user_id]
    
    if user_data.empty:
        return {"Error": f"No se encontraron datos para el user_id '{user_id}'"}
    
    price_numeric_total = user_data.iloc[0]['price_numeric']
    recommend_percentage = (user_data['recommend'].sum() / len(user_data)) * 100
    items_count = user_data.iloc[0]['items_count']
    result = {
        "user_id": user_id,
        "price_numeric": f"{price_numeric_total} USD",
        "% de recomendación": f"{recommend_percentage}%",
        "items_count": items_count
    }
    
    return result

# Tercera Funcion : UserForGenre

@app.get("/UserForGenre")

def UserForGenre( genero : str ):
    """
    Ingrese genero y devuelve el usuario on más horas jugadas  
    y una lista de la acumulación de horas jugadas por año.
    """
    genero_1 = df_UsersForGenre[df_UsersForGenre["main_genre"]== genero]
    user_max = genero_1.loc[genero_1["playtime_hour"].idxmax()]["user_id"]
    horas_x_año = genero_1.groupby(["year_posted"])["playtime_hour"].sum().reset_index()
    horas_lista = horas_x_año.rename(columns={'year_posted': 'Año', 'playtime_hour': 'Horas'}).to_dict(orient="records")
    result_ug = {
        f"Usuario con más horas jugadas para Género: {genero}": user_max,
        "Horas jugadas": horas_lista
    }
    return result_ug

# Cuarta Funcion : best_developer_year

@app.get("/best_developer_year")

def best_developer_year(año: int):
    """
    Ingrese año y devuelve el top 3 de desarrolladores con juegos MAS 
    recomendados por usuarios.
    """
    if type(año) != int:
        return {"Debes colocar el año en entero, Ejemplo:2012"}
    if año < df_BestDeveloper["year_posted"].min() or año > df_BestDeveloper["year_posted"].max():
        return {"Año no encontrado "}
    df_año = df_BestDeveloper[df_BestDeveloper["year_posted"] == año]
    df_ordenado_recomendacion = df_año.sort_values(by="recommendation_count", ascending=False)
    top_3_developers = df_ordenado_recomendacion.head(3)[["developer", "recommendation_count"]]
    result_bd = [{"Puesto {}: ".format(i+1) + top_3_developers.iloc[i]['developer'] for i in range(len(top_3_developers))}]
    return result_bd

# Quinta Funcion : developer_reviews_analysis

@app.get("/developer_reviews_analysis")

def developer_reviews_analysis(desarrolladora: str):
    """
    Ingrese desarrolladora y devuelve un diccionario con el nombre de la desarrolladora como llave y una lista 
    con la cantidad total de registros de reseñas de usuarios que se encuentren categorizados con 
    un análisis de sentimiento como valor.
    """
    if type(desarrolladora) != str:
        return "Debes colocar un developer de tipo str, EJ:'07th Expansion'"
    if len(desarrolladora) == 0:
        return "Debes colocar un developer en tipo String"
    df_developer = df_DeveloperReviews[df_DeveloperReviews["developer"] == desarrolladora]
    sentiment_counts = df_developer.groupby("sentiment_analysis")["sentiment_analysis_count"].sum().to_dict()
    sentiment_dicc = {0: "Negativo", 2: "Positivo"}
    result_dr = {desarrolladora: [f"{sentiment_dicc[key]} = {value}" for key, value in sentiment_counts.items() if key in sentiment_dicc]}
    return result_dr

# Sexta funcion: Sistema de recomendacion de juegos

modelo_recomendacion = pd.read_csv("Datasets/modelo_reco.csv")

def recomendacion_juego(id_juego):
    try:
        id_juego = int(id_juego)
        juego_seleccionado = modelo_recomendacion[modelo_recomendacion["id"] == id_juego]
        if juego_seleccionado.empty:
            return {"error": f"El juego con el ID '{id_juego}' no se encuentra."}
        indice_juego = juego_seleccionado.index[0]
        muestra = 3000
        df_muestra = modelo_recomendacion.sample(n=muestra, random_state=50)
        juego_features = modelo_recomendacion.iloc[indice_juego, 3:]
        muestra_features = df_muestra.iloc[:, 3:]
        similitud = cosine_similarity([juego_features], muestra_features)[0]
        recomendaciones = sorted(enumerate(similitud), key=lambda x: x[1], reverse=True)[:5]
        recomendaciones_indices = [i[0] for i in recomendaciones]
        recomendaciones_names = df_muestra["app_name"].iloc[recomendaciones_indices].tolist()

        return {"Juegos_similares": recomendaciones_names}

    except ValueError:
        return {"error": "El ID del juego debe ser un número entero válido."}


@app.get("/Recomendacion_Juego")
def obtener_recomendaciones(juego_id: int):
    try:
        recomendaciones = recomendacion_juego(juego_id)
        return {"recomendaciones": recomendaciones}
    except Exception as e:
        return {"error": f"Error en la recomendación: {str(e)}"}