# Proyecto Machine Learning Operations (MLOps) - Steam

## INTRODUCCION

Bienvenidos al proyecto MLOps de Steam En esta oportunidad, nos pondremos como un Ingeniero MLOps en Steam, una renombrada plataforma de juegos a nivel mundial.  
* Objetivos: 
* Revisar y limpiar los datasets y preparar los modelos de aprendizaje automático, y de esa manera optimizar el rendimiento de la API y el entrenamiento del modelo.
* Disponibilizar los datos de la empresa usando el framework FastAPI, con 05 consultas.
* Creación de un sistema de recomendación de videojuegos mediante el uso de técnicas de aprendizaje automático.
* El MVP tiene que ser una API que pueda ser consumida segun los criterios de API REST 

## DATA - STEAM

Trabajamos con tres archivos JSON que contienen datos sobre los juegos en la plataforma Steam. Cada archivo brinda informacion diferente:

### output_steam_games.json: <br>

Brinda la informacion de los juegos con las siguientes variables:
* publisher = Empresa publicadora del contenido
* genres = Genero del contenido (anidado)
* app_name = Nombre del contenido
* title = Titulo del contenido
* url = URL de publicación del contenido
* release_date = Fecha de lanzamiento
* tags = etiquetas de contenido (anidado)
* reviews_url = Reviews de contenido
* specs = Especificaciones (anidado)
* price = Precio del contenido
* early_access = acceso temprano
* id = identificador unico de contenido
* developer = Desarrollador

### australian_user_reviews.json: <br>

Brinda informacion sobre la Reseña de juegos hecha por los usuarios con las siguientes variables:
* user_id : identificador unico de usuario
* user_url : URL perfil del usuario
* reviews : Review de usuario en formato Json

### australian_users_items.json:<br>

Brinda informacion de los juegos por cada usuario con las siguientes variables:
* user_id : identificador unico de usuario
* user_url : URL perfil del usuario
* items : Items de usuario en formato Json

## DESARROLLO DEL PROYECTO 

### Ingeniería de Datos

- **Transformación Y Limpieza de Datos:** El trabajo a realizar se centrara en la limpieza de valores nulos y duplicados, trabajar con las variables anidadas y borrar las columnas que no son importantes para el analisis, para optimizar el rendimiento de la API y el entrenamiento del modelo. El ETL, se realizara a los tres conjuntos de datos que fueron proporcionados <br>

- **ETL OUTPUT :** ["ETL Output"](1_ETL_OUTPUT.ipynb) <br>

El dataframe limpio de output steam games (Informacion de juegos) contiene las siguientes columnas:
* app_name : Nombre del juego 
* id : Id de juego
* developer : desarrollador
* price_numeric : Precio de lanzamiento
* release_year : Año de lanzamiento
* main_genre : Genero principal

- **ETL ITEMS :** ["ETL Users Items"](1_ETL_ITEMS.ipynb) <br>

El dataframe limpio de Items (Informacion de juegos por usuario) contiene las siguientes columnas:
* user_id : Id de usuario
* items_count : Cantidad de juegos por usuario
* item_id : Id del juego
* item_name : Nombre de juego
* playtime_forever : Tiempo jugado en minutos
* playtime_hour : Timpo juego en horas

- **ETL REVIEWS :** ["ETL Users Reviews"](1_ETL_REVIEWS.ipynb) <br>

El dataframe limpio de Reviews (Reseñas de juegos) contiene las siguientes columnas:
* user_id : Id de usuario
* item_id : Id de juego
* recommend : Si recomienda el juego
* year_posted : Año de recomendacion
* sentiment_analysis : Análisis de Sentimiento : Se solicito crear una nueva columna, 'sentiment_analysis', aplicando análisis de sentimiento mediante Procesamiento de Lenguaje Natural (NLP) a las reseñas de usuarios. La escala que se utilizo fue: '0' negativos, '1' neutrales y '2' positivos.









