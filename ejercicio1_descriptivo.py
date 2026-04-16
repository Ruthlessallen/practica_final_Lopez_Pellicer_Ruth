# Pimero importamos las librerías para poder trabajar con ellas
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sl
import scipy as sc
import matplotlib.pyplot as plt

# Pequeña configuración para que en ningún momento Python oculte columnas:
pd.set_option("display.max_columns", None)

def cargar_datos(ruta):
    # Carga de los datos.
    df = pd.read_csv(ruta)
    return df


def resumen_estructural(df):
    # Número de filas, usamos shape:
    filas = df.shape[0]
    # Número de columnas, usamos shape:
    columnas = df.shape[1]
    # Tamaño en memoria, usamos memory_usage, con parametro deep=true para que cuente cuánto ocupan las palabras y añadimos sum() para que nos dé la suma total:
    memoria = df.memory_usage(deep=True).sum()
    # Para que hacerlo mas interpretable, lo vamos a pasar a MB:
    memoria = round(memoria / (1024**2), 2)
    return filas, columnas, memoria

def tratamiento_nulos(df):
    # Tratamiento de nulos:
    # Para el tratammiento de nulos se ha decidido eliminarlos:
    df = df.dropna()
    return df


def estadistica_descriptiva(df):
    # B. ESTADÍSTICOS DESCRIPTIVOS DE VARIABLES NUMÉRICAS.
    # Mediana, con .median()
    mediana = df.median(numeric_only=True)
    # Moda, con .mode()
    moda = df.mode().round(2)
    # Desviacion tipica usando .std()
    desviacion = df.std(numeric_only=True)
    # Varianza, usando .var()
    varianza = df.var(numeric_only=True).round(2)
    # Minimo usando.min():
    minimos = df.min(numeric_only=True)
    # Máximo usando.max():
    maximos = df.max(numeric_only=True)
    # Percentiles usando .quantile(p):
    # Como pandas tiene problemas a la hora de usar numeric_only, y coge también booleanos
    # Vamos a generar una variable temporal con las columnas solo numericas usando select_dtypes
    col_num = df.select_dtypes(include=["number"])
    # P25:
    P25 = col_num.quantile(0.25).round(2)
    # P50
    P50 = col_num.quantile(0.50).round(2)
    # P75:
    P75 = col_num.quantile(0.75).round(2)

    # IQR de job_satisfaction_score, para eso calculamos P25 y P75 (o Q1 y Q3)
    Q1 = df["job_satisfaction_score"].quantile(0.25)
    Q3 = df["job_satisfaction_score"].quantile(0.75)
    # Hacemos la resta de Q3 -Q1:
    IQR = Q3 - Q1

    # Skewness con .skew()
    skewness = df["job_satisfaction_score"].skew()
    # Kurtosis con .kurt()
    kurtosis = df["job_satisfaction_score"].kurt()
    return mediana, moda, desviacion, varianza, minimos, maximos, P25, P50, P75, IQR, skewness, kurtosis

def crear_descriptivo(df):
    col_num = df.select_dtypes(include=["number"])
    # Para finalizar este modulo, he creado un archivo .csv con .describe() para tener todos los datos a mano:
    descripcion = df.describe(percentiles=[0.25, 0.5, 0.75]).T
    descripcion["varianza"] = df.var(numeric_only=True)
    descripcion["mediana"] = df.median(numeric_only=True)
    descripcion["moda"] = df.mode(numeric_only=True).iloc[0]
    descripcion["iqr"] = col_num.quantile(0.75) - col_num.quantile(0.25)
    descripcion["skewness"] = df.skew(numeric_only=True)
    descripcion["kurtosis"] = df.kurt(numeric_only=True)
    descripcion.to_csv("output/ej1_descriptivo.csv", float_format="%.2f")
    return descripcion

def distribucion_histograma(df):
    col_num = df.select_dtypes(include=["number"])
    # Vamos a crear los histogramas:
    col_num.hist(
        bins=30,  # Cuantas barras habrá en cada histograma.
        figsize=(20, 15),  # El tamaño de la imagen.
        layout=(5, 3),  # Layout de las graficas, 5 filas en 3 columnas.
        edgecolor="black",  # Ponemos un borde negro para que se vean mejor las barras.
    )
    # Ajustamos los márgenes para que el título de las gráficas no choquen.
    plt.tight_layout(pad=3.0)
    # Generamos el archivo:
    plt.savefig("output/ej1_histogramas.png")
    plt.close()
    return


def distribucion_boxplot(df):
    # Boxplots de la variable objetivo, segmentados por cada variable categórica.
    # Definimos las columnas con valores categóricos
    col_cat = df.select_dtypes(include=["object", "category", "string", "bool"]).columns
    # Empezamos con el grid donde van a ir las gráficas boxplot:
    fig, axes = plt.subplots(
        nrows=3, ncols=2, figsize=(12, 18)  # 3 filas  # 2 columnas
    )  # Un tamaño donde tendrá 12 de ancho y 18 de alto
    # Aplanamos para poder iterar mas adelante
    axes = axes.flatten()
    # Iteramos para que nos cree cada una de las gráficas:
    for i, col in enumerate(col_cat):
        # Dibujamos el boxplot:
        sns.boxplot(
            data=df,  # Datos cogidos de nuestro dataset
            x=col,  # Cogemos en el eje x la columna (de las categóricas)
            y="job_satisfaction_score",  # En el eje y, la columna objetivo
            ax=axes[i],
        )  # Lo ponemos en uno de los huecos que hemos aplanado antes con .flatten()
        # Ponemos título al boxplot
        axes[i].set_title(f"Satisfacción según {col}", fontsize=12)
    # Si existe un cuadro blanco que nos sobra, lo vamos a eliminar
    if len(col_cat) < len(axes):
        fig.delaxes(axes[len(col_cat)])
    plt.tight_layout()
    plt.savefig("output/ej1_boxplots_categoricos.png")
    plt.close()
    return


def tratamiento_outliers(df):
    col_num = df.select_dtypes(include=["number"])
    # Detección y tratamiento de outliers (método IQR)
    for col in col_num:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        inferior = q1 - 1.5 * iqr
        superior = q3 + 1.5 * iqr
        outliers = ((df[col] < inferior) | (df[col] > superior)).sum()
        df[col] = df[col].clip(lower=inferior, upper=superior)
    return df

def variable_categorica(df):
    col_cat = df.select_dtypes(include=["object", "category", "string", "bool"]).columns
    # Vamos a calcular la frecuencia absoluta y relativa de los valores en las columnas categóricas.
    for col in col_cat:
        val_absoluto = df[col].value_counts()
        # Al usar normalize= True, Pandas hace una división interna, toma cada suma
        # de cada columna y lo divide por el numero de filas, multiplicando por 100 nos da el %
        val_relativo = df[col].value_counts(normalize=True) * 100

    # Gráficos de cada variable categórica
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 18))
    axes = axes.flatten()
    for i, col in enumerate(col_cat):
        # Ordenamos las barras de mayor a menor
        orden = df[col].value_counts().index
        sns.countplot(data=df, x=col, ax=axes[i], order=orden)
        axes[i].set_title(f"Distribución de {col}", fontsize=14)
    if len(col_cat) < len(axes):
        fig.delaxes(axes[-1])
    plt.tight_layout()
    plt.savefig("output/ej1_analisis_categorico.png")
    plt.close()
    return


def correlacion(df):
    col_num = df.select_dtypes(include=["number"])
    # Mapa de calor:
    # Usamos la función corr con el método Pearson para determinar las correlaciones.
    correlaciones = col_num.corr(method="pearson")
    # Usamos la función de triángulo superior (triu: triangle up) para que no se duplique en el mapa.
    mask = np.triu(np.ones_like(correlaciones, dtype=bool))
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlaciones,
        mask=mask,  # Aplicamos la máscara
        annot=True,  # Escribe los números dentro de los cuadros
        fmt=".2f",  # Formato de 2 decimales
        cmap="coolwarm",  # Rojo para positivo, Azul para negativo
        vmin=-1,
        vmax=1,  # Forzamos la escala de -1 a 1
        center=0,  # El blanco será el 0
        cbar_kws={"shrink": 0.8},
    )  # Ajustamos el tamaño de la barra de color
    plt.title("Matriz de Correlación de Pearson", fontsize=16, pad=20)
    plt.tight_layout()
    plt.savefig("output/ej1_heatmap_correlacion.png", bbox_inches="tight")
    plt.close()

    # Detección de posible multicolinealidad entre predictoras (pares con |r| > 0,9)
    # Vamos a crear una matriz de correlación solo de las predictoras, para ello vamos a excluir la misma columna objetivo.
    df_predic = df.select_dtypes(include=["number"]).drop(
        columns=["job_satisfaction_score"]
    )
    correlacion = df_predic.corr().abs()  # abs() convierte negativos a positivos
    # Hacemos lo mismo que antes para que no se duplique la matriz
    upper = correlacion.where(np.triu(np.ones(correlacion.shape), k=1).astype(bool))

    # Buscamos las colineales, si existe, se guardan en la variable
    colineales = [column for column in upper.columns if any(upper[column] > 0.9)]

    pares = False

    for row in upper.index:
        for col in upper.columns:
            val = upper.loc[
                row, col
            ]  # Mira los valores exactos de la columna y la fila
            if val > 0.9:
                pares = True
    return pares

def crear_dataset_limpio(df):
    # Para finalizar, voy a exportar el dataset con la limpieza de outliers en un csv nuevo:
    df.to_csv("data/social_media_clean.csv", index=False)
    return df

def main():
    
    
    # 1. Cargar datos
    df = cargar_datos("data/social_media_vs_productivity.csv")
    
    # 2. Resumen estructural
    filas, columnas, memoria = resumen_estructural(df)

    # 3. Tratamiento nulos
    df = tratamiento_nulos(df)
    
    # 4. Estadística descriptiva
    estadisticas = estadistica_descriptiva(df)
    
    # 5. Generando output CSV y gráficos
    crear_descriptivo(df)
    distribucion_histograma(df)
    distribucion_boxplot(df)

    #6. Tratamiento de outliers
    df = tratamiento_outliers(df)

    #7. Variables categóricas 
    variable_categorica(df)

    #8. Correlacion
    pares = correlacion(df)

    #9. Creación de dataset limpio    
    df = crear_dataset_limpio(df)
    
 

if __name__ == "__main__":
    main()
