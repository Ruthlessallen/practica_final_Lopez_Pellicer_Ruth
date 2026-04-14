# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo

A. RESUMEN ESTRUCTURAL.

1.  Filas: 30000
    Columnas: 19
    Tamaño en memoria: 8.02 MB.
2.  Tipos de datos por columna:
    Tenemos 9 columnas con datos float, 4 columnas con datos int, 3 columnas con datos str y 2 con datos bool.
3.  Nulos y tratamiento. Respondido en la pregunta 1.4.

B. ESTADÍSTICOS DESCRIPTIVOS DE VARIABLES NUMÉRICAS.

1. He creado un csv con todos las estadísticas de las columnas en ej1_descriptivo.csv, incluyendo aquellas que no están en .describe() como la varianza, mediana, moda, iqr, skewness y kurtosis.
2. Mi variable objetivo job_satisfaction_score tiene un IQR de 2.88, lo que indica que la mayoría de personas (el 50%) tienen una satisfacción laboral de entre 3.53 (P25) y 6.41(P75). Bastante baja en general, incluso suspendida.
3. Skewness de job_satisfaction_score: 0.01. Prácticamente simétrico, no hay outliers e indica una característica de la distribución normal. Kurtosis: -0.45. Esto indica que hay muchos valores dispersos en los datos y no se concentran todos en la media, lo que nos muestra una kurtosis platicúrtica.

C. DISTRIBUCIONES

1. Explicadas en la pregunta 1.2.
2. Según la variable objetivo, lo que podemos ver es que todas las vriables tienen la mediana en el centro, lo que indica una distribución normal, además, al estar relacionadas con un método de evaluación de 0 a 10 puntos, no se presencia desvariaciones exageradas. A nivel milimétrico vemos que la satisfacción sube en el tipo de trabajo IT, baja en el uso de RRSS como Facebook e Instagram.
3. Explicado en la pregunta 1.2.

D. VARIABLES CATEGÓRICAS.

1. Vemos una frecuencia muy similar en columnas como job_type, social_platform_preference. Luego en la columna gender vemos que hay mucha paridad entre hombres y mujeres pero gran disparidad con Other (4%), respecto a uses_focus_apps, solo el 30% usa aplicaciones para enfocarse respecto al 70% que no lo hace y por último la columna has_digital_wellbeing_enabled menos del 25% usa bienestar digital (una herramienta nativa en los dispositivos actuales para poder limitar el uso de aplicaciones, redes sociales, uso de solo x horas al día...) mientras que la amplia mayoría (75%) no lo usa.
2. Las distribuciones de todas las categorías se ven visiblemente en las gráficas aportadas (binomialidad en columnas booleanas y uniformidad en las demás a excepción de gender).
3. Como tal, las columnas has_digital_wellbeing_enabled y uses_focus_apps tienen una dominancia de personas que no usan ninguna de estas dos funcionalidades, con lo que a la hora de analizar las demás columnas vamos a tener que asumir que muy pocas personas usan herramientas para mejorar su relación con el mundo digital, así que no vamos a poder comparar de manera equitativa la diferencia entre no usarlas y usarlas. Sin embargo, podría ser un nicho de estudio muy interesante: ¿las personas que usan este tipo de herramientas rinden mejor, tienen menos estrés o incluso duermen mejor?

E. CORRELACIONES.

1. Mapa generado como ej1_heatmap_correlacion.png
2. Respondido en la pregunta 1.3.
3. Las columnas con mayor colinealidad son perceived_productivity_score y actual_productivity_score con un r = 0.9013. Para un estudio, cualquiera de las dos columnas nos valdría para entrenar un modelo, pero no ambas.

##

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset proviene de Kaggle (https://www.kaggle.com/datasets/mahdimashayekhi/social-media-vs-productivity). Tenemos como campos categóricos las columnas: gender, job_type, social_platform_preference, uses_focus_apps y has_digital_wellbeing_enabled. Como columnas numéricas continuas tenemos: daily_social_media_time, work_hours_per_day, perceived_productivity_score, actual_productivity_score, sleep_hours, screen_time_before_sleep, weekly_offline_hours y job_satisfaction_score. El target es job_satisfaction_score, ya que se puede intentar predecir la satisfacción laboral y si los demás factores (estrés, horas dormidas, uso de redes sociales...) influyen sobre ésta.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> age: distribución uniforme. Sin outliers.
> daily_social_media_time: distribución asimétrica positivam, estirada hacia la derecha debido a la presencia de 348 ouliers.
> number_of_notifications: distribución normal, con presencia de 261 outliers.
> work_hours_per_day: distribución normal, con presencia de 97 outliers.
> perceived_productivity_score: distribución casi uniforme con pico en el centro. 0 outliers.
> actual_productivity_score: distribución casi normal con un gran pico en el centro. 0 outliers.
> stress_level: distribución uniforme. 0 outliers.
> sleep_hours: distribución normal con gran pico en el centro. 0 outliers.
> screen_time_before_sleep: distribución asimétrica ligeramente positiva con gran pico en el centro y al principio de la gráfica con 198 detectados.
> breaks_during_work: distribución uniforme. 0 outliers.
> coffee_consumption_per_day: distribución asimétrica positiva detectando 127 outliers.
> days_feeling_burnout_per_month: distribución uniforme con picos al principio y final de la gráfica. 0 outliers.
> weekly_offline_hours: distribución asimétrica ligeramente positiva con pico en al principio de la gráfica. Presencia de 116 outliers.
> job_satisfaction_score: distribución normal con pico en el centro. 0 outliers.
> En este caso, respecto a los outliers se ha aplicado la técnica de windorización, limitando los valores extremos a los valores superior e inferior de IQR. Entendemos que estos outliers son casos muy extremos y exagerados (como el uso de RRSS de más de 15h por ejemplo, o que haya gente que no use el movil/ordenador mas de 30h o 40h...)

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> Solo hay dos variables que tienen una fuerte correlación con la variable objetivo: perceived_prouctivity_score con una correlcion del 0.79 puntos y actual_productivity_score de 0.81 puntos, lo que indica una gran correlación con la satisfacción laboral con la productividad percibida y la productividad actual: a mayor productividad, mayor satisfacción. A parte de esto, las únicas otras variables que tienen correlación son actual_productivity_Score con perceived_productivity_score, esto nos muestra que la productividad percibida tiene la misma tendencia que la productividad real.

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

Aunque la mayoria de las columnas no tienen nulos, hay 6 que sí.

    Columnas con % de nulos:
    daily_social_media_time: 9.22%
    actual_productivity_score: 7.88%
    stress_level: 6.35%
    sleep_hours: 8.66%
    screen_time_before_sleep: 7.37%
    job_satisfaction_score: 9.10%
    Y tenemos solo 17.074 filas sin ningún nulo. En este caso, al ser un dataset grande y pocos nulos (menos del 10% en 6 columnas), vamos a daily_social_media_time que tiene un skewness muy alto (1.25), lo que nos da una pista de que usar la media con esta columna nos va a dar un número sesgado.
    En las columnas que nos interesan, los nulos se van a rellenar con los siguientes datos:
    daily_social_media_time: 3.03
    actual_productivity_score: 4.95
    stress_level: 5.51
    sleep_hours: 6.50
    screen_time_before_sleep: 1.02
    job_satisfaction_score: 4.96
    Nulos después de rellenar con medias: 0.

---

## Ejercicio 2 — Inferencia con Scikit-Learn

---

A. PREPROCESAMIENTO

1. He eliminado la columna de perceived_productivity_score por tener multicolinealidad con la actual_productivity_score, he codificado las variables categoricas con get_dummies usando drop_first=True para evitar redundancias, además con las categóricas booleanas las he pasado a 0 y 1(astype(int)), y escalado con StandardScaler para evitar que el modelo piense que es más importante una variable que otra (por ejemplo la cantidad de notificaciones puede ser muy superior a tener una determinada edad). Además, he escalado después de la división de train para que no se filtre la información antes de entrenar el modelo, que lo hará a ciegas.
2. He aplicado la división con train_est_split con un random_state de 42. En total 24000 filas para entrenar (80%) y 6000 para testear (20%).

B. MODELO B - REGRESION LINEAL

1. He entrenado el modelo usando X_train e X_test, he generado y_pred_lr (los datos predichos) en base a X_test_scaled.
2. Respondido en la pregunta 2.1.
3. Gráfico generado como ej2_residuos.png. Al observar el gráfico lo que vemos a primera vista es una nube de puntos alrededor de la linea roja, esto nos confirma que hay homocedastividadd: el modelo acierta prediciendo tanto a gente muy satisfecha con su trabajo como a gente que no lo es, y a la vez se equivoca de manera equilibrada. Lo segundo que vemos es una linea recta vertical en el centro de la gráfica: hay ciertas personas que el modelo no ha podido predecir, son aquellas personas que a pesar de tener buenas condiciones en sus categorias puntuan a la baja su satisfaccion en el trabajo y viceversa. Lo tercero es la diagonal: y es que hay que tener en cuenta que antes teniamos en el dataset casi un 10% de nulos en la variable de job_satisfaccion_score, al rellenarlos con su media (4.96), ese 10% va a generar una linea diagonal perfecta (si el modelo por ejemplo dice que esa persona tendría una satisfacción de 2 y la realidad es que tiene 4.96, se va a ir 2.96 puntos para arriba, y así con todos los demás). En otro contexto, se podría usar el modelo para "adivinar" estos datos nulos.
4. Como lo comentado en la pregunta 2.1, el modelo es bueno, tiene un desempeño notable a la hora de predecir, teniendo en cuenta que la satisfacción laboral puede ser determinada por otras cuestiones que no existen en el dataset (salario, relación con compañeros/jefes, ambiente laboral, cantidad de vacaciones al año...).
   No hay underfitting: r2 por encima del 0.5, lo que indica que es capaz de ver la estructura y tendencia de los datos. Tampoco hay overfitting: el modelo rinde bien con el 20% de los datos (r2= 0.65). Aquí no tenemos varias variables, tenemos una que es la que más nos dice de la satisfacción en el trabajo: actual_productivity_score, con 1.63 puntos. Eso quiere decir que a más producción, mayor satisfacción. Tenemos otras variables mucho más pequeñas detrás de esta variable: el uso de Telegram (0.021) y trabajar en IT (0.018), lo cual no cambia mucho a la hora de tener mayor satisfacción laboral.

C. CONCLUSIONES.
El modelo funciona correctamente. A pesar de que la mayoría de las variables no están correlacionadas con la satisfacción en el trabajo, también nos muestra información muy valiosa: no importa el genero, la edad, el estrés, el tipo de trabajo, el uso de redes sociales o el sueño entre otras variables para sentir satisfacción en el trabajo, lo que realmente importa es la productividad del trabajador para sentirse satisfecho.
Es importante también señalar que esto solo nos lo explica al 65%, el 35% restante podría tener más peso: un buen salario, un buen ambiente en el trabajo puede hacer que la satisfacción mejore en gran medida (o no), lo cual seguramente sería una mejora significativa en el dataset. Em este caso, nuestro intercepto es 4.96, un 5 raspado en nuestra score: si todo lo demás fallara, seguiriamos asumiendo una nota de 5 en nuestra satisfacción laboral, y por cada punto que añadamos en nuestra productividad, esa nota subirá un 1.63.
Sin duda, y tras analizar los errores del modelo y ver que funciona correctamente, la información más util es la de la importancia de las variables, pues con ellas podemos trabajar: si tuvieramos un departamento de RRHH con esta información, podríamos enfocar los flujos de trabajo con la productividad, incluso añadiendo recompensas por dicha productividad, ya que a mayor ssatisfacción en el trabajo, menor rotación de empleados, mayor compromiso con la empresa, el proyecto...

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

El calculo de R2 es de 0.65, no es 100% perfecto pero se acerca mucho. Es mejor que usar solo la media sin duda.
Respecto al MAE: nos alejamos casi 1 punto respecto al valor predicho, además, al ser el IQR de nuestra variable un 2.88 estamos dentro de ese porcentaje, con lo que podemos sentirnos tranquilos con el error.
Respecto al RMSE: 1.19, lo que se considera que los errores son bastante consistentes, sobre todo respecto al MAE, siempre el modelo se va a equivocar alrededor de 1 punto.
Por lo general, nuestro modelo predice muy bien nuestra variable objetivo: explica el 65% de lo que a una persona le hace estar satisfecha en el trabajo, en promedio se queda a menos de 1 punto de la realidad, y no se sugiere que el modelo tenga grandes errores con datos atípicos grandes que desvien las predicciones.

---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---

Tenemos un intercepto de 4.86 en nuestra regresión lineal, lo que significa que si todas nuestras features (b1, b2, b3) fueran 0, el intercepto empezaría con 4.86. Nuestros pesos son:
b1: 2.06
b2: -1.11
b3: 0.43
Cada vez que se añada 1 de b1, nuestro intercepto subirá 2.06 ( y así con todas nuestras variables).
A la hora de entrenar nuestro modelo, vemos que tiene un error R2 de 0.68, que es hasta un 68% mejor que usar solo la media de nuestros datos. También tenemos un MAE de 1.16, el modelo se aleja un poco más de un punto respecto a lo predicho y un RMSE de 1.49, los errores son más o menos consistentes.
La regresión se realiza correctamente en una linea perfectamente diagonal, lo que nos indica que lo predicho con lo real es correlacional.

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

Para saber el intercepto (b), primero calculamos:
X: la matriz de 4 columnas por 200 filas
XT es igual a la transposición de la matriz X: 200 columnas y 4 filas
^-1, al elevarlo a -1 nos devuelve las métricas realistas y proporcionales.
y es la columna de datos que queremos investigar, aprender (con train) y predecir (con test).
Entonces (XTX)^-1: multiplicamos la matriz transpuesta por la matriz original, esto nos da una matriz de 4*4 elevada al cuadrado (número exageradamente grandes), al elevarlo a -1 nos devuelve numeros normales. Nos da una matriz con las pistas de nuestros datos que más tarde usaremos para saber sus pesos.
XT*y: multiplicamos la matriz transpuesta por la columna de a investigar, nos dará 4 elementos.
Si multiplicamos la matriz final con estos 4 elementos, lo que nos dará como resultado es el peso de cada una de las variables con respecto a nuestro intercepto.
Es necesario tener una columna de 1 ya que la formula trabaja con pares y b0 no tiene ningún coeficiente con el que multiplicarse (como sí lo hacen b1, b2 y b3)

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
| --------- | ---------- | -------------- |
| β₀        | 5.0        | 4.86499486     |
| β₁        | 2.0        | 2.0636177      |
| β₂        | -1.0       | -1.11703839    |
| β₃        | 0.5        | 0.43851694     |

> bo = se aleja por -0.14 puntos.
> b1 = se aleja por 0.06 puntos.
> b2 = se aleja por -0.11 puntos
> b3 = se aleja por -0.07 puntos.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> MAE = 1.1665. Se aleja por -0.04 puntos
> RMSE = 1.4612. Se aleja por -0.04 puntos
> R² = 0.6897. Se aleja por 0.12 puntos
> Se aproximan a los de referencia.

---

## Ejercicio 4 — Series Temporales

---

La serie comienza el 01/01/2018 y termina el 31/12/2023. Son 6 años, se analiza día a día, es por eso que tenemos 2191 observaciones. La media se situa en 106,28 puntos con una desviación de 32.81 puntos y en cualquier caso tiene un mínimo histórico de 44.33 puntos y un máximo histórico de 171.78 puntos.
La evolución de la serie presenta una tendencia alcista, con un ciclo entre los años 2020 y 2022 que baja la tendencia pero no la rompe. Además, se aprecia una clara estacionalidad anual, con bajadas de unos 30 puntos y subidas de 30 puntos cada año. A lo largo de todos esos años, la serie temporal ha aumentado más de 100 puntos lo que nos confirma la tendencia positiva.
El residuo es consistente, se aprecia una nube en la gráfica de descomposición que abarca de -10 a 10 puntos sin outliers, casi sin skewness ni kurtosis y una media muy pequeña (0.12 puntos), además el test de normalidad indica también la consistencia de estos residuos, p-valor es igual a 0 lo que muestra que los datos son estacionarios (estables en media y varianza), confirmando que el residuo no arrastra tendencias del pasado. Además, viendo el histograma del ruido, que es una perfecta campana de Gauss, que la mayoria de las veces no existe ese ruido y cuando lo hay, va de -10 a 10 (como vimos anteriormente en la nube). Por último, el ACF y PACF nos muestra practicamente los mismos resultados: el azar es puro, no hay ninguna correlación en el tiempo respecto a los datos, lo que refuerza que el ruido es aleatorio e independiente.

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Presenta una tendencia lineal ascendente pasando de un valor de 60 en 2018 hasta 160 en 2024 (100 puntos en 6 años, o aproximadamente 16.6 puntos cada año). Es una tendencia suave y constante, mostrando crecimiento sostenido a lo largo de la serie que no se ve alterado por los ciclos estacionales.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> Existe una estacionalidad anual (365.25 días). Existe una amplitud aproximada de 30 puntos (sube 30 puntos y luego vuelve a bajar 30 puntos aproximadamente).

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Se aprecia un ligero descenso desde 2020 a 2022, es decir, la tendencia no es una linea perfectamente recta, en esos dos años la tendencia para de subir a la misma cadencia, a partir de 2022 aproximadamente vuelve a tener una tendencia ascendente de nuevo. Si dibujaramos una linea recta entre el primer año y el último podriamos ver perfectamente donde aparece el ciclo y donde finaliza.
> La tendencia es un movimiento con una dirección persistente a largo plazo, mientras que el ciclo se representa en las gráficas como un oscilamiento de varios años alrededor de esa misma dirección, además, es algo temporal, y suele quedar por encima o por debajo de nuestra linea de tendencia.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Media del residuo: 0.1271
> Desviación típica: 3.2220
> P-valor test normalidad (JB): 0.5766
> El residuo se ajusta al ruido ideal ya que la media se acerca mucho a 0, indica que el modelo no tiene un sesgo sistemático. La desviación típica es muestra consistencia en el residuo de 3.22 puntos. Además, el test de normalidad se aleja bastante de 0, son residuos típicos producidos por el azar.

---

_Fin del documento de respuestas_
