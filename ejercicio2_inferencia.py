from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np

# Introducimos el dataset limpio de outliers:
df = pd.read_csv("data/social_media_clean.csv")

# A. PREPROCESAMIENTO.
# 1. Aplicación de transformaciones.
# Primero eliminamos la columna que sabemos que tiene colinealidad.
col_eliminada = "perceived_productivity_score"
df = df.drop(columns=col_eliminada, errors="ignore")
# Codificación de las variables categóricas:
df = pd.get_dummies(
    df,
    columns=["gender", "job_type", "social_platform_preference"],
    drop_first=True,  # Usamos drop_first para asegurar que si no es ninguna de las demás variables, por descarte es la que queda.
)
# Además, vamos a tener que transformar aquellas que son booleanas:
df["uses_focus_apps"] = df["uses_focus_apps"].astype(int)
df["has_digital_wellbeing_enabled"] = df["has_digital_wellbeing_enabled"].astype(int)

# Vamos a separar los valores predictores (x) y valores objetivo (y)
X = df.drop(columns=["job_satisfaction_score"])
y = df["job_satisfaction_score"]
# 2. Dividimos: train (80%) y test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#  Ajustamos el escalador solo con los datos train y luego transformamos ambos
scaler = StandardScaler()
# Escalamos las variables numéricas
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Imprimimos:
print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de X_test: {X_test.shape}")

# B. MODELO A. REGRESIÓN LINEAL.
# 1. Entrenamiento.
lr_model = LinearRegression()
# Entrenamos el modelo con los valores que hemos separado (train)
lr_model.fit(X_train_scaled, y_train)
# Seguimos para ver si el entrenamiento ha funcionado realizando el test
y_pred_lr = lr_model.predict(X_test_scaled)

# Calculamos el R2
r2 = r2_score(y_test, y_pred_lr)
print(f"Coeficiente de determinación (R2): {r2:.4f}")

# Creamos la gráfica
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--r", linewidth=2)
plt.xlabel("Valores Reales (Satisfacción)")
plt.ylabel("Predicciones del Modelo")
plt.title(f"Regresión Lineal: Real vs Predicho (R² = {r2:.2f})")
plt.show()
plt.close()

print(f"Intercepto (Base): {lr_model.intercept_:.4f}")


# 2. Evaluación sobre el test
# MAE:
mae = mean_absolute_error(y_test, y_pred_lr)
# MSE (Ppara calcular RMSE):
mse = mean_squared_error(y_test, y_pred_lr)
# RMSE:
rmse = np.sqrt(mse)
# No calculamos R2 ahora porque lo habíamos calculado anteriormente
# Imprimimos los resultados
print(f"MAE (Error Absoluto Medio): {mae:.4f}")
print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
print(f"R² (Coeficiente de Determinación): {r2:.4f}")

with open("output/ej2_metricas_regresion.txt", "w", encoding="utf-8") as f:
    f.write("MÉTRICAS DEL MODELO DE REGRESIÓN LINEAL\n")
    f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    f.write(f"Coeficiente de Determinación (R2): {r2:.4f}\n")

# 3. Gráfica de residuos.
# Primero vamos a calcular los residuos:
residuos = y_test - y_pred_lr
#  Creamos el gráfico:
fig = plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred_lr, y=residuos, alpha=0.5)
# Dibujamos la linea en el 0
plt.axhline(y=0, color="r", linestyle="--")
plt.xlabel("Valores predichos de satisfacción")
plt.ylabel("Residuos (Error en la predicción)")
plt.title("Gráfico de residuos.")
plt.tight_layout()
fig.savefig("output/ej2_residuos.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# Vamos a ver cuales son aquellas categorías que influyen en nuestra score de satisfacción en el trabajo:
importancia = pd.DataFrame({"Variable": X.columns, "Coeficiente": lr_model.coef_})

# Lo ordenamos de mayor a menor influencia
importancia = importancia.sort_values(by="Coeficiente", ascending=False)
print(f"Variables más influyentes en la Satisfacción:\n {importancia}")
