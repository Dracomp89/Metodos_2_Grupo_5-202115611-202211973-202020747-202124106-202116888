import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos
df = pd.read_csv('Rhodium.csv')
df['Diferencia'] = df["Intensity (mJy)"].diff()

# Limpieza inicial con una ventana de varianza para reducir ruido
varianza = 0.02
df_corregido = df[((df["Diferencia"] >= -varianza) & (df["Diferencia"] <= varianza))].copy()

# Parámetros iniciales
grado = 10
tolerancia = 0.2  # Tolerancia para considerar un punto como outlier
max_iteraciones = 10000
iteraciones = 0

# Proceso iterativo de ajuste y eliminación de outliers
while iteraciones < max_iteraciones:
    # Extraer los datos
    x = df_corregido["Wavelength (pm)"]
    y = df_corregido["Intensity (mJy)"]

    # Ajustar un polinomio al fondo
    coeficientes = np.polyfit(x, y, grado)
    polinomio = np.poly1d(coeficientes)

    # Calcular residuos
    ajuste_polinomico = polinomio(x)
    residuos = y - ajuste_polinomico
    df_corregido['Residuos'] = residuos

    # Identificar puntos lejanos al ajuste (outliers)
    max_residuo = abs(residuos).max()
    if max_residuo > tolerancia:
        # Eliminar el punto más lejano
        indice_outlier = abs(residuos).idxmax()
        df_corregido = df_corregido.drop(indice_outlier).reset_index(drop=True)
    else:
        # Si no hay outliers significativos, detener el ciclo
        break

    iteraciones += 1

# Gráfica final del ajuste
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Datos sin outliers', color='blue', alpha=0.6)
plt.plot(x, polinomio(x), label=f'Ajuste polinómico de grado {grado}', color='red', linestyle='--')
plt.axhline(0, color='black', linestyle=':')
plt.xlabel('Longitud de onda (pm)')
plt.ylabel('Intensidad (mJy)')
plt.title('Ajuste al fondo con eliminación iterativa de outliers')
plt.legend()
plt.show()
