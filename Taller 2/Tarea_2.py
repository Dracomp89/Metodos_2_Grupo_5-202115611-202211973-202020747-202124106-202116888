# -*- coding: utf-8 -*-
"""Tarea 2.py
"""
# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft

import pandas as pd



# 3. FILTROS -----------------------------
## 3a. Filtro gaussiano

# Cargar y leer el archivo de datos
file_path = "list_aavso-arssn_daily.txt"

with open(file_path, "r") as file:
    lines = file.readlines()

# Extraer última columna: SSN (Sunspot Number)
ssn_values = []
for line in lines:
    parts = line.split()  # Divide la línea en columnas
    if len(parts) >= 4:  # Asegurar que tiene al menos 4 columnas
        try:
            ssn_values.append(float(parts[3]))  # Tomar la última columna
        except ValueError:
            continue  # Saltar líneas con datos corruptos

data = np.array(ssn_values)

# Muestras
N = len(data)

# Crear el eje de frecuencias -> Nyquist
freqs = np.fft.fftfreq(N)

# Valores de α para el filtro gaussiano
alpha_values = [0.1, 1.5, 4.5, 5.0, 8.0, 10.0, 12.5, 15.0, 20.0, 30.0]

# Calcular la FFT de la señal original
fft_data = np.fft.fft(data)

# Gráfica
fig, axes = plt.subplots(len(alpha_values), 2, figsize=(14, 20))

# Aplicar el filtro para cada valor de α
for i, alpha in enumerate(alpha_values):
    # Crear el filtro gaussiano - filtro pasabajas
    gaussian_filter = np.exp(- (freqs*alpha)**2)
    # Filtrar en la frecuencia
    filtered_fft_data = fft_data*gaussian_filter
    # Transformada inversa para obtener la señal filtrada
    filtered_signal = np.fft.ifft(filtered_fft_data).real
    # Graficar la señal (Colum.1)
    axes[i, 0].plot(data, label="Original", alpha=0.6)
    axes[i, 0].plot(filtered_signal, label=f"Filtrada (α={alpha})")
    axes[i, 0].set_title(f"Señal Filtrada con α={alpha}")
    axes[i, 0].legend()
    # Graficar la transformada de Fourier (Colum.2)
    axes[i, 1].plot(np.log1p(np.abs(fft_data)), label="Original", alpha=0.6)
    axes[i, 1].plot(np.log1p(np.abs(filtered_fft_data)), label=f"Filtrada (α={alpha})")
    axes[i, 1].set_title("FFT de la Señal")
    axes[i, 1].legend()

# Ajustar diseño y guardar la figura
plt.tight_layout()
plt.savefig("3.1.pdf")




# 2. TRANSFORMADA RÁPIDA ------------------------------------
## 2a. Comparativa


## 2b. Manchas Solares
### 2b.a. Período del ciclo solar

# Leer el archivo txt, omitiendo la primera fila ("American")
df = pd.read_csv('list_aavso-arssn_daily.txt', delimiter='\s+')

df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df[df["date"] < "2010-01-01"]
df = df.sort_values("date")

t = np.arange(len(df))  # Tiempo en días desde el inicio
y = df["SSN"].values  # Número de manchas solares

Y = rfft(y)

plt.plot(Y)
plt.yscale('log')
plt.xscale('log')

freqs = rfftfreq(len(y), d=1)  # Frecuencias en 1/día

# Encontrar el pico principal
pico_principal = np.argmax(np.abs(Y[1:])) + 1  # Evitamos freq[0]
frecuencia_ciclo = freqs[pico_principal]
P_solar = 1 / frecuencia_ciclo / 365.25  # Convertir a años
print(f"2.b.a) {P_solar = }")

### 2b.b. Extrapolación con suavizado
M = min(50, len(Y))  # Asegurar que no excedemos la cantidad de coeficientes
N = len(y)

# Crear tiempo futuro desde 2012 hasta 2025
fecha_inicio = df["date"].min()
dias_desde_inicio = np.array((pd.date_range("2012-01-01", "2025-02-17", freq="D") - fecha_inicio).days)

# Aplicar ventana de suavizado a los armónicos
ventana = np.exp(-0.01 * np.arange(len(Y)))  # Atenuación exponencial
Y_suavizado = Y * ventana

# Asegurar que Y_suavizado y freqs tengan la forma correcta
y_pred = np.real(
    1/N * np.sum(
        (Y_suavizado[:M, None] * np.exp(2j * np.pi * freqs[:M, None] * dias_desde_inicio)),
        axis=0
    )
)

# Obtener la predicción del día de entrega (10 de febrero de 2025)
n_manchas_hoy = y_pred[-1]
print(f"2.b.b) {n_manchas_hoy = }")

# Graficar los datos originales y la extrapolación
plt.figure(figsize=(10, 5))
plt.plot(df["date"], y, label="Datos originales", alpha=0.6)
plt.plot(pd.to_datetime("2012-01-01") + pd.to_timedelta(dias_desde_inicio, unit="D"), y_pred, label="Predicción suavizada", linestyle="dashed")
plt.xlabel("Fecha")
plt.ylabel("Número de manchas solares")
plt.legend()
plt.grid()
plt.savefig("2.b.pdf")






