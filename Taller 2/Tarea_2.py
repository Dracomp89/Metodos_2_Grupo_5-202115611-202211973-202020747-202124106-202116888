# -*- coding: utf-8 -*-
"""Tarea 2.py
"""




# 3. FILTROS
## 3a. Filtro gaussiano

# Importar librerías
import numpy as np
import matplotlib.pyplot as plt

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
plt.savefig("/content/drive/MyDrive/Colab_Notebooks/metodos2/t2/3.1.pdf")
