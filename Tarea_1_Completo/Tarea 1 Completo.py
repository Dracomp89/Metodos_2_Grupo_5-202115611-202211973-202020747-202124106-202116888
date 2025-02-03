# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:40:34 2025

@author: eduar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import re
from scipy.signal import find_peaks

df = pd.read_csv('Rhodium.csv')
df['Diferencia']=df["Intensity (mJy)"].diff()
df_corregido=df[((df["Diferencia"]>=-0.02) & (df["Diferencia"] <=0.02))]

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df["Wavelength (pm)"], df["Intensity (mJy)"], color='blue', label='Datos originales', s=10)
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.8)
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Datos Originales')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(df_corregido["Wavelength (pm)"], df_corregido["Intensity (mJy)"], color='green', label='Datos corregidos', s=10)
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.8)
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Datos Corregidos')
plt.legend()

plt.tight_layout()
plt.savefig('limpieza.pdf')
plt.show()


n_eliminados = len(df) - len(df_corregido)
print(f'1.a) Número de datos eliminados: {n_eliminados}')
    

df = pd.read_csv('Rhodium.csv')
df['Diferencia'] = df["Intensity (mJy)"].diff()

varianza = 0.02
df_corregido = df[((df["Diferencia"] >= -varianza) & (df["Diferencia"] <= varianza))].copy()

fondo=df_corregido.copy()
picos = pd.DataFrame()

grado = 100
limite = 0.001  
max_it = 100
it = 0

while it < max_it:
    x = fondo["Wavelength (pm)"]
    y = fondo["Intensity (mJy)"]
    coeficientes = np.polyfit(x, y, grado)
    polinomio = np.poly1d(coeficientes)
    ajuste= polinomio(x)
    residuos = y - ajuste
    fondo['Residuos'] = residuos
    max_residuo = abs(residuos).max()
    if max_residuo > limite:
        indice= abs(residuos).idxmax()
        picos = pd.concat([picos, fondo.iloc[[indice]]], ignore_index=True)
        fondo = fondo.drop(indice).reset_index(drop=True)
    else:
        break
    it += 1
picos = picos.sort_values(by="Wavelength (pm)").reset_index(drop=True)
picos = picos.iloc[3:-4].reset_index(drop=True)
X=df_corregido["Wavelength (pm)"]
Y = df_corregido["Intensity (mJy)"]

plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='Datos', color='blue', alpha=0.6)
plt.plot(x, polinomio(x), label=f'Polinomio grado {grado}', color='red', linestyle='--')
plt.scatter(picos["Wavelength (pm)"], picos["Intensity (mJy)"], label='Picos', color='green', marker='x')
plt.axhline(0, color='black', linestyle=':')
plt.xlabel('Longitud de onda (pm)')
plt.ylabel('Intensidad (mJy)')
plt.title('Espectro completo')
plt.legend()
plt.tight_layout()
plt.savefig('espectro completo.pdf')
plt.show()

def gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

x_picos1=picos["Wavelength (pm)"].iloc[15:35]
y_picos1=picos["Intensity (mJy)"].iloc[15:35]
x_picos2=picos["Wavelength (pm)"].iloc[50:90]
y_picos2=picos["Intensity (mJy)"].iloc[50:90]
p0_2 = [y_picos2.max(), x_picos2.mean(), 10]
p0_1 = [y_picos1.max(), x_picos1.mean(), 10]
popt1, pcov = curve_fit(gauss , x_picos1, y_picos1, p0_1)
popt2, pcov = curve_fit(gauss , x_picos2, y_picos2, p0_2)

x_fit1 = np.linspace(70, 120, 200)  
y_fit1 = gauss(x_fit1, *popt1)
y_fit2 = gauss(x_fit1, *popt2)

print(picos)
# Graficar los picos detectados
plt.scatter(picos["Wavelength (pm)"], picos["Intensity (mJy)"], label='Picos detectados', color='green', zorder=5)
plt.plot(x_fit1, y_fit1, label='picos 1', color='blue')
plt.plot(x_fit1, y_fit2, label='pico 2', color='red')
plt.axhline(0, color='black', linestyle=':')
plt.xlabel('Longitud de onda (pm)')
plt.ylabel('Intensidad (mJy)')
plt.title('Ajuste de picos con Gaussianas')
plt.legend()
plt.tight_layout()
plt.savefig('ajuste_picos_gaussianas.pdf')
plt.show()
 
def calcular_fwhm(x, y):
    # Validación: si y está vacío, retornar NaN
    if len(y) == 0:
        return np.nan, np.nan, np.nan

    max_index = np.argmax(y)
    max_value = y[max_index]
    half_max = max_value / 2

    
    left_indices = np.where(y[:max_index] <= half_max)[0]
    right_indices = np.where(y[max_index:] <= half_max)[0]

    
    if len(left_indices) == 0 or len(right_indices) == 0:
        return x[max_index], max_value, np.nan

    left = left_indices[-1]
    right = right_indices[0] + max_index
    fwhm = x[right] - x[left]

    return x[max_index], max_value, fwhm


max_fondo, max_fondo_valor, fwhm_fondo = calcular_fwhm(fondo["Wavelength (pm)"], fondo["Intensity (mJy)"])


max_pico, max_pico_valor, fwhm_pico = calcular_fwhm(picos["Wavelength (pm)"], picos["Intensity (mJy)"])

print(f'1.c) Máximo fondo: {max_fondo:.4f} pm, Valor: {max_fondo_valor:.4f} mJy, FWHM: {fwhm_fondo if not np.isnan(fwhm_fondo) else "No disponible"} pm')
print(f'1.c) Máximo picos: {max_pico:.4f} pm, Valor: {max_pico_valor:.4f} mJy, FWHM: {fwhm_pico if not np.isnan(fwhm_pico) else "No disponible"} pm')


energia_total = np.trapz(df_corregido["Intensity (mJy)"], x=df_corregido["Wavelength (pm)"])
incertidumbre = 0.02 * energia_total

print(f'1.d) Energía total: {energia_total:.4f} mJ·pm, Incertidumbre: ±{incertidumbre:.4f} mJ·pm')




# Leer el archivo CSV como texto
archivo = "hysteresis.dat"

"""##**2.a. Datos**"""

filas = []
lista_1 = []
lista_2 = []
lista_3 = []
with open(archivo, "r") as file:
    for linea in file:
        patron = r'\b(\d{3})(\d\.)'
        linea = re.sub(patron, r'\1,\2', linea)
        linea = linea.replace(" ",",")
        linea = linea.replace("-",",-")
        elementos = linea.split(",")
        elementos = [elemento.strip() for elemento in elementos]
        lista_1.append(elementos[0])
        lista_2.append(elementos[1])
        lista_3.append(elementos[2])

tiempo = [float(num) for num in lista_1]
campo_externo = [float(num) for num in lista_2]
densidad_campo = [float(num) for num in lista_3]
print("Lista 1:", tiempo)
print("Lista 2:", campo_externo)
print("Lista 3:", densidad_campo)

plt.plot(tiempo, campo_externo, label='Campo Externo B [mT]', color='b')
plt.plot(tiempo, densidad_campo, label="Densidad de Campo Interno H [A/m]", color='r')


plt.xlabel('Tiempo [ms]')
plt.ylabel('B [mT] y H [A/m]')
plt.title('B,H vs. t')
plt.legend()
plt.grid()
plt.show()
plt.savefig("histérico.pdf", format="pdf")
plt.close()

"""##**2.b. Frecuencia de la señal**"""

time_seconds = [t / 1000 for t in tiempo]

peaks, _ = find_peaks(campo_externo, height=0)
peak_times = [time_seconds[i] for i in peaks]


if len(peak_times) > 1:
    periods = np.diff(peak_times)
    frequency = 1 / np.mean(periods)

    print(f"Frecuencia de oscilación de B: {frequency:.2f} Hz")
    print("Método: Se identificaron los picos de la señal B y se calculó el promedio de los períodos.")
else:
    print("No se encontraron suficientes picos para calcular la frecuencia.")

"""##**2.c. Energía perdida**"""

plt.plot(campo_externo, densidad_campo, label='Prueba', color='b')
plt.title('Histéresis magnética')
plt.xlabel('Campo Externo [mT]')
plt.ylabel('Densidad de Campo Interno [A/m]')
plt.grid()
plt.show()
plt.savefig("energy.pdf", format="pdf")
plt.close()

area = np.trapz(densidad_campo, campo_externo)
area= round(area,2)
print(f"Energía perdida por unidad de volumen: {area} J/m³")
