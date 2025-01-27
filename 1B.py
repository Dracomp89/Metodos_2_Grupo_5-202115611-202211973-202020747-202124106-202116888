import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

plt.figure(figsize=(10, 6))
plt.plot(picos["Wavelength (pm)"], picos["Intensity (mJy)"], label='Picos', color='green', linestyle='--')
plt.axhline(0, color='black', linestyle=':')
plt.xlabel('Longitud de onda (pm)')
plt.ylabel('Intensidad (mJy)')
plt.title('Picos')
plt.xlim(0, 300)
plt.legend()

# Guardar la última figura en un archivo PDF
plt.tight_layout()
plt.savefig('picos.pdf')
plt.show() 