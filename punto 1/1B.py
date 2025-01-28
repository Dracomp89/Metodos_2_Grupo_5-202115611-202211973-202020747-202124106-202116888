import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
 