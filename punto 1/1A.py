import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv('Rhodium.csv')
df['Diferencia']=df["Intensity (mJy)"].diff()
#limpieza
df_corregido=df[((df["Diferencia"]>=-0.02) & (df["Diferencia"] <=0.02))]

# Datos originales
plt.figure(figsize=(12, 6))

# Gráfico original
plt.subplot(1, 2, 1)
plt.scatter(df["Wavelength (pm)"], df["Intensity (mJy)"], color='blue', label='Datos originales', s=10)
plt.axhline(y=0, color='black', linestyle='solid', linewidth=0.8)
plt.xlabel('Wavelength (pm)')
plt.ylabel('Intensity (mJy)')
plt.title('Datos Originales')
plt.legend()

# Gráfico limpio
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


# Contar datos eliminados
n_eliminados = len(df) - len(df_corregido)
print(f'1.a) Número de datos eliminados: {n_eliminados}')
    