# -*- coding: utf-8 -*-
"""Tarea 2.py
"""
# Importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq, irfft
import pandas as pd
from scipy import fftpack
from skimage import io, color

# 1. Transformada general
#a.
def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float],
 frecuencias:NDArray[float], ruido:float) -> NDArray[float]:
 ts = np.arange(0.,t_max,dt)
 ys = np.zeros_like(ts,dtype=float)
 for A,f in zip(amplitudes,frecuencias):
     ys += A*np.sin(2*np.pi*f*ts)
     ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
 return ts,ys

def Fourier(t:NDArray[float], y:NDArray[float], f:float) -> complex:
 return np.sum(y*np.exp(-2j*np.pi*t*f))


frecuencias=np.array([0.1,0.2,0.5])
amplitudes=np.array([3,2,1])
ruido1=0.0
ruido2=3
tmax=50
dt=0.1
x=np.arange(0.,tmax,dt)

freq=np.arange(0.0,0.6,0.0001)

ts1,señal1=datos_prueba(tmax,dt,amplitudes,frecuencias,ruido1)
ts2,señal2=datos_prueba(tmax,dt,amplitudes,frecuencias,ruido2)

F1=[Fourier(ts1,señal1,f) for f in freq]
F2=[Fourier(ts2,señal2,f) for f in freq]

plt.scatter(ts1, señal1)
plt.show()
plt.scatter(ts2, señal2)
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6))  # Dos filas, una columna

axs[0].plot(freq, np.abs(F1))
axs[0].set_title("Transformada de Fourier sin ruido")
axs[0].set_xlabel("Frecuencia")
axs[0].set_ylabel("Amplitud")
axs[0].grid()

axs[1].plot(freq, np.abs(F2))
axs[1].set_title("Transformada de Fourier con ruido")
axs[1].set_xlabel("Frecuencia")
axs[1].set_ylabel("Amplitud")
axs[1].grid()
plt.tight_layout()  # Ajustar para evitar superposición
plt.savefig("1.a.pdf", dpi=300, bbox_inches='tight')
plt.show()
print("1.a) Se pierde resolución de los picos objetivo")
#b

frecuenciab = np.array([5])
amplitudb = np.array([3])
ruido1 = 0.0
tmax = np.arange(10, 100, 10)
dt = 0.001
anchos=[]

for i in tmax:
    x = np.arange(0., i, dt)
    fb = np.arange(4.7, 5.3, 0.0001)  
    ts1b, señal1b = datos_prueba(i, dt, amplitudb, frecuenciab, ruido1)
    
    F1b = np.array([Fourier(ts1b, señal1b, f) for f in fb])
    F1b_abs = np.abs(F1b)
    pico_max = np.max(F1b_abs)
    mitad = pico_max / 2
    indices_cercanos = np.argsort(np.abs(F1b_abs - mitad))[:2]
    indices_cercanos.sort()
    widths = np.abs(fb[indices_cercanos[1]] - fb[indices_cercanos[0]])
    anchos.append(widths)
    
    print(f"Ancho medio para t_max={i}: {widths:.65}")  

    plt.plot(fb, F1b_abs)
    plt.title(f"Transformada de Fourier para $t_{{max}}$ = {i}")
    plt.xlabel("Frecuencia")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.show()

print(anchos)
loga = np.log(anchos)
logt = np.log(tmax)
def modelo(x,A,B):
    return A*x+B
coeficiente, _ =curve_fit(modelo,logt,loga)
A,B=coeficiente
print(f"Ajuste: y = {A:.4f}X+ {B:.4f}")

# 🔹 Generar valores ajustados
logt_p = np.linspace(min(logt), max(logt), 100)
loga_p = modelo(logt_p, A, B)

# 🔹 Graficar los datos y el ajuste
plt.scatter(logt, loga, label="Datos experimentales", color="red")
plt.plot(logt_p, loga_p, label=f"Ajuste: y = {A:.4f}X+{B:.4f}", color="blue")
plt.xlabel("Log(t)")
plt.ylabel("Log(Anchos)")
plt.legend()
plt.grid()
plt.tight_layout() 
plt.savefig("1.b.pdf", dpi=300, bbox_inches='tight')
plt.show()

# c

def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float],
 frecuencias:NDArray[float], ruido:float) -> NDArray[float]:
 ts = np.arange(0.,t_max,dt)
 ys = np.zeros_like(ts,dtype=float)
 for A,f in zip(amplitudes,frecuencias):
     ys += A*np.sin(2*np.pi*f*ts)
     ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
 return ts,ys

def Fourier(t:NDArray[float], y:NDArray[float], f:float) -> complex:
 return np.sum(y*np.exp(-2j*np.pi*t*f))

columnas = ["Tiempo", "Intensidad", "Incertidumbre"]
url="https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/phot/I/OGLE-LMC-CEP-0001.dat"
datos=pd.read_csv(url, delim_whitespace=True, header=None, names=columnas)

fc=np.arange(-0.005,0.005,0.00001)
F1c=[Fourier(datos["Tiempo"], datos["Intensidad"], f) for f in fc]
plt.plot(fc,np.abs(F1c))
plt.title("Transformada de Fourier")
plt.xlabel("Frecuencia")
plt.ylabel("Amplitud")
plt.grid()
plt.show()

dt_medio = np.median(np.diff(datos["Tiempo"]))  # Mediana de los intervalos de tiempo
frecuencia_nyquist = 1 / (2 * dt_medio)  # Frecuencia de Nyquist


print(f"1.c) f Nyquist: {frecuencia_nyquist}")

def Fourier_multiple(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
    N = len(t)
    resultados = np.zeros(len(f)) + 0.0j
    for i, freq in enumerate(f):
        exp_term = np.exp(-2j * np.pi * t * freq)
        resultados[i] = np.sum(y * exp_term) / N
    return resultados

columnas = ["Tiempo", "Intensidad", "Incertidumbre"]
url = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/lmc/cep/phot/I/OGLE-LMC-CEP-0001.dat"
datos = pd.read_csv(url, sep='\s+', header=None, names=columnas)

# Calcular diferencias temporales
dt_dif = np.diff(datos["Tiempo"].values)

# Centrar la señal
y_centrada = datos["Intensidad"].values - np.mean(datos["Intensidad"].values)

# Crear espacio de frecuencias
frecuencias_eval = np.linspace(0, 10, 100000)

# Aplicar transformada de Fourier
transformada = Fourier_multiple(datos["Tiempo"].values, y_centrada, frecuencias_eval)
amplitud_transformada = np.abs(transformada)

# Encontrar el pico principal
pico_index = np.argmax(amplitud_transformada)
f_true = frecuencias_eval[pico_index]
print(f"1.c) f true: {f_true}")

# Calcular la fase
fase = np.mod(f_true * datos["Tiempo"].values, 1)

# Graficar intensidad vs fase
plt.scatter(fase, y_centrada, s=1, color='blue')
plt.xlabel("Fase φ")
plt.ylabel("Intensidad centrada")
plt.title("Intensidad vs. Fase")
plt.grid()
plt.savefig("1.c.pdf")
plt.show()

# 2. Transformada rápida
# a
datos = pd.read_csv("H_field (1).csv", sep=",")
t = datos["t"].to_numpy()
H = datos["H"].to_numpy()

dt = np.mean(np.diff(t))
n = len(H)
f_fft = np.fft.rfftfreq(n, dt)  
F_fast = np.fft.rfft(H)
indice_max_fast = np.argmax(np.abs(F_fast))  
f_fast = f_fft[indice_max_fast]  

def Fourier(t: np.ndarray, y: np.ndarray, f: float) -> complex:
    return np.sum(y * np.exp(-2j * np.pi * t * f))

f_gft=np.arange(0,6,0.005)
F_general = np.array([Fourier(t, H, f) for f in f_gft])

indice_max_general = np.argmax(np.abs(F_general))
f_general = f_gft[indice_max_general]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(f_fft, np.abs(F_fast))
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.title("Transformada Rápida de Fourier (rFFT)")
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(f_gft, np.abs(F_general))
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Amplitud")
plt.title("Transformada General de Fourier")
plt.grid()

plt.tight_layout()
plt.show()

print(f"2.a) f_fast = {f_fast:.5f} Hz; f_general = {f_general:.5f} Hz")

phi_f = np.mod(f_fast * t, 1)
phi_g = np.mod(f_general * t, 1)

# Graficar los resultados
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(phi_f, H, s=10, label=r"$\varphi_{\mathrm{fast}}$")
plt.xlabel("Fase")
plt.ylabel("H")
plt.title("Transformada Rápida de Fourier (rFFT)")
plt.grid()

plt.subplot(1, 2, 2)
plt.scatter(phi_g, H, s=5, label=r"$\varphi_{\mathrm{general}}$")
plt.xlabel("Fase")
plt.ylabel("H")
plt.title("Transformada General de Fourier (gFFT)")
plt.grid()
plt.tight_layout() 
plt.savefig("2.a.pdf")
plt.show()

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


### 3b. Limpieza

# Cargar imagen
image_path = 'catto.png'
image = io.imread(image_path)

# Convertir a escala de grises (si es necesario)
gray_image = color.rgb2gray(image)

# Mostrar la imagen original
plt.figure(figsize=(6,6))
plt.imshow(gray_image, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')
plt.show()

# Aplicar la Transformada de Fourier
f_transform = fftpack.fft2(gray_image)
f_transform_shifted = fftpack.fftshift(f_transform)  # Centrar la frecuencia en 0

# Mostrar el espectro de la imagen
magnitude_spectrum = np.abs(f_transform_shifted)
plt.figure(figsize=(6,6))
plt.imshow(np.log(magnitude_spectrum + 1), cmap='gray')  # +1 para evitar log(0)
plt.title('Espectro de Frecuencia')
plt.axis('off')

# Eliminar las frecuencias correspondientes al ruido periódico
# Para este ejemplo, supondremos que el ruido está en bandas específicas, por ejemplo, cerca del centro
rows, cols = gray_image.shape
crow, ccol = rows // 2, cols // 2  # Centro de la imagen

# Crear una máscara de paso bajo (eliminar las frecuencias altas relacionadas con el ruido)
mask = np.ones((rows, cols))
radius = 30  # Ajusta este valor según la imagen para capturar el ruido

# Borrar las frecuencias cercanas al centro (las que podrían ser ruido)
mask[crow-radius:crow+radius, ccol-radius:ccol+radius] = 0

# Aplicar la máscara en el dominio de la frecuencia
f_transform_shifted_filtered = f_transform_shifted * mask

# Transformada inversa para obtener la imagen filtrada
f_transform_filtered = fftpack.ifftshift(f_transform_shifted_filtered)
image_filtered = np.abs(fftpack.ifft2(f_transform_filtered))

# Mostrar la imagen filtrada
plt.figure(figsize=(6,6))
plt.imshow(image_filtered, cmap='gray')
plt.title('Imagen Filtrada (Ruido Eliminado)')
plt.axis('off')



# 2. TRANSFORMADA RÁPIDA ------------------------------------
## 2b. Manchas Solares
### 2b.a. Período del ciclo solar

import matplotlib.pyplot as plt
from datetime import datetime

# Inicializar listas vacías
dates = []
ssn_values = []

# Leer el archivo manualmente
with open(file_path, "r") as file:
    lines = file.readlines()

# Extraer fechas y valores de manchas solares
for line in lines:
    parts = line.split()
    if len(parts) != 4:  # Saltar líneas incorrectas
        continue
    try:
        # Convertir los datos numéricos
        year, month, day, ssn = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
        date = datetime(year, month, day)

        # Filtrar solo datos hasta el 1 de enero de 2010
        if date < datetime(2010, 1, 1):
            dates.append(date)
            ssn_values.append(ssn)

    except ValueError:
        continue  # Saltar encabezados o líneas mal formateadas

# Número de datos
N = len(ssn_values)

# FFT y frecuencias
fft_data = np.fft.fft(ssn_values)
freqs = np.fft.fftfreq(N, d=1)  # d=1 asume separación de 1 día

# Usar solo frecuencias positivas
positive_freqs = freqs[freqs > 0]
positive_fft = np.abs(fft_data[freqs > 0])

# Graficar FFT en escala log-log
plt.figure(figsize=(8, 5))
plt.loglog(positive_freqs, positive_fft)
plt.xlabel("Frecuencia (1/día)")
plt.ylabel("Magnitud de la FFT")
plt.title("Transformada de Fourier de las manchas solares")
plt.grid()

# Encontrar la frecuencia con mayor magnitud
dominant_freq = positive_freqs[np.argmax(positive_fft)]

# Convertir a período en años
P_solar = 1 / dominant_freq / 365.25

print(f"2.b.a) P_solar = {P_solar:.2f} años")



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






