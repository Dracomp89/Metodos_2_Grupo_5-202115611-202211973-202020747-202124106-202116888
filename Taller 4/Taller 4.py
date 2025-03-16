# -*- coding: utf-8 -*-
"""Untitled14.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1m1hn6uA73MFuB2xrs8Ps6-Ca8seXALDJ
"""

#Taller 4 - Grupo 5

import numpy as np
import matplotlib.pyplot as plt

# 1. INTEGRACIÓN INDIRECTA
# 1.a GENERAR DATOS ALEATORIOS

def g_x(x, n=10, alpha=4/5):
    return sum(np.exp(-(x - k)**2 * k) / k**alpha for k in range(1, n+1))

def metropolis(f, x0, n, sigma):
    """Genera una cadena de Markov usando el algoritmo de Metropolis-Hastings."""
    samples = np.zeros(n)
    samples[0] = x0
    for i in range(1, n):
        sample_new = samples[i-1] + np.random.normal(0, sigma)
        if np.random.rand() < f(sample_new) / f(samples[i-1]):
            samples[i] = sample_new
        else:
            samples[i] = samples[i-1]
    return samples

# Generar muestras
samples = metropolis(lambda x: g_x(x), x0=5, n=500000, sigma=1.0)

# Crear histograma
plt.figure(figsize=(8,6))
plt.hist(samples, bins=200, density=True, alpha=0.6, color='b')
plt.xlabel("x")
plt.ylabel("Frecuencia")
plt.title("Histograma de muestras generadas con Metropolis-Hastings")
plt.savefig("1.a.pdf")

# 1.b INTEGRAR

def g_x_norm(x, n=10, alpha=4/5):
    return sum(np.exp(-(x - k)**2 * k) / k**(alpha + 1/2) for k in range(1, n+1))

def f_x(x):
    return np.exp(-x**2)

f_over_g = f_x(samples) / g_x_norm(samples)
A = np.sqrt(np.pi) / np.mean(f_over_g)
std_A = (np.sqrt(np.pi) / np.sqrt(len(samples))) * np.std(f_over_g)

# Imprimir resultado
print(f"1.b) A = {A} ± {std_A}")



# 3. MODELO DE ISING CON METROPOLIS-HASTINGS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

# Parámetros
N = 150  # Matriz (N x N)
J = 0.2  # Cte. de interacción entre espines, J>0 hace que los espines
                 # se alineen en la misma dirección (regiones homogéneas) -> ferromagnético
beta = 10  # Inverso de la temperatura (1/k_B*T) ~ T=0.1 -> baja temp.
num_frames = 500  # Frames
iteraciones_por_frame = 400  # Iteraciones entre frame y frame

# 1. Inicializar espines aleatorios en -1 y 1
spins = np.random.choice([-1, 1], size=(N, N))

def calcular_delta_energia(spins, i, j, J):
  """
  Calcula la diferencia de energía entre un espín y sus vecinos.
  """
  # Condiciones de frontera periódicas
  arriba = spins[(i - 1) % N, j]
  abajo = spins[(i + 1) % N, j]
  izquierda = spins[i, (j - 1) % N]
  derecha = spins[i, (j + 1) % N]

  # 3. Energía con el espín actual y energía si se invierte
  suma_vecinos = arriba + abajo + izquierda + derecha
  delta_E = 2 * J * spins[i, j] * suma_vecinos
  return delta_E

# Configurar la figura de la animación
fig, ax = plt.subplots()
custom_cmap = ListedColormap(["purple", "yellow"])
im = ax.imshow(spins, cmap=custom_cmap, vmin=-1, vmax=1)  # Mapa de colores

def actualizar(frame):
  """
  Función que se ejecuta en cada frame de la animación usando Metropolis-Hastings.
  """
  global spins
  for _ in range(iteraciones_por_frame): # 6. goto 2
      # 2. Elegir un espín al azar
      i, j = np.random.randint(0, N, size=2) 
      # 3. Calcular ΔE
      delta_E = calcular_delta_energia(spins, i, j, J)

      # Condiciones de aceptación de Metropolis
      if delta_E <= 0 or np.random.rand() < np.exp(-beta * delta_E): # Forma: if 4. or 5.: acepta la nueva conf.
          spins[i, j] *= -1  # Cambiar el espín

  im.set_array(spins)  # Actualizar la imagen
  return [im]

# Crear animación
ani = animation.FuncAnimation(fig, actualizar, frames=num_frames, interval=50, blit=True)
# Guardar el video
ani.save("3.mp4", writer=animation.FFMpegWriter(fps=30))

#PUNTO 2
# Parámetros en cm
D1 = 50  
D2 = 50  
wavelength = 670e-7  # (670 nm)
A = 0.04 
a = 0.01 
d = 0.1  
# Número de muestras
N = 100000  

# Muestras en x
x_samples = np.random.uniform(-A/2, A/2, N)

# Generar muestras en y 
y_samples = np.random.choice([-d/2, d/2], size=N) + np.random.uniform(-a/2, a/2, N)

# Valores de z para evaluar la intensidad
z_values = np.linspace(-0.4, 0.4, 1000)

# Evaluación de la integral de camino con Monte Carlo
def feynman_integral(z, x_samples, y_samples, N):
    a = (2 * np.pi / wavelength) * (D1 + D2)
    term1 = np.exp(1j * a)
    term2 = np.exp(1j * np.pi / (wavelength * D1) * (x_samples - y_samples) ** 2)
    term3 = np.exp(1j * np.pi / (wavelength * D1) * (z - y_samples) ** 2)
    integrand = term1 * term2 * term3
    I = np.mean(integrand)
    error = np.sqrt(np.var(integrand) / N)

    return np.abs(I) ** 2, error

intensidad_feynman = np.zeros(len(z_values))
errores_feynman = np.zeros(len(z_values))

for i, z in enumerate(z_values):
    intensidad_feynman[i], errores_feynman[i] = feynman_integral(z, x_samples, y_samples, N)

# Normalización por el máximo
intensidad_feynman /= np.max(intensidad_feynman)

theta = np.arctan(z_values / D2)
clasica = (np.cos(np.pi * d / wavelength * np.sin(theta))**2 *
           np.sinc(a / wavelength * np.sin(theta))**2)

# Normalización por el máximo
clasica /= np.max(clasica)


plt.figure(figsize=(8, 5))
plt.plot(z_values, intensidad_feynman, label="Integral de camino (Feynman)", linestyle="dashed")
plt.fill_between(z_values, intensidad_feynman - errores_feynman, 
                 intensidad_feynman + errores_feynman, alpha=0.3, color="blue")
plt.plot(z_values, clasica, label="Modelo clásico (Fresnel)", linestyle="solid")
plt.xlabel("Posición en la pantalla (cm)")
plt.ylabel("Intensidad normalizada")
plt.title("Comparación entre la integral de camino y el modelo clásico")
plt.legend()
plt.savefig("2.pdf", bbox_inches="tight")
plt.show()



