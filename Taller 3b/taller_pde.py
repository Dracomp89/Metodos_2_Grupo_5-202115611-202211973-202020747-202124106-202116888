# -*- coding: utf-8 -*-
"""Taller PDE.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1CLYCUWsZAN1WkqyDgjioK7-oKs2ucIMZ
"""

# TALLER 3B
# 1. Poisson en un disco -------------------------------------------------------
from numba import njit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Malla Generada longitud NxN distanciada dx
N =25

# Crear la malla
x = np.linspace(-1.1, 1.1, N)
y = np.linspace(-1.1, 1.1, N)
dx = np.diff(x)[0]
X, Y = np.meshgrid(x, y)
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

#valores aleatorios y parametros
tolerancia = 1e-8
max_iter = 15000
phi = np.random.rand(N, N)
# Función densidad
rho = -X - Y
#dentro del disco
dentro = r <= 1

@njit
def Jacobi(phi, rho, inside, dx, tolerance, max_iter):
    for iteracion in range(max_iter):
        phi_vieja = phi.copy()
        for i in range(1, phi.shape[0]-1):
            for j in range(1, phi.shape[1]-1):
                if inside[i, j]:
                    phi[i, j] = 0.25 * (phi_vieja[i+1, j] + phi_vieja[i-1, j] + phi_vieja[i, j+1] + phi_vieja[i, j-1]) + (dx**2) * np.pi * rho[i, j]
        # Verificar convergencia
        diff = np.trace(np.abs(phi - phi_vieja))
        if diff < tolerance:
            print(f'Convergió en {iteracion} iteraciones.')
            break
    else:
        print('No convergió.')

    return phi


phi = Jacobi(phi, rho, dentro, dx, tolerancia, max_iter)
phi[r >= 1] = np.sin(7 * theta[r >= 1])

# Graficar condiciones de frontera
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.title("Condiciones de frontera")
plt.pcolormesh(X, Y, phi, shading='auto', cmap='jet')
plt.colorbar(label=r'$\phi$')
plt.xlabel('x')
plt.ylabel('y')

# Graficar solución 3D
ax = plt.subplot(1, 2, 2, projection='3d')
ax.set_title("Muestra de solución Jacobi")
surf = ax.plot_surface(X, Y, phi, cmap='jet')
fig = plt.gcf()
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Guardar figura
plt.tight_layout()
plt.savefig('1.png')

# 2. ONDAS 1D Y REFLEXIONES ----------------------------------------------------
import matplotlib.animation as animation
import os

# Parámetros del problema
L = 2          # Longitud del dominio
Nx = 100       # Número de puntos en x
dx = L / (Nx - 1)  # Paso espacial
c = 1.0        # Velocidad de la onda
dt = 0.005     # Paso temporal
Nt = 700       # Número de pasos en el tiempo

# Cálculo del coeficiente de Courant
C = c * dt / dx
if C >= 1:
    raise ValueError(f"El coeficiente de Courant C = {C:.2f} debe ser menor que 1 para estabilidad numérica.")
else:
    print(f"El coeficiente de Courant C = {C:.2f} es adecuado para estabilidad numérica.")

# Pedir al usuario la carpeta de salida
#output_folder = input("Ingrese la ruta de la carpeta donde desea guardar el video: ")
#os.makedirs(output_folder, exist_ok=True)

# Condición inicial
def u_inicial(x):
    return np.exp(-125 * (x - 1) ** 2)

# Inicialización de la malla para cada condición de frontera
tabla_dirichlet = np.zeros((Nt, Nx))
tabla_neumann = np.zeros((Nt, Nx))
tabla_periodicas = np.zeros((Nt, Nx))
x = np.linspace(0, L, Nx)

# Condiciones iniciales
tabla_dirichlet[0, :] = u_inicial(x)
tabla_neumann[0, :] = u_inicial(x)
tabla_periodicas[0, :] = u_inicial(x)

# Primer paso de integración
tabla_dirichlet[1, 1:-1] = tabla_dirichlet[0, 1:-1] + 0.5 * (C ** 2) * (tabla_dirichlet[0, :-2] - 2 * tabla_dirichlet[0, 1:-1] + tabla_dirichlet[0, 2:])
tabla_neumann[1, 1:-1] = tabla_neumann[0, 1:-1] + 0.5 * (C ** 2) * (tabla_neumann[0, :-2] - 2 * tabla_neumann[0, 1:-1] + tabla_neumann[0, 2:])
tabla_periodicas[1, 1:-1] = tabla_periodicas[0, 1:-1] + 0.5 * (C ** 2) * (tabla_periodicas[0, :-2] - 2 * tabla_periodicas[0, 1:-1] + tabla_periodicas[0, 2:])

# Iteración en el tiempo
for n in range(1, Nt - 1):
    # Dirichlet
    tabla_dirichlet[n + 1, 1:-1] = (2 * tabla_dirichlet[n, 1:-1] - tabla_dirichlet[n - 1, 1:-1] +
                                     C ** 2 * (tabla_dirichlet[n, :-2] - 2 * tabla_dirichlet[n, 1:-1] + tabla_dirichlet[n, 2:]))
    tabla_dirichlet[n + 1, 0] = 0
    tabla_dirichlet[n + 1, -1] = 0

    # Neumann
    tabla_neumann[n + 1, 1:-1] = (2 * tabla_neumann[n, 1:-1] - tabla_neumann[n - 1, 1:-1] +
                                   C ** 2 * (tabla_neumann[n, :-2] - 2 * tabla_neumann[n, 1:-1] + tabla_neumann[n, 2:]))
    tabla_neumann[n + 1, 0] = tabla_neumann[n + 1, 1]   # Reflejo en x = 0
    tabla_neumann[n + 1, -1] = tabla_neumann[n + 1, -2] # Reflejo en x = L

    # Periódicas
    tabla_periodicas[n + 1, 1:-1] = (2 * tabla_periodicas[n, 1:-1] - tabla_periodicas[n - 1, 1:-1] +
                                     C ** 2 * (tabla_periodicas[n, :-2] - 2 * tabla_periodicas[n, 1:-1] + tabla_periodicas[n, 2:]))
    tabla_periodicas[n + 1, 0] = tabla_periodicas[n + 1, -2]  # Condición periódica en x = 0
    tabla_periodicas[n + 1, -1] = tabla_periodicas[n + 1, 1]  # Condición periódica en x = L

# Animación con tres subplots
fig, axes = plt.subplots(3, 1, figsize=(6, 12))
labels = ["Dirichlet", "Neumann", "Periódicas"]
tablas = [tabla_dirichlet, tabla_neumann, tabla_periodicas]
lines = []

for ax, label, tabla in zip(axes, labels, tablas):
    line, = ax.plot(x, tabla[0, :], 'k')
    ax.set_ylim(-1, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,t)")
    ax.set_title(label)
    ax.text(1.5, 0.8, label, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    lines.append(line)

def actualizar(frame):
    for line, tabla in zip(lines, tablas):
        line.set_ydata(tabla[frame, :])
    return lines

ani = animation.FuncAnimation(fig, actualizar, frames=np.linspace(0, Nt - 1, 60, dtype=int), interval=30, blit=True, repeat=True)

# Guardar el video
ani.save('2.mp4', writer=animation.FFMpegWriter(fps=30))

# 3. Ondas no lineales: plasma y fluidos ---------------------------------------

# Parámetros del problema
alpha = 0.022
L = 2  # Dominio espacial x ∈ (0, 2]
N = 200  # Número de puntos espaciales
h = L / N  # Paso espacial
k = 1e-4  # Paso temporal
T_max = 2  # Tiempo máximo de simulación
steps = int(T_max / k)  # Número de pasos de tiempo

# Malla espacial y condición inicial
x = np.linspace(0, L, N, endpoint=False)
U = np.cos(np.pi * x)  # u(0, x) = cos(pi*x)

u = np.zeros( (N,steps) )
u[:,0]=U
u[:,1]=U

# poniendo los tipos de los argumentos numba compila solo
from numba import njit
@njit("i8(f8[:,:],f8,f8,f8)")
def evolve_heat(u, dx, dt, α): # Pass steps and N as arguments
    # Esquema numérico de diferencias finitas
    for j in range(1, u.shape[1] - 1):
        for i in range(u.shape[0]):
            ip1 = (i + 1) % N
            im1 = (i - 1) % N
            ip2 = (i + 2) % N
            im2 = (i - 2) % N
            u[i,j+1] = u[i,j-1] - ((1/3)*(k/h)*(u[ip1,j]+u[i,j]+u[im1,j])*(u[ip1,j]-u[im1,j]))\
                    - (((alpha**2)*k)/(h**3))*(u[ip2,j]-(2*u[ip1,j])+(2*u[im1,j])-u[im2,j])
    return 0


evolve_heat(u, h, k, alpha)

fig, ax = plt.subplots(figsize=(8, 3))
cax = ax.imshow(u.T, aspect='auto', origin='lower', extent=[0, L, 0, T_max], cmap='magma')
fig.colorbar(cax, label='$\psi(t, x)$')
ax.set_xlabel("Time [s]")
ax.set_ylabel("Position x [m]")


# 4. Ecuación de onda 2D con velocidad variable --------------------------------

# Parámetros del dominio
Lx, Ly = 1.0, 2.0  # Dimensiones del tanque (m)
dx, dy = 0.01, 0.01  # Resolución espacial
Nx, Ny = int(Lx/dx), int(Ly/dy)  # Número de puntos en la malla

# Parámetros de la onda
c_base = 0.5  # Velocidad de onda en m/s
c_lente = c_base / 5  # Velocidad en la apertura

# Condiciones de la pared y apertura
w_x, w_y = 0.4, 0.04  # Dimensiones de la apertura en metros
x_wall = Lx / 4  # Posición de la pared

# Creación de la malla espacial
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y)

# Definir velocidad variable c(x, y)
c = np.ones((Ny, Nx)) * c_base

# Aplicar condición de pared
mask_wall = (np.abs(Y - Ly / 2) < w_y / 2) & (np.abs(X - x_wall) < w_x / 2)
c[mask_wall] = 0  # Paredes fijas

# Aplicar condición de lente
mask_lente = ((X - x_wall) ** 2 + 3 * (Y - Ly / 2) ** 2 <= 1 / 25) & (Y > Ly / 2)
c[mask_lente] = c_lente

# Parámetros temporales
dt = 0.002  # Paso de tiempo
Nt = 1000  # Número de pasos de tiempo

# Inicialización de la onda
u = np.zeros((Ny, Nx), dtype=np.float64)
u_prev = np.zeros((Ny, Nx), dtype=np.float64)
u_next = np.zeros((Ny, Nx), dtype=np.float64)

# Fuente de oscilación
frec_fuente = 10  # Hz
fuente_x, fuente_y = 0.5, 0.5  # Posición de la fuente
idx_fuente = (np.abs(x - fuente_x)).argmin()
idy_fuente = (np.abs(y - fuente_y)).argmin()

# Parámetro de Courant
C = c * dt / dx
assert np.all(C < 1), "El coeficiente de Courant debe ser menor que 1 para estabilidad"

@njit
def actualizar_onda(u, u_prev, u_next, C, mask_wall, idy_fuente, idx_fuente, frec_fuente, t):
    Ny, Nx = u.shape
    for i in range(1, Ny - 1):
        for j in range(1, Nx - 1):
            if not mask_wall[i, j]:  # Solo actualiza donde no hay pared
                u_next[i, j] = (2 * u[i, j] - u_prev[i, j] +
                                (C[i, j] ** 2) * (
                                    (u[i, j + 1] - 2 * u[i, j] + u[i, j - 1]) +
                                    (u[i + 1, j] - 2 * u[i, j] + u[i - 1, j])
                                ))

    # Aplicar la fuente oscilante
    u_next[idy_fuente, idx_fuente] = np.sin(2 * np.pi * frec_fuente * t * dt)

    return u_next, u

# Configuración de la animación
fig, ax = plt.subplots()
cmap = ax.imshow(u, extent=[0, Lx, 0, Ly], origin='lower', cmap='seismic', vmin=-1, vmax=1)
plt.colorbar(cmap)

def update(frame):
    global u, u_prev, u_next
    u_next, u_prev = actualizar_onda(u, u_prev, u_next, C, mask_wall, idy_fuente, idx_fuente, frec_fuente, frame)
    u = u_next.copy()
    cmap.set_array(u)
    return cmap,

ani = animation.FuncAnimation(fig, update, frames=Nt, interval=dt * 1000, blit=False)

# Guardar la animación
ani.save('onda.gif', writer='pillow')