import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Función para generar peso según la estatura usando la "Fórmula de Broca"
def generar_peso(altura_metros):
    altura_cm = altura_metros * 100  # Convertir altura de metros a centímetros
    peso_broca = altura_cm - 100 # Calcular el peso ideal basado en la fórmula de Broca
    peso_minimo = random.uniform(15, 30) # Establecer un peso mínimo aleatorio entre 15 y 30 kg
    peso = random.uniform(peso_broca - 5, peso_broca + 5)# Generar un peso alrededor del peso conseguido por 
                                                         # la formula de Broca, con una variación de +/- 5 kg
    return max(peso, peso_minimo) # Asegurarse de que el peso no sea menor que el peso mínimo

estatura_metros = []
pesos = []

for i in range(100):
    altura_metros = random.uniform(0.8, 2.05)  # Altura entre 0.8 y 2.05 metros
    peso = generar_peso(altura_metros)
    estatura_metros.append(altura_metros)
    pesos.append(peso)

# Mostrar los 100 datos generados 
print("No. | Estatura (m) | Peso (kg)")
cont = 1
for i in range(len(estatura_metros)):
    altura = estatura_metros[i]
    peso = pesos[i]
    print(f"Nro: {cont:02d} | {altura:.2f}(m) | {peso:.2f}(kg)")
    cont += 1

# Graficar los datos
plt.scatter(estatura_metros, pesos, color='blue', label='Datos generados')

# Función de ajuste de la curva
def func(x, a, b, c):
    return a * x**2 + b * x + c

params, covariance = curve_fit(func, estatura_metros, pesos)

# Generar la curva de mejor ajuste
x_vals = np.linspace(min(estatura_metros), max(estatura_metros), 100)
y_vals = func(x_vals, *params)

# Graficar la curva de mejor ajuste
plt.plot(x_vals, y_vals, color='red', label='Curva de mejor ajuste')
plt.xlabel('Estatura (m)')
plt.ylabel('Peso (kg)')
plt.legend()
plt.show()
