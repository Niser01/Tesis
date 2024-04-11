import random
import numpy as np


#parametros del problema
num_camps = 6  # Número de campamentos del problema
num_hormigas = 50  # Número de hormigas en la colonia
num_generaciones = 100  # Número de generaciones en el algoritmo genético
factor_evaporacion = 0.5  # Tasa de evaporación de feromonas
#cant_inicial_feromona = 1.0  # Nivel inicial de feromonas en las aristas
factor_cruce = 0.7  # Tasa de cruza genética
factor_mutacion = 0.1  # Tasa de mutación genética
factor_distancia = 0.6 # peso que recibe el factor distancia
factor_inclinacion = 0.1 # peso que recibe el factor inclinación
factor_clima = 0 # peso que recibe el clima

grafo_distancias = np.random.uniform(low=1.0, high=300.0, size=(num_camps, num_camps))
np.fill_diagonal(grafo_distancias, 0) 

tipos_de_terreno = [["Pavimentado",1], ["Barro",0.2], ["Rocas",0.6], ["Matorral",0.3]]
grafo_terreno = [[None for _ in range(num_camps)] for _ in range(num_camps)]
for i in range(num_camps):
    for j in  range(num_camps):
        terreno = random.choice(tipos_de_terreno)
        grafo_terreno[i][j] = terreno

grafo_inclinaciones = np.random.uniform(low=-60, high=60, size=(num_camps, num_camps))
np.fill_diagonal(grafo_inclinaciones, 0) 


print(grafo_terreno)        
print(grafo_distancias)
print(grafo_inclinaciones)

# Inicialización de feromonas en las aristas del grafo
pheromones = np.full((num_camps, num_camps), 0)

print(pheromones)

def generate_ant_solution():
    # Genera una solución candidata (recorrido de ciudades) para una hormiga
    return random.sample(range(num_camps), num_camps)

def evaluate_solution(solution):
    # Evalúa la longitud total de un recorrido de ciudades (solución)
    total_distance = sum(graph[solution[i-1], solution[i]] for i in range(num_camps))
    return total_distance

def update_pheromones(best_solution):
    # Actualiza las feromonas en función de la mejor solución encontrada
    pheromones *= (1 - evaporation_rate)  # Evaporación de feromonas
    for i in range(num_camps):
        pheromones[best_solution[i-1], best_solution[i]] += 1.0 / evaluate_solution(best_solution)


best_global_solution = None
best_global_distance = float('inf')

for generation in range(num_generations):
    # Generar soluciones para cada hormiga y evaluarlas
    solutions = [generate_ant_solution() for _ in range(num_ants)]
    distances = [evaluate_solution(sol) for sol in solutions]
    
    # Actualizar feromonas y encontrar la mejor solución de la generación
    best_solution_index = np.argmin(distances)
    best_solution = solutions[best_solution_index]
    if distances[best_solution_index] < best_global_distance:
        best_global_solution = best_solution
        best_global_distance = distances[best_solution_index]
    update_pheromones(best_solution)

    # Aplicar operadores genéticos a una parte de la población de hormigas
    selected_indices = random.sample(range(num_ants), int(crossover_rate * num_ants))
    selected_solutions = [solutions[idx] for idx in selected_indices]
    # Aplicar cruza y mutación a las soluciones seleccionadas
    # (Aquí debes implementar tus operadores de cruza y mutación según el esquema genético que elijas)

print("Mejor solución encontrada:", best_global_solution)
print("Distancia de la mejor solución:", best_global_distance)
