import numpy as np
from scipy.spatial import distance
import networkx as nx
import matplotlib.pyplot as plt
import json

# Genera puntos aleatorios en un espacio bidimensional
n_points = 20
pos = {i: (np.random.random(), np.random.random()) for i in range(n_points)}

# Crea un grafo vacío con NetworkX
G = nx.Graph()
for i in range(n_points):
    G.add_node(i)

# Calcula la distancia euclidiana como métrica para el problema TSP
print(distance.euclidean(pos[0], pos[1]))

# Algoritmo genético

# Tamaño de la población inicial
n_population = 1000

# Genera una lista de todos los puntos disponibles
possible_gen = list(pos.keys())

# Crea una población inicial con rutas aleatorias
initial_population = [np.random.choice(possible_gen, n_points, False) for i in range(n_population)]

# Función para calcular la distancia total de una ruta
def get_total_distance(order, pos):
    order = list(order)
    ring_path = order.copy()
    ring_path.append(order[0])
    total_distance = 0
    for i in range(n_points):
        pos1 = pos[ring_path[i]]
        pos2 = pos[ring_path[i+1]]
        total_distance += distance.euclidean(pos1, pos2)
    return total_distance

# Calcula la distancia total de una ruta de ejemplo en la población inicial
print(get_total_distance(initial_population[0], pos))

# Calcula la distancia total de toda la población inicial y normaliza las distancias
current_population = initial_population
fitness = np.array([get_total_distance(parent, pos) for parent in current_population])
fitness = fitness.max() - fitness + 1
fitness = fitness / fitness.sum()

# Función de mutación
def mutation(order):
    mutation_rate = 0.001
    for i in range(n_points):
        if np.random.random() < mutation_rate:
            j = np.random.randint(n_points)
            order[i], order[j] = order[j], order[i]
    return order

# Combinación de genes de dos padres
def combine_genes(parent1, parent2):
    parent1 = list(parent1)
    parent2 = list(parent2)
    new_order = parent1.copy()
    for i in range(n_points):
        if np.random.random() < 0.5:
            idx = new_order.index(parent2[i])
            new_order[idx] = new_order[i]
            new_order[i] = parent2[i]
    return new_order

# Evolución

# Número de épocas
epochs = 1000

# Listas para almacenar las distancias mínimas y medias
min_path_distance = []
min_distance = []
mean_distance = []

for t in range(epochs):
    new_generation = []
    for _ in range(n_population - 1):
        parents = np.random.choice(n_population, p=fitness, size=2, replace=False)
        parent1 = current_population[parents[0]]
        parent2 = current_population[parents[1]]
        new_path = combine_genes(parent1, parent2)
        new_path = mutation(new_path)
        new_generation.append(new_path)

    idx = np.where(fitness == fitness.max())[0][0]
    new_generation.append(current_population[idx])
    current_population = new_generation

    fitness = [get_total_distance(parent, pos) for parent in current_population]
    min_distance.append(min(fitness))
    mean_distance.append(np.mean(fitness))
    fitness = np.array(fitness)
    fitness = fitness.max() - fitness + 1
    fitness = fitness / fitness.sum()

    if t % 100 == 0:
        min_idx = np.where(fitness == fitness.max())[0][0]
        d = get_total_distance(current_population[min_idx], pos)
        print(t, d)
        if d not in min_path_distance:
            min_path_distance.append(d)

            draw_edges = current_population[min_idx].copy()
            draw_edges = list(draw_edges)
            draw_edges.append(draw_edges[0])

            G = nx.Graph()
            for i in range(n_points):
                G.add_node(i)

            for i in range(len(draw_edges) - 1):
                G.add_edge(draw_edges[i], draw_edges[i + 1])

            nx.draw(G, pos, with_labels=True)
            plt.show()

# Visualización de resultados
plt.plot(min_distance)
plt.show()

plt.plot(mean_distance)
plt.show()

all_distances = [get_total_distance(parent, pos) for parent in current_population]
plt.hist(all_distances)
plt.show()

min_idx = np.where(fitness == fitness.max())[0][0]
draw_edges = current_population[min_idx].copy()
draw_edges.append(draw_edges[0])

G = nx.Graph()
for i in range(n_points):
    G.add_node(i)

for i in range(len(draw_edges) - 1):
    G.add_edge(draw_edges[i], draw_edges[i + 1])

nx.draw(G, pos, with_labels=True)
plt.show()

print(get_total_distance(current_population[min_idx], pos))

# Guarda datos en un archivo JSON
TSP = json.dumps(pos)
with open("./mapa_TSP.txt", "a") as file:
    resultado = json.dumps(TSP)
    file.write(resultado)
    file.write('\n')
