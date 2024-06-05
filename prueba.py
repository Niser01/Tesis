#implementación del mapa gráfico (actualizar por un mapa en colombia o de la zona donde va a ser utulizado)
#se incluye la representación visual de las ciudades (campamentos)
# y de las conexiones entre ellas

import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

united_states_map = mpimg.imread("input/united_states_map.png")
def show_cities(path, w=12, h=8):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    if isinstance(path, dict):      path = list(path.values())
    if isinstance(path[0][0], str): path = [ item[1] for item in path ]    
    plt.imshow(united_states_map)    
    for x0, y0 in path:
        plt.plot(x0, y0, 'y*', markersize=15)  # y* = yellow star for starting point        
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])

   
def show_path(path, starting_city=None, w=12, h=8):
    """Plot a TSP path overlaid on a map of the US States & their capitals."""
    if isinstance(path, dict):      path = list(path.values())
    if isinstance(path[0][0], str): path = [ item[1] for item in path ]
    
    starting_city = starting_city or path[0]
    x, y = list(zip(*path))
    #_, (x0, y0) = starting_city
    (x0, y0) = starting_city
    plt.imshow(united_states_map)
    plt.plot(x0, y0, 'y*', markersize=5)  # y* = yellow star for starting point
    plt.plot(x + x[:1], y + y[:1])  # include the starting point at the end of path
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches([w, h])
    
    
def polyfit_plot(x,y,deg, **kwargs):
    coefficients = np.polyfit(x,y,deg,**kwargs)
    poly  = np.poly1d(coefficients)
    new_x = np.linspace(x[0], x[-1])
    new_y = poly(new_x)
    plt.plot(x, y, "o", new_x, new_y)
    plt.xlim([x[0]-1, x[-1] + 1 ])
    
    terms = []
    for p, c in enumerate(reversed(coefficients)):
        term = str(round(c,1))
        if p == 1: term += 'x'
        if p >= 2: term += 'x^'+str(p)
        terms.append(term)        
    plt.title(" + ".join(reversed(terms)))   

    

def distance(xy1, xy2) -> float:
    if isinstance(xy1[0], str): xy1 = xy1[1]; xy2 = xy2[1];               # if xy1 == ("Name", (x,y))
    return math.sqrt( (xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2 )

def path_distance(path) -> int:
    if isinstance(path, dict):      path = list(path.values())            # if path == {"Name": (x,y)}
    if isinstance(path[0][0], str): path = [ item[1] for item in path ]   # if path == ("Name", (x,y))
    return int(sum(
        [ distance(path[i],  path[i+1]) for i in range(len(path)-1) ]
      + [ distance(path[-1], path[0]) ]                                   # include cost of return journey
    ))

import csv

cities = {}

with open('input/Factor_Distancia.csv', newline='', encoding='utf-8-sig') as csvfile:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        city = row['city']
        x = float(row['x'].replace(',', '.'))
        y = float(row['y'].replace(',', '.'))
        cities[city] = (x, y)

cities = list(sorted(cities.items()))
print((cities))
show_cities(cities)

#muestra el camino 

show_path(cities)
path_distance(cities)

import pandas as pd

file_path = 'input/Factor_Inclinacion.csv'  
df = pd.read_csv(file_path, delimiter=';')

df.set_index('cities', inplace=True)


def get_inclination(city1, city2):
    try:
        inclination = df.at[city1, city2]
        return inclination
    except KeyError:
        return None

city1 = 'Harrisburg'
city2 = 'Boston'
inclination = get_inclination(city1, city2)
if inclination is not None:
    print(f'La inclinación entre {city1} y {city2} es {inclination}.')
else:
    print(f'No se encontró la inclinación entre {city1} y {city2}.')


#Factor terreno

descripcion_tipos_de_terreno = {
    1: "Asfalto: Superficie pavimentada común en carreteras y calles urbanas.",
    2: "Grava: Pequeñas piedras sueltas, a menudo utilizadas en caminos rurales.",
    3: "Tierra: Camino de suelo desnudo, puede estar compactado o suelto.",
    4: "Arena: Terreno arenoso, común en áreas costeras y desiertos.",
    5: "Barro: Terreno fangoso, especialmente después de lluvias.",
    6: "Roca: Superficies rocosas, que pueden ser lisas o irregulares.",
    7: "Hierba: Áreas cubiertas de césped o pasto.",
    8: "Nieve: Terreno cubierto de nieve, común en áreas montañosas durante el invierno.",
    9: "Hielo: Superficie congelada, muy resbaladiza.",
    10: "Bosque: Terreno boscoso, con raíces de árboles y vegetación densa.",
    11: "Desierto: Terreno árido y seco, con poca vegetación.",
    12: "Pantano: Área húmeda y lodosa, con agua estancada y vegetación específica.",
    13: "Colinas: Terreno ondulado, con subidas y bajadas.",
    14: "Montaña: Áreas elevadas con terrenos rocosos y empinados.",
    15: "Valle: Áreas bajas entre colinas o montañas, a menudo con un río o arroyo.",
    16: "Pradera: Áreas planas o ligeramente onduladas, con pastos y pocos árboles.",
    17: "Caminos de adoquines: Superficie de piedras pequeñas, común en áreas urbanas antiguas.",
    18: "Senderos de montaña: Caminos estrechos y empinados, a menudo con terreno irregular.",
    19: "Caminos de grava compactada: Superficie de grava apisonada, más estable que la grava suelta.",
    20: "Terreno agrícola: Campos de cultivo, que pueden variar en textura y estabilidad."
}

dificultad_tipos_de_terreno = {
    1: 1,  # Asfalto
    2: 2,  # Grava
    3: 2,  # Tierra
    4: 3,  # Arena
    5: 4,  # Barro
    6: 3,  # Roca
    7: 1,  # Hierba
    8: 4,  # Nieve
    9: 5,  # Hielo
    10: 3,  # Bosque
    11: 3,  # Desierto
    12: 5,  # Pantano
    13: 3,  # Colinas
    14: 4,  # Montaña
    15: 2,  # Valle
    16: 1,  # Pradera
    17: 2,  # Caminos de adoquines
    18: 4,  # Senderos de montaña
    19: 2,  # Caminos de grava compactada
    20: 2   # Terreno agrícola
}



import pandas as pd

file_path = 'input/Factor_Terreno.csv'  
df = pd.read_csv(file_path, delimiter=';')

df.set_index('cities', inplace=True)


def get_terreno(city1, city2):
    
    descripcion_tipos_de_terreno = {
    1: "Asfalto: Superficie pavimentada común en carreteras y calles urbanas.",
    2: "Grava: Pequeñas piedras sueltas, a menudo utilizadas en caminos rurales.",
    3: "Tierra: Camino de suelo desnudo, puede estar compactado o suelto.",
    4: "Arena: Terreno arenoso, común en áreas costeras y desiertos.",
    5: "Barro: Terreno fangoso, especialmente después de lluvias.",
    6: "Roca: Superficies rocosas, que pueden ser lisas o irregulares.",
    7: "Hierba: Áreas cubiertas de césped o pasto.",
    8: "Nieve: Terreno cubierto de nieve, común en áreas montañosas durante el invierno.",
    9: "Hielo: Superficie congelada, muy resbaladiza.",
    10: "Bosque: Terreno boscoso, con raíces de árboles y vegetación densa.",
    11: "Desierto: Terreno árido y seco, con poca vegetación.",
    12: "Pantano: Área húmeda y lodosa, con agua estancada y vegetación específica.",
    13: "Colinas: Terreno ondulado, con subidas y bajadas.",
    14: "Montaña: Áreas elevadas con terrenos rocosos y empinados.",
    15: "Valle: Áreas bajas entre colinas o montañas, a menudo con un río o arroyo.",
    16: "Pradera: Áreas planas o ligeramente onduladas, con pastos y pocos árboles.",
    17: "Caminos de adoquines: Superficie de piedras pequeñas, común en áreas urbanas antiguas.",
    18: "Senderos de montaña: Caminos estrechos y empinados, a menudo con terreno irregular.",
    19: "Caminos de grava compactada: Superficie de grava apisonada, más estable que la grava suelta.",
    20: "Terreno agrícola: Campos de cultivo, que pueden variar en textura y estabilidad."
    }

    dificultad_tipos_de_terreno = {
        1: 1,  # Asfalto
        2: 2,  # Grava
        3: 2,  # Tierra
        4: 3,  # Arena
        5: 4,  # Barro
        6: 3,  # Roca
        7: 1,  # Hierba
        8: 4,  # Nieve
        9: 5,  # Hielo
        10: 3,  # Bosque
        11: 3,  # Desierto
        12: 5,  # Pantano
        13: 3,  # Colinas
        14: 4,  # Montaña
        15: 2,  # Valle
        16: 1,  # Pradera
        17: 2,  # Caminos de adoquines
        18: 4,  # Senderos de montaña
        19: 2,  # Caminos de grava compactada
        20: 2   # Terreno agrícola
    }
    try:
        terreno = df.at[city1, city2]
        descripcion_terreno = descripcion_tipos_de_terreno.get(terreno, "Tipo de terreno no encontrado.")
        dificlutad_terreno = dificultad_tipos_de_terreno.get(terreno, "Tipo de terreno no encontrado.")
        return dificlutad_terreno
    except KeyError:
        return None

print(type(get_terreno('Harrisburg', 'Boston')))

def get_city_item(city_name, cities):

  cities_dict = {city[0]: city for city in cities}


  if city_name in cities_dict:
    return cities_dict[city_name]
  else:

    raise ValueError(f"City '{city_name}' not found in the list.")


city_name = "Boston"
city_item = get_city_item(city_name, cities)
print(f"City item for '{city_name}': {city_item}")


class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, generations, pheromone_power):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.pheromone_power = pheromone_power

    def initialize_population(self):
        return [
            {
                "Fact_dist": random.uniform(0, 1),
                "Fact_incli": random.uniform(0, 1),
                "Fact_Terr": random.uniform(0, 1)
            }
            for _ in range(self.population_size)
        ]

    def fitness(self, weights, ants, index, distances, pheromones):
        total_cost = 0
        this_node = ants['path'][index][-1]


        for next_node in ants['remaining'][index]:

            distancia = distances[this_node][next_node]
            inclinacion = get_inclination(this_node[0], next_node[0])
            dificultad = get_terreno(this_node[0], next_node[0])

            print(this_node, next_node)
            print(weights["Fact_dist"])
            print(weights["Fact_incli"])
            print(weights["Fact_Terr"])


            reward = (
                pheromones[this_node][next_node] ** self.pheromone_power
                * ((weights["Fact_dist"] * distancia) + (weights["Fact_incli"] * inclinacion) + (weights["Fact_Terr"] * dificultad))
            )
            total_cost += reward
        return 1 / total_cost  

    def select_parents(self, population, fitnesses):
        selected = random.choices(population, weights=fitnesses, k=2)
        return selected[0], selected[1]

    def crossover(self, parent1, parent2):
        child = {}
        for key in parent1.keys():
            if random.random() > 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        return child

    def mutate(self, child):
        for key in child.keys():
            if random.random() < self.mutation_rate:
                child[key] = random.uniform(0, 1)
        return child

    def evolve(self, ants, index, distances, pheromones):
        population = self.initialize_population()
        for generation in range(self.generations):
            fitnesses = [self.fitness(individual, ants, index, distances, pheromones) for individual in population]
            new_population = []
            for _ in range(self.population_size):
                parent1, parent2 = self.select_parents(population, fitnesses)
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                new_population.append(child)
            population = new_population
        best_individual = max(population, key=lambda ind: self.fitness(ind, ants, index, distances, pheromones))
        return best_individual


import time
from itertools import chain
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import random



class AntColonySolver:
    def __init__(self,
                 cost_fn:                 Callable[[Any,Any], Union[float,int]],                         
                 
                 time=0,                  # run for a fixed amount of time
                 min_time=0,              # minimum runtime
                 timeout=0,               # maximum time in seconds to run for
                 stop_factor=2,           # how many times to redouble effort after new new best path
                 min_round_trips=10,      # minimum number of round trips before stopping
                 max_round_trips=0,       # maximum number of round trips before stopping                 
                 min_ants=0,              # Total number of ants to use
                 max_ants=0,              # Total number of ants to use
                 
                 ant_count=64,            # this is the bottom of the near-optimal range for numpy performance
                 ant_speed=1,             # how many steps do ants travel per epoch

                 distance_power=1,        # power to which distance affects pheromones                 
                 pheromone_power=1.25,    # power to which differences in pheromones are noticed
                 decay_power=0,           # how fast do pheromones decay
                 reward_power=0,          # relative pheromone reward based on best_path_length/path_length 
                 best_path_smell=2,       # queen multiplier for pheromones upon finding a new best path                  
                 start_smell=0,           # amount of starting pheromones [0 defaults to `10**self.distance_power`]



                 verbose=False,

    ):
        assert callable(cost_fn)        
        self.cost_fn         = cost_fn
        self.time            = int(time)
        self.min_time        = int(min_time)
        self.timeout         = int(timeout)
        self.stop_factor     = float(stop_factor)
        self.min_round_trips = int(min_round_trips)
        self.max_round_trips = int(max_round_trips)
        self.min_ants        = int(min_ants)
        self.max_ants        = int(max_ants)
    
        self.ant_count       = int(ant_count)
        self.ant_speed       = int(ant_speed)
        
        self.distance_power  = float(distance_power)     
        self.pheromone_power = float(pheromone_power)
        self.decay_power     = float(decay_power)
        self.reward_power    = float(reward_power)
        self.best_path_smell = float(best_path_smell)
        self.start_smell     = float(start_smell or 10**self.distance_power)


        
        self.verbose         = int(verbose)
        self._initalized     = False
        
        if self.min_round_trips and self.max_round_trips: self.min_round_trips = min(self.min_round_trips, self.max_round_trips)
        if self.min_ants and self.max_ants:               self.min_ants        = min(self.min_ants, self.max_ants)


    def solve_initialize(
            self,
            problem_path: List[Any],
    ) -> None:

        self.distances = {
            source: {
                dest: self.cost_fn(source, dest)
                for dest in problem_path
            }
            for source in problem_path
        }


        self.distance_cost = {
            source: {
                dest: 1 / (1 + self.distances[source][dest]) ** self.distance_power
                for dest in problem_path
            }
            for source in problem_path
        }


        self.pheromones = {
            source: {

                dest: self.start_smell
                for dest in problem_path
            }
            for source in problem_path
        }
        

        if self.ant_count <= 0:
            self.ant_count = len(problem_path)
        if self.ant_speed <= 0:
            self.ant_speed = np.median(list(chain(*[ d.values() for d in self.distances.values() ]))) // 5
        self.ant_speed = int(max(1,self.ant_speed))
        

        self.ants_used   = 0
        self.epochs_used = 0
        self.round_trips = 0
        self._initalized = True        


    def solve(self,
              problem_path: List[Any],
              restart=False,
    ) -> List[Tuple[int,int]]:
        if restart or not self._initalized:
            self.solve_initialize(problem_path)


        ants = {
            "distance":    np.zeros((self.ant_count,)).astype('int32'),
            "path":        [ [ problem_path[0] ]   for n in range(self.ant_count) ],
            "remaining":   [ set(problem_path[1:]) for n in range(self.ant_count) ],
            "path_cost":   np.zeros((self.ant_count,)).astype('int32'),
            "round_trips": np.zeros((self.ant_count,)).astype('int32'),
        }

        best_path       = None
        best_path_cost  = np.inf
        best_epochs     = []
        epoch           = 0
        time_start      = time.perf_counter()

        while True:
            epoch += 1

            ants_travelling = (ants['distance'] > self.ant_speed)
            ants['distance'][ ants_travelling ] -= self.ant_speed
            if all(ants_travelling):
                continue  
            
            ants_arriving       = np.invert(ants_travelling)
            ants_arriving_index = np.where(ants_arriving)[0]
            for i in ants_arriving_index:

                this_node = ants['path'][i][-1]
                next_node = self.next_node(ants, i)
                ants['distance'][i]  = self.distances[ this_node ][ next_node ]
                ants['remaining'][i] = ants['remaining'][i] - {this_node}
                ants['path_cost'][i] = ants['path_cost'][i] + ants['distance'][i]
                ants['path'][i].append( next_node )

                if not ants['remaining'][i] and ants['path'][i][0] == ants['path'][i][-1]:
                    self.ants_used  += 1
                    self.round_trips = max(self.round_trips, ants["round_trips"][i] + 1)

                    was_best_path = False
                    if ants['path_cost'][i] < best_path_cost:
                        was_best_path  = True
                        best_path_cost = ants['path_cost'][i]
                        best_path      = ants['path'][i]
                        best_epochs   += [ epoch ]
                        if self.verbose:
                            print({
                                "path_cost":   int(ants['path_cost'][i]),
                                "ants_used":   self.ants_used,
                                "epoch":       epoch,
                                "round_trips": ants['round_trips'][i] + 1,
                                "clock":       int(time.perf_counter() - time_start),
                            })

                    reward = 1
                    if self.reward_power: reward *= ((best_path_cost / ants['path_cost'][i]) ** self.reward_power)
                    if self.decay_power:  reward *= (self.round_trips ** self.decay_power)
                    for path_index in range( len(ants['path'][i]) - 1 ):
                        this_node = ants['path'][i][path_index]
                        next_node = ants['path'][i][path_index+1]
                        self.pheromones[this_node][next_node] += reward
                        self.pheromones[next_node][this_node] += reward
                        if was_best_path:
                            # Queen orders to double the number of ants following this new best path                            
                            self.pheromones[this_node][next_node] *= self.best_path_smell
                            self.pheromones[next_node][this_node] *= self.best_path_smell

                    ### reset ant
                    ants["distance"][i]     = 0
                    ants["path"][i]         = [ problem_path[0] ]
                    ants["remaining"][i]    = set(problem_path[1:])
                    ants["path_cost"][i]    = 0
                    ants["round_trips"][i] += 1

            if not len(best_epochs): 
                continue 
            

            if self.time or self.min_time or self.timeout:
                clock = time.perf_counter() - time_start
                if self.time:
                    if clock > self.time: 
                        break
                    else:                 
                        continue
                if self.min_time and clock < self.min_time: 
                    continue
                if self.timeout  and clock > self.timeout:  
                    break
            

            if self.min_round_trips and self.round_trips <  self.min_round_trips: 
                continue        
            if self.max_round_trips and self.round_trips >= self.max_round_trips:
                break

          
            if self.min_ants and self.ants_used <  self.min_ants: 
                continue        
            if self.max_ants and self.ants_used >= self.max_ants: 
                break            
            
            if self.stop_factor and epoch > (best_epochs[-1] * self.stop_factor): 
                break
                                
            if True: 
                continue
                                    
        self.epochs_used = epoch
        self.round_trips = np.max(ants["round_trips"])
        return best_path




    def next_node(self, ants, index):
        # Parámetros del algoritmo genético
        population_size = 50
        mutation_rate = 0.1
        generations = 100

        # Inicializa el algoritmo genético
        ga = GeneticAlgorithm(population_size, mutation_rate, generations, self.pheromone_power)

        # Encuentra los mejores pesos usando el algoritmo genético
        best_weights = ga.evolve(ants, index, self.distances, self.pheromones)

        this_node = ants['path'][index][-1]
        weights = []
        weights_sum = 0

        if not ants['remaining'][index]: 
            return ants['path'][index][0]

        for next_node in ants['remaining'][index]:
            distancia = self.distances[get_city_item(this_node[0], cities)][get_city_item(next_node[0], cities)]
            inclinacion = get_inclination(this_node[0], next_node[0])
            dificultad = get_terreno(this_node[0], next_node[0])

            if next_node == this_node: 
                continue
            
            # Validar valores devueltos
            if distancia is None or inclinacion is None or dificultad is None:
                raise ValueError("Uno de los valores devueltos es None")

            reward = (
                self.pheromones[this_node][next_node] ** self.pheromone_power
                * ((best_weights["Fact_dist"] * distancia) + (best_weights["Fact_incli"] * inclinacion) + (best_weights["Fact_Terr"] * dificultad))
            )

            weights.append((reward, next_node))
            weights_sum += reward

        rand = random.random() * weights_sum
        for (weight, next_node) in weights:
            if rand > weight:
                rand -= weight
            else:
                break
        return next_node


def AntColonyRunner(cities, verbose=False, plot=False, label={}, algorithm=AntColonySolver, **kwargs):
    solver     = algorithm(cost_fn=distance, verbose=verbose, **kwargs)
    start_time = time.perf_counter()
    result     = solver.solve(cities)
    stop_time  = time.perf_counter()
    if label: kwargs = { **label, **kwargs }
        
    for key in ['verbose', 'plot', 'animate', 'label', 'min_time', 'max_time']:
        if key in kwargs: del kwargs[key]
    print("N={:<3d} | {:5.0f} -> {:4.0f} | {:4.0f}s | ants: {:5d} | trips: {:4d} | "
          .format(len(cities), path_distance(cities), path_distance(result), (stop_time - start_time), solver.ants_used, solver.round_trips)
          + " ".join([ f"{k}={v}" for k,v in kwargs.items() ])
    )
    if plot:
        show_path(result)
    return result


results = AntColonyRunner(cities, distance_power=0, min_time=30, verbose=True, plot=True)
print()
print(results)    