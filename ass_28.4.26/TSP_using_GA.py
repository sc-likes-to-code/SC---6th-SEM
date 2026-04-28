'''
Q. Write a Python program to implement a Genetic Algorithm using Order Encoding 
for solving the Travelling Salesman Problem for any given sample dataset. 
'''

import random
import math

# list of cities (coordinates)
cities = [(0,0), (1,5), (5,2), (6,6), (8,3)]

# distance between two cities
def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

# total distance of a route (fitness function)
def route_distance(route):
    total = 0
    for i in range(len(route)):
        total += dist(cities[route[i]], cities[route[(i+1)%len(route)]])  # include return to start
    return total

# generate random route (permutation encoding)
def create_route():
    route = list(range(len(cities)))
    random.shuffle(route)
    return route

# crossover: combine two parents to create child
def crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))  # pick segment
    child = p1[a:b]  # take part from parent1
    for x in p2:
        if x not in child:
            child.append(x)  # fill remaining from parent2
    return child

# mutation: swap two cities
def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]

# initial population
population = [create_route() for _ in range(20)]

# genetic algorithm loop
for _ in range(200):
    population.sort(key=route_distance)  # select best
    new_pop = population[:10]  # keep top 50%
    
    while len(new_pop) < 20:
        p1, p2 = random.sample(new_pop, 2)  # pick parents
        child = crossover(p1, p2)  # generate child
        if random.random() < 0.1:
            mutate(child)  # apply mutation
        new_pop.append(child)
    
    population = new_pop  # next generation

# best solution
best = min(population, key=route_distance)

'''
Output:

Best Route: [4, 2, 0, 1, 3]
Distance: 22.35103276995244
'''
print("Best Route:", best)
print("Distance:", route_distance(best))
