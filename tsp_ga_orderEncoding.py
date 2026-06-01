import random
import math

cities = [(0,0), (1,5), (5,2), (6,6), (8,3)]

def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def create_route():
    route = list(range(len(cities)))
    random.shuffle(route)
    return route

def route_distance(route):
    total = 0
    for i in range(len(route)):
        total += dist(cities[route[i]], cities[route[(i+1)%len(route)]])
    return total

def crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = p1[a:b]

    for city in p2:
        if city not in child:
            child.append(city)

    return child

def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]

population = [create_route() for _ in range(20)]

for _ in range(200):
    population.sort(key=route_distance)
    new_pop = population[:10]
    
    while len(new_pop) < 20:
        p1, p2 = random.sample(new_pop, 2)
        child = crossover(p1, p2)
        if random.random() < 0.1:
            mutate(child)
        new_pop.append(child)
    population = new_pop

best = min(population, key=route_distance)

print("Best Route:", best)
print("Distance:", route_distance(best))
