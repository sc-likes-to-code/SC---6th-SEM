'''

Imports

Cities

Distance

Create Route

Fitness function

Crossover

Mutation

Population 

Crossover

Mutation

Next Generation

Best Output

'''

# Imports
import random
import math

# Cities
cities = [(2, 3), (5, 6), (7, 2), (9, 3), (1, 8)]

# Distance between 2 cities
def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
    
# Create random Route
def create_route():
    route = list(range(len(cities)))
    random.shuffle(route)
    return route
    
# Fitness function
def route_distance(route):
    total = 0
    for i in range(len(route)):
        total += dist(cities[route[i]], cities[route[(i + 1) % len(route)]])
    return total

# Order Crossover
def crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))
    child = p1[a : b]
    
    for city in p2:
        if city not in child:
            child.append(city)
            
    return child
    
# Swap Mutation
def mutate(route):
    i, j = random.sample(range(len(route)), 2)
    route[i], route[j] = route[j], route[i]
    
# Initial Population
population = [create_route() for _ in range(20)]

# GA generations
for _ in range(200):
    
    # Select best routes
    population.sort(key = route_distance)
    new_pop = population[:10]
    
    # Generate children
    while len(new_pop) < 20:
        # Select parents
        p1, p2 = random.sample(new_pop, 2)
        
        # Crossover
        child = crossover(p1, p2)
        
        # Random Mutation
        if random.random() < 0.1:
            mutate(child)
            
        # Add child
        new_pop.append(child)
    
    population = new_pop

# Best solution
best = min(population, key = route_distance)

print("Best Route: ", best)
print("Distance: ", route_distance(best))
