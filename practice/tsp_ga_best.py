import random

# Distance matrix
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Initial Population
population = [
    [0,1,2,3],
    [0,2,1,3],
    [0,3,1,2],
    [0,1,3,2]
]

# Fitness Function
def fitness(path):
    cost = 0
    for i in range(len(path)-1):
        cost += graph[path[i]][path[i+1]]
    cost += graph[path[-1]][path[0]]
    return cost

# Selection (Best 2 Parents)
population.sort(key=fitness)
parent1 = population[0]
parent2 = population[1]

# Crossover
child = parent1[:2]
for city in parent2:
    if city not in child:
        child.append(city)

# Mutation (Swap two cities except starting city)
child[1], child[2] = child[2], child[1]

# New Generation
population.append(child)

# Best Solution
best = min(population, key=fitness)

print("Best Path:", best + [best[0]])
print("Minimum Cost:", fitness(best))