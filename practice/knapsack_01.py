'''

Input
↓
Create Chromosome
↓
Fitness
↓
Crossover
↓
Mutation
↓
Initial Population
↓
GA Loop
↓
Print Best

'''

import random

# input
weights = [2, 3, 4, 5]
values = [3, 4, 5, 6]
capacity = 5

population_size = 4
generations = 10
mutation_rate = 0.1

# create random chromosome
def create_chromosome():
    return [random.randint(0, 1) for _ in range(len(weights))]
    
# fitness function
def fitness(chromosome):
    total_weight = 0
    total_value = 0
    
    for i in range(len(chromosome)):
        if (chromosome[i] == 1):
            total_weight += weights[i]
            total_value += values[i]
    
    if total_weight > capacity:
        return 0
        
    return total_value
    
# one-point crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    child = parent1[:point] + parent2[point:]
    return child
    
# mutation
def mutate(chromosome):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
            
    return chromosome
    
# initial population
population = [create_chromosome() for _ in range(population_size)]

# GA loop
for _ in range(generations):
    population = sorted(population, key = fitness, reverse = True)
    
    new_population = population[:2] # best 2 survive
    
    while len(new_population) < population_size:
        child = crossover(population[0], population[1])
        child = mutate(child)
        new_population.append(child)
        
    population = new_population

best = max(population, key = fitness)

print("Best Solution: ", best)
print("Maximum value: ", fitness(best))
