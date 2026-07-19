weights = [2, 3, 5, 7, 1]
profits = [10, 5, 15, 7, 6]

capacity = 10

chromosome = input("Enter binary chromosome(5 bits): ")

total_weight = 0
total_profit = 0

for i in range(len(chromosome)):
    if (chromosome[i] == '1'):
        total_weight += weights[i]
        total_profit += profits[i]
        
print("\nSelected items:\n")

for i in range(len(chromosome)):
    if (chromosome[i] == '1'):
        print("Item", i+1)
        
print("\nTotal Weight =", total_weight)
print("Total Profit =", total_profit)

if total_weight <= capacity:
    print("\nValid solution")
else:
    print("\nInvalid solution(capacity exceeded)")