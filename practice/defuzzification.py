# Number of elements
n = int(input("Enter number of elements: "))

x = []
mu = []

print("Enter element and membership value:")

for i in range(n):
    element = float(input(f"Element {i+1}: "))
    membership = float(input(f"Membership value of {element}: "))

    x.append(element)
    mu.append(membership)

# -------------------------
# Centroid of Area (COA)
# -------------------------
numerator = 0
denominator = 0

for i in range(n):
    numerator += x[i] * mu[i]
    denominator += mu[i]

centroid = numerator / denominator

# -------------------------
# Bisector of Area (BOA)
# -------------------------
total_area = sum(mu)
half_area = total_area / 2

current_area = 0

for i in range(n):
    current_area += mu[i]

    if current_area >= half_area:
        bisector = x[i]
        break

# -------------------------
# MOM and SOM
# -------------------------
max_mu = max(mu)

max_positions = []

for i in range(n):
    if mu[i] == max_mu:
        max_positions.append(x[i])

mom = sum(max_positions) / len(max_positions)

som = min(max_positions)

# -------------------------
# Output
# -------------------------
print("\nDefuzzification Results")
print("-----------------------")
print("Centroid of Area (COA) =", round(centroid, 4))
print("Bisector of Area (BOA) =", bisector)
print("Mean of Maximum (MOM) =", mom)
print("Smallest of Maximum (SOM) =", som)