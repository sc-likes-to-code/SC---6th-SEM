def triangular(x, a, b, c):
    if x <= a or x >= c:
        return 0
    
    elif x == b:
        return 1
    
    elif x < b:
        return (x - a) / (b - a)
    
    else:
        return (c - x) / (c - b)
    
bp = float(input("Enter Blood Pressure: "))
temp = float(input("Enter Temperature (F): "))

# BP Memberships
bp_low = triangular(bp, 80, 90, 100)
bp_normal = triangular(bp, 90, 120, 140)
bp_high = triangular(bp, 130, 160, 180)

# Temperature Memberships
temp_normal = triangular(temp, 97, 98.6, 100)
temp_high = triangular(temp, 99, 101, 104)

rule1 = min(bp_high, temp_high)
rule2 = min(bp_normal, temp_normal)
rule3 = min(bp_low, temp_normal)

max_rule = max(rule1, rule2, rule3)

if (max_rule == rule1):
    health = "Poor"
    
elif (max_rule == rule2):
    health = "Good"
    
else:
    health = "Normal"
    
# Output
print("\nMembership Values")
print("BP Low =", round(bp_low, 3))
print("BP Normal =", round(bp_normal, 3))
print("BP High =", round(bp_high, 3))

print("Temperature Normal =", round(temp_normal, 3))
print("Temperature High =", round(temp_high, 3))

print("\nRule Strengths")
print("Rule 1 (Poor) =", round(rule1, 3))
print("Rule 2 (Good) =", round(rule2, 3))
print("Rule 3 (Normal) =", round(rule3, 3))

print("\nHealth Condition =", health)