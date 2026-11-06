
import numpy as np
import pandas as pd
import math

# Step 1: Create contingency table
data = np.array([
    [8, 2],
    [5, 3],
    [1, 7]
])

# Step 2: Define rows and columns
rows = ["High", "Medium", "Low"]
columns = ["Pass", "Fail"]

# Step 3: Create DataFrame
df = pd.DataFrame(
    data,
    index=rows,
    columns=columns
)

print("Contingency Table:")
print(df)

# Step 4: Calculate total observations
total = data.sum()

print("\nTotal Observations:", total)

# Step 5: Calculate row and column totals
row_totals = data.sum(axis=1)

column_totals = data.sum(axis=0)

print("\nRow Totals:")
print(row_totals)

print("\nColumn Totals:")
print(column_totals)

# Step 6: Calculate marginal probabilities
print("\nMarginal Probabilities:")

for i in range(len(rows)):
    probability = row_totals[i] / total
    print(f"P({rows[i]}) = {probability:.3f}")

for j in range(len(columns)):
    probability = column_totals[j] / total
    print(f"P({columns[j]}) = {probability:.3f}")

# Step 7: Calculate Mutual Information
MI = 0

print("\nMutual Information Calculation:")

for i in range(len(rows)):
    for j in range(len(columns)):

        # Joint probability
        p_xy = data[i][j] / total

        # Marginal probabilities
        p_x = row_totals[i] / total
        p_y = column_totals[j] / total

        # Ratio
        ratio = p_xy / (p_x * p_y)

        # Contribution to mutual information
        contribution = p_xy * math.log2(ratio)

        MI = MI + contribution

        print(f"\nCell: ({rows[i]}, {columns[j]})")
        print(f"Frequency = {data[i][j]}")
        print(f"P(X,Y) = {p_xy:.3f}")
        print(f"P(X) = {p_x:.3f}")
        print(f"P(Y) = {p_y:.3f}")
        print(f"Ratio = {ratio:.3f}")
        print(f"Contribution = {contribution:.3f}")

# Step 8: Final result
print("\nFinal Result:")
print("Mutual Information =", round(MI, 3), "bits")
