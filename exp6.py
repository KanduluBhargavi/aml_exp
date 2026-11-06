
import pandas as pd

# Step 1: Create dataset
students = ["A", "B", "C", "D", "E", "F"]

study_hours = [2, 4, 6, 8, 10, 12]

marks = [50, 60, 80, 70, 90, 100]

# Step 2: Create DataFrame
df = pd.DataFrame({
    "Student": students,
    "Study_Hours": study_hours,
    "Marks": marks
})

# Step 3: Calculate ranks
df["Rank X"] = df["Study_Hours"].rank(
    method="average", ascending=True
).astype(int)

df["Rank Y"] = df["Marks"].rank(
    method="average", ascending=True
).astype(int)

print("Original Data with Ranks:")
print(df)

# Step 4: Calculate difference between ranks
df["d"] = df["Rank X"] - df["Rank Y"]

print("\nRank Differences:")
print(df)

# Step 5: Calculate squared differences
df["d2"] = df["d"] ** 2

print("\nSquared Differences:")
print(df)

# Step 6: Calculate sum of d2
sum_d2 = df["d2"].sum()

print("\nSum of d2 =", sum_d2)

# Step 7: Calculate number of students
n = len(df)

print("Number of students (n) =", n)

# Step 8: Calculate Spearman correlation
rho = 1 - (6 * sum_d2) / (n * (n**2 - 1))

print("\nSpearman Rank Correlation Coefficient:")
print("rho =", round(rho, 3))

# Step 9: Interpretation
if rho > 0.9:
    print("Interpretation: Very strong positive correlation")

elif rho > 0.7:
    print("Interpretation: Strong positive correlation")

elif rho > 0.3:
    print("Interpretation: Moderate positive correlation")

elif rho > 0:
    print("Interpretation: Weak positive correlation")

elif rho == 0:
    print("Interpretation: No correlation")

else:
    print("Interpretation: Negative correlation")
