
# Two-Way ANOVA without using external libraries

# Dataset
# Factor A = Teaching Method
# Factor B = Gender
# Dependent Variable = Exam Score

data = {
    "Online_Male": [70, 75, 72, 68],
    "Offline_Male": [80, 85, 82, 78],
    "Online_Female": [75, 78, 76, 74],
    "Offline_Female": [85, 88, 86, 84]
}

# Number of levels
a = 2       # Teaching methods
b = 2       # Gender
n = 4       # Observations per combination

# Arrange data into groups
groups = [
    data["Online_Male"],
    data["Offline_Male"],
    data["Online_Female"],
    data["Offline_Female"]
]

# 1. Calculate Grand Mean
all_values = []

for group in groups:
    all_values.extend(group)

N = len(all_values)
grand_mean = sum(all_values) / N

print("Grand Mean:", grand_mean)

# 2. Calculate Group Means
group_means = []

for group in groups:
    mean = sum(group) / len(group)
    group_means.append(mean)

print("\nGroup Means:", group_means)

# 3. Calculate Factor A Means (Teaching Method)
online_mean = sum(
    data["Online_Male"] + data["Online_Female"]
) / (2 * n)

offline_mean = sum(
    data["Offline_Male"] + data["Offline_Female"]
) / (2 * n)

factor_A_means = [online_mean, offline_mean]

print("\nFactor A Means:", factor_A_means)

# 4. Calculate Factor B Means (Gender)
male_mean = sum(
    data["Online_Male"] + data["Offline_Male"]
) / (2 * n)

female_mean = sum(
    data["Online_Female"] + data["Offline_Female"]
) / (2 * n)

factor_B_means = [male_mean, female_mean]

print("\nFactor B Means:", factor_B_means)

# 5. Calculate Sum of Squares for Factor A
SSA = b * n * sum(
    (mean - grand_mean) ** 2
    for mean in factor_A_means
)

# 6. Calculate Sum of Squares for Factor B
SSB = a * n * sum(
    (mean - grand_mean) ** 2
    for mean in factor_B_means
)

# 7. Calculate Sum of Squares Within Groups (Error)
SSE = 0

for i, group in enumerate(groups):
    for value in group:
        SSE += (value - group_means[i]) ** 2

# 8. Calculate Interaction Sum of Squares
SSAB = 0

for i in range(a):
    for j in range(b):

        cell_mean = group_means[i * b + j]

        interaction = (
            cell_mean
            - factor_A_means[i]
            - factor_B_means[j]
            + grand_mean
        )

        SSAB += n * interaction ** 2

# 9. Calculate Total Sum of Squares
SST = sum(
    (value - grand_mean) ** 2
    for value in all_values
)

print("\nSum of Squares:")
print("SSA (Teaching Method):", SSA)
print("SSB (Gender):", SSB)
print("SSAB (Interaction):", SSAB)
print("SSE (Error):", SSE)
print("SST (Total):", SST)

# 10. Calculate Degrees of Freedom
df_A = a - 1
df_B = b - 1
df_AB = (a - 1) * (b - 1)
df_E = a * b * (n - 1)
df_T = N - 1

print("\nDegrees of Freedom:")
print("Factor A:", df_A)
print("Factor B:", df_B)
print("Interaction:", df_AB)
print("Error:", df_E)
print("Total:", df_T)

# 11. Calculate Mean Squares
MSA = SSA / df_A
MSB = SSB / df_B
MSAB = SSAB / df_AB
MSE = SSE / df_E

print("\nMean Squares:")
print("MSA:", MSA)
print("MSB:", MSB)
print("MSAB:", MSAB)
print("MSE:", MSE)

# 12. Calculate F Statistics
FA = MSA / MSE
FB = MSB / MSE
FAB = MSAB / MSE

print("\nF Statistics:")
print("F for Teaching Method:", FA)
print("F for Gender:", FB)
print("F for Interaction:", FAB)

# 13. Final ANOVA Table
print("\nTWO-WAY ANOVA TABLE")
print("-" * 75)
print(f"{'Source':<22}{'SS':<12}{'DF':<8}{'MS':<12}{'F-value':<12}")
print("-" * 75)

print(f"{'Teaching Method':<22}{SSA:<12.2f}{df_A:<8}{MSA:<12.2f}{FA:<12.2f}")
print(f"{'Gender':<22}{SSB:<12.2f}{df_B:<8}{MSB:<12.2f}{FB:<12.2f}")
print(f"{'Interaction':<22}{SSAB:<12.2f}{df_AB:<8}{MSAB:<12.2f}{FAB:<12.2f}")
print(f"{'Error':<22}{SSE:<12.2f}{df_E:<8}{MSE:<12.2f}")
print(f"{'Total':<22}{SST:<12.2f}{df_T:<8}")
print("-" * 75)

print("\nANOVA calculation completed!")
