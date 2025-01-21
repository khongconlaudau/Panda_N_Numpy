import pandas as pd

# DataFrame
df = pd.DataFrame({
    "Name": [
        "Duc Manh Van",
        "Duc Manh ",
        "Mason Van",
    ],
    "Age": [20, 21, 22],
    "Gender": ["Male", "Male"," Male"],
})

print(df)
print()
print(df["Age"])
# Function with DF
print("The Greatest Age")
print(df["Age"].max())
print()
print(df.describe())
#      Age
# count   3.0
# mean   21.0
# std     1.0  Standard Deviation
# min    20.0
# 25%    20.5
# 50%    21.0
# 75%    21.5
# max    22.0
print("\n\n\n\n")

# A Series treats as a column of df


ages = pd.Series([20,21,22], name="Age")
print(ages)
# Function with Series
print("The Greatest Age")
print(ages.max())