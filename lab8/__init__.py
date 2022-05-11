import pandas as pd


df = pd.read_csv("data_packed_long.csv")

df["Area"] = df["Area"] / 39.37
df["Perimeter"] = df["Perimeter"] / 39.37
df["Major_Axis_Length"] = df["Major_Axis_Length"] / 39.37
df["Minor_Axis_Length"] = df["Minor_Axis_Length"] / 39.37

df.to_csv("mm_data.csv", index=False)