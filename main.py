import pandas as pd

df = pd.read_csv("seeds_pre.csv")
print(df.head())
sample = df.sample(120)
sample = sample.drop(["Class", "Aspect_Ration"], axis=1)
sample["Area"] = pd.to_numeric(sample["Area"], downcast="float")
sample["Convex_Area"] = pd.to_numeric(sample["Convex_Area"], downcast="float")
print(sample.head())
sample.to_csv("seeds_post.csv", float_format="%12.4f", index=False)
