import pandas as pd

# 讀取原始 CSV
df = pd.read_csv("output.csv")

# 建立新欄位 ImageName
df["ImageName"] = df["Group"] + "_" + df["Sample ID"] + "_" + df["Image"]

# 建立新 DataFrame，只保留需要的欄位
df_new = df[["ImageName", "Size", "Annotation"]].copy()

# 統一修改 Size
df_new["Size"] = "224x224"

# 輸出成新的 CSV
df_new.to_csv("paddedimage.csv", index=False)
