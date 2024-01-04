import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("det_k_fold_results.csv")
df["epoch"] = df.model_path.apply(lambda x : int(x.split("/")[-1].split("_")[-1]))
df.tail(10)
df["set"] = df.label_path.apply(lambda x : x.split("/")[4].split("_")[0])
df["k_fold"] = df.model_path.apply(lambda x : "_".join(x.split("/")[3].split("___")[2].split("_")[:4]))
df["model"] = df.model_path.apply(lambda x : x.split("/")[3].split("___")[1])

df["dataset"] = df.label_path.apply(lambda x : "_".join(x.split("/")[3].split("_")[:-4]))

####################################################################################################################################

model, dataset = "ml_PP-OCRv3_det", "ai_hub_det_08_02_90"
model, dataset = "MobileNetV3_large_x0_5", "ai_hub_det_08_02_90"



df = df[(df["model"]==model)&(df["dataset"]==dataset)]

df = df.groupby(["model", "k_fold", "epoch", "set"]).mean()


df = df.reset_index()

train_df = df[df["set"] == "train"].reset_index().rename(columns={"precision":"train_precision"})

val_df = df[df["set"] == "val"].reset_index().rename(columns={"precision":"val_precision"})[["val_precision"]]

acc_df = pd.concat([train_df, val_df], axis=1)

acc_df.plot(x="epoch", y=["train_precision", "val_precision"]).get_figure().savefig("그래프.png")