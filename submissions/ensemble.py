import pandas as pd

sub_files = [
    'resnet-50_partial_freeze_best_fold0',
    'resnet-50_partial_freeze_model_fold0',
    'resnet-50_partial_freeze_extra_dataset_model_fold0',
    'resnet-50_partial_freeze_extra_dataset_best_fold0_epoch30',
    'resnet-50_partial_freeze_extra_dataset_best_fold0_epoch120'
]

dfs = [ pd.read_csv(f"./{file}_submission.csv") for file in sub_files ]

dd = {}
for unq in dfs[0].Emotion.unique():
    dd[unq] =0 

df = {
    "Frame_ID": [],
    "Emotion": []
}

for idx in range(dfs[0].shape[0]):
    d = dd.copy()
    for i in range(len(dfs)):
        d[dfs[i].values[idx][1]] +=1

    maxi = "angry"
    for unq in dfs[0].Emotion.unique():
        if d[maxi] < d[unq]:
            maxi = unq

    df["Frame_ID"].append(dfs[0].values[idx][0])
    df["Emotion"].append(maxi)

df = pd.DataFrame(df)
df.to_csv("ensemble.csv", index = False)
print(df.Emotion.value_counts())