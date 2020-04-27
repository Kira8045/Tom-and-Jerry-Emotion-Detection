import pandas as pd
import cv2
import os
import PIL
import random
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

data_dir = "./Dataset/"

df = pd.read_csv(f"{data_dir}/train_corrected.csv")

df.loc[df.Emotion == "suprised", "Emotion"] = "surprised"
df = df[df.Emotion != "Unknown"]
df = df[df.Frame_ID.isin(os.listdir(data_dir + "train"))].reset_index(drop = True)

dff = df.copy()
print(df.shape[0])
for clss in dff["Emotion"].unique():
    while dff[dff["Emotion"] == clss].shape[0] < 90:
        d = {
            "Frame_ID": [],
            "Emotion": []
        }

        imageFile, imageLabel = random.choice(df[df["Emotion"] == clss].values)
        im=Image.open(f"{data_dir}/train/" + imageFile)
        im=im.convert("RGB")
        r,g,b=im.split()
        r=r.convert("RGB")
        g=g.convert("RGB")
        b=b.convert("RGB")
        im_blur=im.filter(ImageFilter.GaussianBlur)
        im_unsharp=im.filter(ImageFilter.UnsharpMask)

        r.save(f"{data_dir}/train/" + 'r_'+imageFile)
        g.save(f"{data_dir}/train/" +'g_'+imageFile)
        b.save(f"{data_dir}/train/" +'b_'+imageFile)
        im_blur.save(f"{data_dir}/train/" +'bl_'+imageFile)
        im_unsharp.save(f"{data_dir}/train/" +'un_'+imageFile)

        d["Frame_ID"].append('r_'+imageFile)
        d["Emotion"].append(imageLabel)

        d["Frame_ID"].append('g_'+imageFile)
        d["Emotion"].append(imageLabel)

        d["Frame_ID"].append('b_'+imageFile)
        d["Emotion"].append(imageLabel)

        d["Frame_ID"].append('bl_'+imageFile)
        d["Emotion"].append(imageLabel)
        
        d["Frame_ID"].append('un_'+imageFile)
        d["Emotion"].append(imageLabel)
        
        
        dff = pd.concat([dff, pd.DataFrame(d)])

print(dff.shape[0])
dff.to_csv(f"{data_dir}/train_corrected_balanced.csv", index=False)