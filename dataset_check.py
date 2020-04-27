import pandas as pd
from tqdm import tqdm
import cv2

df = pd.read_csv("./Dataset/Test.csv")

for i in tqdm(df.Frame_ID.values):
    image = cv2.imread(f"./Dataset/test_frames/{i}")
    if image is None:
        print(i)