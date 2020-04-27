import pandas as pd
import cv2
import glob
import random
dataset_dir = "./Dataset/"


if __name__ == "__main__":
    df_train = pd.read_csv(dataset_dir + "train_corrected.csv")
    df_test = pd.read_csv(dataset_dir + "Test.csv")
    box = pd.read_csv(dataset_dir + "box_coordinates.csv")
    df_train["count"] = 0
    df_test["count"] = 0
    for mode in ["train", "test"]:
        print(mode)
        x =0
        if mode == "train":
            df = df_train
            b = box[box.img_name.isin(df.Frame_ID.values)]
            
        else:
            df = df_test
            b = box[box.img_name.isin(df.Frame_ID.values)]


        if mode == "train":
            save_dir = dataset_dir + "face_extracted/train/"
        else:
            save_dir = dataset_dir + "face_extracted/test/"
        for c in b.values:
            if c[5] ==-1:
                continue
            # print(c[0])
            if mode == "train":
                img = cv2.imread(dataset_dir + f"{mode}_frames/{c[0]}")
            else:
                img = cv2.imread(dataset_dir + f"test_frames/{c[0]}")

            img = cv2.resize(img, (300,300),interpolation = cv2.INTER_AREA)
            img = img[ c[1]:c[2], c[3]:c[4]]
            # print(img.shape)
            img = cv2.resize(img, (128,128),interpolation = cv2.INTER_AREA)
            cv2.imwrite(save_dir + f"{c[0].split('.')[0]}.jpg", img)
            df.loc[df.Frame_ID == c[0], "count"] = df.loc[df.Frame_ID == c[0], "count"] + 1
            x+=1
        df.to_csv(dataset_dir + f"{mode}_count.csv")
    
        
