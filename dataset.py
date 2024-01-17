## --- Titolo: Import Dati e Costruzione Dataset
## --- Libraries ---
import pandas as pd
import numpy as np
import shutil # for images
from tqdm import tqdm

if __name__ == "__main__":

    ## --- Brain Waves ---
    df = pd.read_csv("./data/Onde_rosso_blue.csv")

    # la quarta riga è l'ultima sessione,
    # dati delle onde tra la colonna 27:32
    rosso = df.iloc[[4], 27:32]
    blue = df.iloc[[3], 27:32] # df con una riga sola

    # i dati sulle onde sono numpy darray che contengono un elemento che è una stringa
    def df_from_raw_waves(color_df):
        color_dict = {}
        for col in color_df.columns:
            print(col)
            # rosso[col] holds a ndarray; 
            # let's take the first and only element, which is a string, and convert it to list
            val = color_df[col].values[0][1:-2].split(", ") # [1:-2] to avoid brakets [...]
            key = "_".join(col.split("_")[:2]) # just take y_wave from y_wave_raw_data
            color_dict[key] = val # dict for dataframe

        color_df = pd.DataFrame(color_dict)
        return color_df

    rosso_df = df_from_raw_waves(rosso)
    blue_df = df_from_raw_waves(blue)
    # green_df  = df_from_raw_waves(green)
    # yellow_df = df_from_raw_waves(yellow)

    final_df = pd.concat([rosso_df, blue_df], ignore_index=False)
    final_df.to_csv("./data/waves/final_df.csv", header=False)

    print(final_df.head())
    print(final_df.columns)
    print(final_df.shape)

    """
    ## --- Images ---
    # number of images must equal number of brain-waves
    num_red_imgs = rosso_df.shape[0]
    num_blue_imgs = blue_df.shape[0]
    # num_green_imgs = green_df.shape[0]
    # num_yellow_imgs = yellow_df.shape[0]

    def duplicate_imgs(num_imgs, color: str):
        for i in tqdm(range(num_imgs)):
            shutil.copy2(src = "./data/images/" + color + '_square.png', dst = "./data/images/" + 'pic_' + str(i) + '.png')

    duplicate_imgs(num_red_imgs, "red")
    duplicate_imgs(num_blue_imgs, "blue")
    #duplicate_imgs(num_green_imgs, "green")
    #duplicate_imgs(num_yellow_imgs, "yellow")
    """