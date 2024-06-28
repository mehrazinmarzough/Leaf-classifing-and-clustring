import os
import pandas as pd
from PIL import Image


def find_folder(p: str, first_char: int):
    for entry in os.listdir(p):
        a = int(first_char / 10)
        b = first_char % 10
        if a == 0:
            if os.path.isdir(os.path.join(p, entry)) and entry[0] == str(b):
                return os.path.join(p, entry)  # Return full path
        else:
            if os.path.isdir(os.path.join(p, entry)) and entry[0] == str(a) and entry[1] == str(b):
                return os.path.join(p, entry)  # Return full path

    return None


def extract_features(image_path):
    image = Image.open(image_path)
    image = image.convert('L')
    image = image.resize((50, 50))
    features = list(image.getdata())
    return features


def get_new_features():
    cols = list()
    for i in range(16, 2516):
        cols.append(f'{i}')
    new_df = pd.DataFrame(columns=cols, dtype=float)

    for i in range(1, 37):
        if 15 < i < 22:
            continue
        path = f'leaves'
        full_path = find_folder(path, i)
        for filename in os.listdir(full_path):
            image_path = full_path + "\\" + filename
            row = extract_features(image_path)
            new_df.loc[new_df.shape[0]] = row

    return new_df


df = pd.read_csv("leaves.csv", header=None)
df = df.join(get_new_features())
df.to_csv("new_data.csv", index=False)
