from os import listdir
from config import RC_DATA_PATH
import pandas as pd


def get_rc_data_set(data_set='train'):
    data_folder = fr'{RC_DATA_PATH}\{data_set}'
    images_folders = listdir(data_folder)
    return images_folders


def get_label2charmap():
    label2chardict = {}
    file = fr'{RC_DATA_PATH}\laber2char.xlsx'
    dfs = pd.read_excel(file, engine='openpyxl', sheet_name='Sheet1')
    for entry in dfs.iloc:
        label2chardict[entry.label] = entry.char

    return label2chardict


if __name__ == "__main__":
    print(get_label2charmap())
