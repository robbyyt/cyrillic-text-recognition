from config import RC_DATA_PATH
import pandas as pd
import tensorflow as tf

from utils.nn import training_params


def get_rc_data_set(data_set_type='train'):
    data_folder = fr'{RC_DATA_PATH}\{data_set_type}'
    data_set = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        validation_split=0.99,
        subset="training",
        seed=123,
        batch_size=training_params.get('batch_size'),
        shuffle=True,
        image_size=training_params.get('img_size'),
        color_mode='grayscale'
    )
    print(data_set.class_names)
    return data_set


def get_label2char_map():
    label2chardict = {}
    file = fr'{RC_DATA_PATH}\laber2char.xlsx'
    dfs = pd.read_excel(file, engine='openpyxl', sheet_name='Sheet1')
    for entry in dfs.iloc:
        label2chardict[entry.label] = entry.char

    return label2chardict


if __name__ == "__main__":
    print(get_rc_data_set())
