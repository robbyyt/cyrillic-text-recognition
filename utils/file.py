import pandas as pd
import tensorflow as tf
from config import RC_DATA_PATH
from utils.nn import training_params


def get_rc_training_data(percentage_of_data_to_use=0.2):
    data_folder = fr'{RC_DATA_PATH}\train'
    validation_split = 1 - percentage_of_data_to_use
    data_set = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        validation_split=validation_split,
        subset="training",
        seed=123,
        batch_size=training_params.get('batch_size'),
        shuffle=True,
        image_size=training_params.get('img_size'),
    )
    return data_set


def get_rc_validation_data():
    data_folder = fr'{RC_DATA_PATH}\val'
    data_set = tf.keras.utils.image_dataset_from_directory(
        data_folder,
        seed=123,
        batch_size=training_params.get('batch_size'),
        shuffle=True,
        image_size=training_params.get('img_size'),
    )
    return data_set


def get_label2char_map():
    label2chardict = {}
    file = fr'{RC_DATA_PATH}\laber2char.xlsx'
    dfs = pd.read_excel(file, engine='openpyxl', sheet_name='Sheet1')
    for entry in dfs.iloc:
        label2chardict[entry.label] = entry.char

    return label2chardict
