from dataclasses import dataclass
import cv2
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import time
import io
import tempfile
import random
import base64
from PIL import Image
from tensorflow import keras
from tqdm import tqdm
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import os.path

def load_dnn_model(path):
    """
    (Transfer learning) Loads the chosen model weights file, pre-trained on the WIDSM dataset.
    """
    model = load_model(path)
    return model

# def create_segments_and_labels(df, time_period, step_length):
#     """
#     Segments the data.
#     df: dataframe
#     time_period: length of each segment
#     step_length: distance between adjacent segments
#     """
#     n_axis = 3
#     segments = []
#     for i in range(0, len(df) - time_period, step_length):
#         xs = df[df.columns[df.columns.str.contains("x")]].values[i: i + time_period]
#         ys = df[df.columns[df.columns.str.contains("y")]].values[i: i + time_period]
#         zs = df[df.columns[df.columns.str.contains("z")]].values[i: i + time_period]

#         segments.append([xs, ys, zs])

#     reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_period, n_axis)

#     return reshaped_segments

def predict(df, time_period, step_length, model):
    """
    Makes predictions based on input data.

    df (pd.DataFrame): dataframe
    time_period (int): length of each segment
    step_length (int): distance between adjacent segments

    """
    # df = normalise(df)

    # X = create_segments_and_labels(df, time_period, step_length)
    # input_shape = time_period * 3
    # X = X.reshape(X.shape[0], input_shape)

    # predictions = model.predict(X)
    # label = np.zeros(len(predictions))
    # conf = np.zeros(len(predictions))
    # for i in range(len(predictions)):
    #     label[i] = np.argmax(predictions[i])
    #     conf[i] = np.max(predictions[i])

    return conf, label

def drop_index(df):
    if 'index' in df.columns:
        df = df.drop(['index'], axis=1)
    return df

def normalise(df):
    """
    Function that normalises the data in the dataframe to take values within -1 and 1.

    df (pd.DataFrame): Input dataset.s

    """
    filtered_df = df.dtypes[(df.dtypes == np.int64) | (df.dtypes == np.float64)]
    column_names = list(filtered_df.index)
    # print(column_names)
    normalised_df = pd.DataFrame()
    for column in column_names:
        normalised_df[column] = df[column] / abs(df[column]).max()
    return normalised_df
    # print(normalised_df)

def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

def save_model(model_name,model):
    # # using pickle to save the trained model

    # return pickle.dump(model, open('{}'.format(model_name), 'wb'))
    if os.path.isfile(model_name) is False:
        model.save(model_name)


# def train_model(train_data, train_label, test_data, test_label, val_data, val_label):
#         # model building
#         model = keras.Sequential([
#             keras.layers.Flatten(input_shape = (28,28)), 
#             keras.layers.Dense(128, activation = 'relu'),
#             keras.layers.Dense(10, activation = 'softmax')
#         ])
#         model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy']) # adam is just the algorithm that performs gradient descent
#         model.fit(train_data, train_label, epochs = 10)
#         val_loss, val_acc = model.evaluate(val_data, val_label, verbose = 1)
#         # print('Test accuracy:', test_acc)
#         # st.write("Test accuracy: {}".format(val_acc))
#         predictions = model.predict(test_data)
#         return predictions, val_loss, val_acc, model

def generate_data(new_data_streams, length_of_data, *args, **kwargs):
    
    return data

