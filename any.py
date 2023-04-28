from re import L
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from sklearn.preprocessing import MinMaxScaler

# d = {"age":[12, 32, 23],
# "name":["adrian", "enoch", "karen"], 
# "year":[1,2,3]}
# df = pd.DataFrame(data = d)

# filtered_df = df.dtypes[(df.dtypes == np.int64) | (df.dtypes == np.float64)]
# column_names = list(filtered_df.index)
# print(column_names)
# normalised_df = pd.DataFrame()
# for column in column_names:
#     normalised_df[column] = df[column] / abs(df[column]).max()
    
# print(normalised_df)

number_of_sets = 20
df = pd.DataFrame(columns=['weight', 'labels', 'set number'])
for n in range(number_of_sets):
    weight = []
    labels = []
    zeros = random.randint(1,3)

    for i in range(zeros):
        weight.append(0)
        labels.append(0)
    weight.append(400)
    labels.append(1)
    bites = 400 / 25
    for i in range(int(bites)):
        bite = random.randrange(2, 8, 2)
        for j in range(int(bite)):
            weight.append(weight[-1])
            labels.append(2)
        new_weight = weight[-1] - (25 + random.gauss(0,1))
        if new_weight < 0:
            new_weight = 0.0
        weight.append(new_weight)
        labels.append(3)
    finish = random.randint(1,3)
    for k in range(finish):
        weight.append(weight[-1])
        labels.append(4)

    set = [int(n)] * len(weight)
    d = {'weight': weight, 'labels': labels, 'set number': set}
    df_n = pd.DataFrame(data = d)
    df = pd.concat([df, df_n], ignore_index=True)

# labels = tf.keras.utils.to_categorical(df["labels"], num_classes = 5)
labels = df["labels"]
# print(labels)
data = df[["weight", "set number"]]
# df.to_csv('test.csv')
# print(data)
train_df = data[data["set number"] < 16]["weight"]
train_labels = labels[:len(train_df)]
test_df = data[data["set number"] > 15]["weight"]
test_labels = labels[len(train_df):]

train_df = np.asarray(train_df).astype('float32')
train_labels = np.asarray(train_labels).astype('float32')
test_df = np.asarray(test_df).astype('float32')
test_labels = np.asarray(test_labels).astype('float32') 


label = ["nothing", "add food", "swallowing", "eating", "finished eating"]
# print(type(train_df))
# print(type(train_labels))
# print(train_labels)
new_label = []
for i in range(len(train_labels)):
    new_label.append(label[int(train_labels[i])])
# print(new_label)
# print(train_df.shape)
# print(train_df)
# print(type(test_df))
# print(type(test_labels))

def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 2, 15, 40
    model = keras.Sequential()
    model.add(layers.LSTM(100, return_sequences=True, input_shape= (1, 1)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(100, activation="relu"))
    model.add(layers.Dense(5, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(trainX, trainy, epochs = epochs, batch_size=batch_size, verbose=verbose)
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy

print(evaluate_model(train_df, train_labels, test_df, test_labels))
