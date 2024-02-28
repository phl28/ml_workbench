# train model
from enum import unique
from unittest import result
from functions2 import *
import streamlit as st
from sklearn.model_selection import train_test_split
import pandas as pd
import altair as alt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.utils import to_categorical


def app():
    # code where we allow users to train model start here
    # num_file_formats = st.number_input(label = "How many different file types are there?", min_value=1, max_value = 5, value=1, step = 1)
    # num_files = 0
    # while True:
    #     uploaded_data = st.file_uploader("Upload data for training and testing:** ", type = ["csv"], accept_multiple_files = True)
    #     st.markdown('<p class="small-font">**All files should have the same columns', unsafe_allow_html= True)
    #     num_files += 1
    #     if num_files == num_file_formats:
    #         break

    uploaded_data = st.file_uploader("Upload data for training and testing: ", type = ["csv"], accept_multiple_files = True)

    sampling_rate = st.number_input(label = "Sampling rate (Hz)", min_value = 0.0, max_value = None, value = 20.0, step = 0.1)

    if len(uploaded_data) != 0:
        column_labels = []
        li = []
        for files in uploaded_data:
            df = pd.read_csv(files, index_col = None, header = 0)
            li.append(df)
            # column_labels += list(df.columns)
        data = pd.concat(li, axis = 0, ignore_index = True)
        data = drop_index(data)
        column_labels = list(set(data.columns))
        # column_labels.remove("index")
        features = st.multiselect("Which columns do you want for prediction: ", column_labels)
        labels = st.selectbox("What do you want to predict? ", column_labels, )   
        # tbh we might not be able to do it this way because this is a continous time series data


        unique_set = pd.unique(data['set number'])
        loaded = list()
        for s in unique_set:
            new_data = data['weight'].where(data['set number'] == s).to_list()
            loaded.append(new_data)
        
        loaded = np.dstack(loaded)
        st.write(loaded.shape)

        # creating a temp df just to calculate the accurate time based on sampling rate
        max_t = 0
        for i in range(len(data.columns)):
            df_temp = pd.DataFrame(data[data.columns[i]])
            df_temp['idx'] = df_temp.index / sampling_rate
        
            max_t = df_temp['idx'].values[-1] if df_temp['idx'].values[-1] > max_t else max_t


        if features is not None:
            # normalise data
            normalised_data = normalise(data[features])
            unique_labels = pd.unique(data[labels])
            num_labels = len(unique_labels)

            st.markdown('<p class="medium-font">Categorical <-> Label', unsafe_allow_html= True)
            for i in range(num_labels):
                st.write("{}: ".format(i), unique_labels[i])
            
            data.labels =  pd.Categorical(data.labels, unique_labels)
            data.labels = data.labels.cat.codes

            label_data = data[labels]

            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("Normalised features data preview")
                with st.empty():
                    st.write(normalised_data.head(10))
                # plot the normalised features data
                st.line_chart(normalised_data[:int(0.1*len(normalised_data))])
        
            with col2:
                st.write("Label data preview")
                with st.empty():
                    st.write(label_data.head(10)) 
        

            label_data = to_categorical(label_data) 
            train_data_prop = st.slider('Train data size', min_value=0.5, max_value=0.8, step = 0.05)
        
            # the following line is written as a thought of maybe we dont use validating set because that will make adjustments to training data difficult
            # train_data = normalised_data
            # creating the training dataset, validation dataset and testing dataset
            train_data, test_val_data, train_label, test_val_label = train_test_split(normalised_data, label_data, test_size = 1-train_data_prop)
            val_data, test_data, val_label, test_label = train_test_split(test_val_data, test_val_label, test_size = 0.5)

            train_data_label_shape = train_label.shape
            train_data_shape = train_data.shape
            st.write(train_data_shape)
            st.write(train_data_label_shape)

            n_timesteps, n_features, n_outputs = train_data_shape[0], train_data_shape[1], train_data_label_shape[1]
            # making predictions
            # model building
            model = Sequential([
                keras.layers.LSTM(100, input_shape = (n_timesteps, n_features)),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(100, activation = 'relu'),
                keras.layers.Dense(n_outputs, activation = 'softmax')
            ])

            model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy']) # adam is just the algorithm that performs gradient descent
            model.fit(train_data, train_label, epochs = 15, verbose = 0, validation_data = (val_data, val_label))
            val_loss, accuracy = model.evaluate(test_data, test_label, verbose = 0)

            predictions = model.predict(test_data)
            # st.write(predictions[:10])
            st.write("Test accuracy: {}%".format(round(accuracy*100, 3)))

            conf = []
            pred = np.argmax(predictions, axis = -1)
            for i in predictions:
                conf.append(max(i))
            pred = pd.DataFrame(pred, columns = ["predictions"])
            conf = pd.DataFrame(conf, columns = ["confidence"])

            # # we could also give the option to plot the data
            chart_data = alt.Chart(pred.reset_index()).mark_line().encode(x='index', y='predictions').properties(width = 1300, height = 300)
            container_data = st.empty()
            container_data.write(chart_data)

            # # we would need to also include a confidence level chart here just to show the results not to do anything else
            chart_conf = alt.Chart(conf.reset_index()).mark_line().encode(x='index', y='confidence').properties(width = 1300, height = 300)
            conf_container = st.empty()
            conf_container.write(chart_conf)

            model_name = st.text_input("Model Name")

            if model_name != "":
                model_name += ".h5"
                save_model(model_name, model)
                st.markdown('<p class="small-font">Model is saved.', unsafe_allow_html= True)
        