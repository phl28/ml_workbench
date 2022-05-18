# train model
from enum import unique
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

    sampling_rate = st.number_input(label = "Sampling rate", min_value = 0.0, max_value = None, value = 20.0, step = 1.0)

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
        labels = st.selectbox("What do you want to predict? ", column_labels)   
        # tbh we might not be able to do it this way because this is a continous time series data

        # creating a temp df just to calculate the accurate time based on sampling rate
        max_t = 0
        for i in range(len(data.columns)):
            df_temp = pd.DataFrame(data[data.columns[i]])
            df_temp['idx'] = df_temp.index / sampling_rate
        
            max_t = df_temp['idx'].values[-1] if df_temp['idx'].values[-1] > max_t else max_t


        # normalise data
        normalised_data = normalise(data[features])
        label_data = data[labels]
        num_labels = len(pd.unique(label_data))

        col1, col2, col3 = st.columns([3, 6, 2])

        with col2:
            st.write("Normalised features data preview")
            with st.empty():
                st.write(normalised_data.head(10))
            # plot the normalised features data
            alt.Chart()
            st.write("Label data preview")
            with st.empty():
                st.write(label_data.head(10)) 
    

        train_data_prop = st.slider('Train data size', min_value=0.5, max_value=0.8, step = 0.05)

        # creating the training dataset, validation dataset and testing dataset
        train_data, test_val_data, train_label, test_val_label = train_test_split(normalised_data, label_data, test_size = 1-train_data_prop)
        val_data, test_data, val_label, test_label = train_test_split(test_val_data, test_val_label, test_size = 0.5)


        # making predictions
        # model building
        model = Sequential([
            keras.layers.LSTM(32, input_shape = (None, None, 1), return_sequences = False),
            keras.layers.LSTM((64)),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dense(num_labels, activation = 'softmax')
        ])

        col1, col2, col3 = st.columns([3, 6, 2])
        
        with col2:
            with st.empty():
                st.write(model.summary())       

        model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy']) # adam is just the algorithm that performs gradient descent
        model.fit(train_data, train_label, epochs = 10, validation_data = (val_data, val_label))
        val_loss, val_acc = model.evaluate(val_data, val_label, verbose = 1)
        # print('Test accuracy:', test_acc)

        predictions = model.predict(test_data)


        # predictions, val_loss, val_acc, model = train_model(train_data, train_label, test_data, test_label, val_data, val_label)
        st.write("Test accuracy: {}".format(val_acc))

        # we could also give the option to plot the data
        chart_data = alt.Chart()
        container_data = st.empty()
        container_data.write(chart_data)

        # we would need to also include a confidence level chart here just to show the results not to do anything else
        chart_conf = alt.Chart()
        conf_container = st.empty()
        conf_container.write(chart_conf)

        # alert_decision = ["Value", "Mean"]
        # alerts = st.selectbox("Confidence level alert based on:", alert_decision)
        # if alerts ==" Value":
        #     alert = st.number_input("Value below (%):", min_value = 50, max_value = 100, value = 80, step = 5)
        # elif alerts == "Mean":
        #     alerts = st.number_input("Number of standard deviations below mean:", min_value = 0.0, max_value = 2.0, value = 0.0, step = 0.5)
        
        # mean_conf = np.mean(confidence)
        # std_conf = np.std(confidence)
        # intervals = []
        # below = False
        # if alerts == "Value":
        #     for i in range(len(confidence)):
        #         if confidence[i] < alert/100 and below == False:
        #             below = True
        #             intervals.append(i)
        #         elif confidence[i] >= alert/100 and below == True:
        #             below = False
        #             intervals.append(i)
        # elif alerts == "Mean":
        #     for i in range(len(confidence)):
        #         if confidence[i] < mean_conf - alert * std_conf and below == False:
        #             below = True
        #             intervals.append(i)
        #         elif confidence[i] >= mean_conf - alert * std_conf and below == True:
        #             below = False
        #             intervals.append(i)

        # interval_start = [round(intervals[i] * max_t / len(confidence), 1) for i in range(len(intervals)) if i % 2 == 0 ]
        # interval_end = [round(intervals[i] * max_t / len(confidence), 1) for i in range(len(intervals)) if i % 2 == 1 ]

        # new_intervals = list(zip(interval_start, interval_end))

        # interest = st.selectbox("Choose the interval you would like to investigate in:", new_intervals)

        # values = st.slider('Select a time range (s) for analysis', 0, int(max_t), (0, int(max_t)))
        # st.write('Selected time range:', values, 's')

        model_name = st.text_input("Model Name", placeholder = "Enter a name for the model (please do not include spaces)")
        model_name += ".h5"
        if model_name is not None:
            save_model(model_name, model)
            st.markdown('<p class="medium-font">Model is saved.', unsafe_allow_html= True)
        