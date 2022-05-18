# update model
from functions2 import *
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import altair as alt
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy


def app():
    # this is where the system allows the user to generate new synchronised data to improve the model 

    uploaded_data = st.file_uploader("Upload data:** ", type = ["csv"], accept_multiple_files = True)
    st.markdown('<p class="small-font">**All files should have the same columns', unsafe_allow_html= True)

    sampling_rate = st.number_input(label = "Sampling rate", min_value = 0.0, max_value = None, value = 20.0, step = 1.0)

    if len(uploaded_data) != 0:
        column_labels = []
        li = []
        for files in uploaded_data:
            df = pd.read_csv(files, index_col = None, header = 0)
            li.append(df)
            # column_labels += list(df.columns)
        data = pd.concat(li, axis = 0, ignore_index = True)

        # creating a temp df just to calculate the accurate timestamp based on sampling rate
        max_t = 0
        for i in range(len(data.columns)):
            df_temp = pd.DataFrame(data[data.columns[i]])
            df_temp['idx'] = df_temp.index / sampling_rate
        
            max_t = df_temp['idx'].values[-1] if df_temp['idx'].values[-1] > max_t else max_t

        data = drop_index(data)
        column_labels = list(data.columns)
        # column_labels.remove("index")
        features = st.multiselect("Which columns do you want for prediction: ", column_labels)
        labels = st.selectbox("What do you want to predict? ", column_labels)   
        # tbh we might not be able to do it this way because this is a continous time series data

        # we need to allow them to add new synchronised dummy data here
        # requirements = 
        # dummy = generate_data()
        # column_name = st.text_input("What is this data stream?")
        # data[column_name] = dummy
        # features.append(column_name)

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
        train_data, test_val_data, train_label, test_val_label = train_test_split(normalised_data, data[labels], test_size = train_data_prop)
        val_data, test_data, val_label, test_label = train_test_split(test_val_data, test_val_label, test_size = 0.5)

        # making predictions
        # predictions, val_loss, val_acc, model = train_model(train_data, train_label, test_data, test_label, val_data, val_label)
       
        # model building
        model = Sequential([
            keras.layers.LSTM(32, input_shape = (None, None, 1), return_sequences = False),
            keras.layers.LSTM((64)),
            keras.layers.Dense(128, activation = 'relu'),
            keras.layers.Dense(num_labels, activation = 'softmax')
        ])
        
        with st.empty():
            st.write(model.summary())
            
        model.compile(optimizer = 'adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy']) # adam is just the algorithm that performs gradient descent
        model.fit(train_data, train_label, epochs = 10, validation_data = (val_data, val_label))
        val_loss, val_acc = model.evaluate(val_data, val_label, verbose = 1)
        # print('Test accuracy:', test_acc)
        # st.write("Test accuracy: {}".format(val_acc))



        predictions = model.predict(test_data)
        st.write("Test accuracy: {}".format(val_acc))

        # we could also give the option to plot the data
        chart_data = alt.Chart()
        container_data = st.empty()
        container_data.write(chart_data)

        # we would need to also include a confidence level chart here just to show the results not to do anything else
        chart_conf = alt.Chart()
        conf_container = st.empty()
        conf_container.write(chart_conf)

        alert_decision = ["Value", "Mean"]
        alerts = st.selectbox("Confidence level alert based on:", alert_decision)
        if alerts ==" Value":
            alert = st.number_input("Value below (%):", min_value = 50, max_value = 100, value = 80, step = 5)
        elif alerts == "Mean":
            alerts = st.number_input("Number of standard deviations below mean:", min_value = 0.0, max_value = 2.0, value = 0.0, step = 0.5)
        
        mean_conf = np.mean(confidence)
        std_conf = np.std(confidence)
        intervals = []
        below = False
        if alerts == "Value":
            for i in range(len(confidence)):
                if confidence[i] < alert/100 and below == False:
                    below = True
                    intervals.append(i)
                elif confidence[i] >= alert/100 and below == True:
                    below = False
                    intervals.append(i)
        elif alerts == "Mean":
            for i in range(len(confidence)):
                if confidence[i] < mean_conf - alert * std_conf and below == False:
                    below = True
                    intervals.append(i)
                elif confidence[i] >= mean_conf - alert * std_conf and below == True:
                    below = False
                    intervals.append(i)

        interval_start = [round(intervals[i] * max_t / len(confidence), 2) for i in range(len(intervals)) if i % 2 == 0 ]
        interval_end = [round(intervals[i] * max_t / len(confidence), 2) for i in range(len(intervals)) if i % 2 == 1 ]

        new_intervals = list(zip(interval_start, interval_end))

        interest = st.selectbox("Choose the interval you would like to investigate in:", new_intervals)

        values = st.slider('Select a time range (s) for analysis', 0, int(max_t), (0, int(max_t)))
        st.write('Selected time range:', values, 's')

        st.markdown('<p class="medium-font"Interest region', unsafe_allow_html= True)
        # plotting data in the targetted region
        chart_data = alt.Chart()
        container_data = st.empty()
        container_data.write(chart_data)

        # confidence chart of the targetted region
        chart_conf = alt.Chart()
        conf_container = st.empty()
        conf_container.write(chart_conf)

        # we would then need to ask if anything corrections need to be made 
        # ie if the user need to add new classification labels
        new_label = st.text_input("What label would you like to add to this region?", value = "", placeholder = "Empty")

        data[labels].loc[values[0]:values[1], labels] = new_label

        col1, col2, col3 = st.columns([3, 6, 2])
        with col2:
            st.write("New label data at the targetted region")
            with st.empty():
                st.write(data[labels].iloc[values[0]-2:values[1]+3])

        model_name = st.text_input("Model Name", placeholder = "Enter a name for the model")
        if model_name is not None:
            save_model(model_name, model)
            st.markdown('<p class="medium-font">Model is saved.', unsafe_allow_html= True)