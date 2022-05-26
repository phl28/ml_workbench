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
from tensorflow.keras.utils import to_categorical


def app():
    # this is where the system allows the user to generate new synchronised data to improve the model 

    uploaded_data = st.file_uploader("Upload data:** ", type = ["csv"], accept_multiple_files = True)
    # st.markdown('<p class="small-font">**All files should have the same columns', unsafe_allow_html= True)
    uploaded_video = st.file_uploader("Upload video", type = ['mp4', 'avi', 'mov'], accept_multiple_files=False)


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
        column_labels = list(data.columns)
        # column_labels.remove("index")
        features = st.multiselect("Which columns do you want for prediction: ", column_labels)
        labels = st.selectbox("What do you want to predict? ", column_labels)   
        # tbh we might not be able to do it this way because this is a continous time series data

        # creating a temp df just to calculate the accurate timestamp based on sampling rate
        max_t = 0
        for i in range(len(data.columns)):
            df_temp = pd.DataFrame(data[data.columns[i]])
            df_temp['idx'] = df_temp.index / sampling_rate
        
            max_t = df_temp['idx'].values[-1] if df_temp['idx'].values[-1] > max_t else max_t
        
        # we need to allow them to add new synchronised dummy data here
        # requirements = 
        # dummy = generate_data()
        # column_name = st.text_input("What is this data stream?")
        # data[column_name] = dummy
        # features.append(column_name)

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

            # creating the training dataset, validation dataset and testing dataset
            train_data, test_val_data, train_label, test_val_label = train_test_split(normalised_data, label_data, test_size = train_data_prop)
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
            model.fit(train_data, train_label, epochs = 15, validation_data = (val_data, val_label))
            val_loss, accuracy = model.evaluate(test_data, test_label, verbose = 0)


            predictions = model.predict(test_data)
            st.write("Test accuracy: {}%".format(round(accuracy*100, 3)))

            conf = []
            pred = np.argmax(predictions, axis = -1)
            for i in predictions:
                conf.append(max(i))
            predict = pd.DataFrame(pred, columns = ["predictions"])
            confidence = pd.DataFrame(conf, columns = ['confidence'])

            # we could also give the option to plot the data
            chart_data = alt.Chart(predict.reset_index()).mark_line().encode(x='index', y='predictions').properties(width = 1300, height = 300)
            container_data = st.empty()
            container_data.write(chart_data)

            # we would need to also include a confidence level chart here just to show the results not to do anything else
            chart_conf = alt.Chart(confidence.reset_index()).mark_line().encode(x='index', y='confidence').properties(width = 1300, height = 300)
            conf_container = st.empty()
            conf_container.write(chart_conf)

            flag = True
            while flag:
                alert_decision = ["Value", "Mean"]
                alerts = st.selectbox("Confidence level alert based on:", alert_decision)
                if alerts == "Value":
                    alert = st.number_input("Value below (%):", min_value = 50, max_value = 100, value = 80, step = 5)
                elif alerts == "Mean":
                    alert = st.number_input("Number of standard deviations below mean:", min_value = 0.0, max_value = 2.0, value = 0.0, step = 0.5)
                
                mean_conf = np.mean(conf)
                std_conf = np.std(conf)
                intervals = []
                below = False
                if alerts == "Value":
                    for i in range(len(conf)):
                        if conf[i] < alert/100 and below == False:
                            below = True
                            intervals.append(i)
                        elif conf[i] >= alert/100 and below == True:
                            below = False
                            intervals.append(i)
                elif alerts == "Mean":
                    for i in range(len(conf)):
                        if conf[i] < mean_conf - alert * std_conf and below == False:
                            below = True
                            intervals.append(i)
                        elif conf[i] >= mean_conf - alert * std_conf and below == True:
                            below = False
                            intervals.append(i)

                indices_start = [intervals[i] for i in range(len(intervals)) if i % 2 == 0 ]
                indices_end = [intervals[i] for i in range(len(intervals)) if i % 2 == 1 ]
                interval_start = [round(intervals[i] * max_t / len(conf), 2) for i in range(len(intervals)) if i % 2 == 0 ]
                interval_end = [round(intervals[i] * max_t / len(conf), 2) for i in range(len(intervals)) if i % 2 == 1 ]

                new_intervals = list(zip(interval_start, interval_end))

                interest = st.selectbox("Choose the interval you would like to investigate in:", new_intervals)
                interested = [float(ele) for ele in interest]
                values = st.slider('Select a time range (s) for analysis', interested[0], interested[1], interested)
                st.write('Selected time range:', values, 's')

                st.markdown('<p class="medium-font">Interest region', unsafe_allow_html= True)

                start_interval = indices_start[interval_start.index(interest[0])]
                end_interval = indices_end[interval_end.index(interest[1])]

                col1, col2 = st.columns([1, 1])
                
                # # plotting data in the targetted region
                with col1:
                    chart_data = alt.Chart(predict.iloc[start_interval:(end_interval+1)].reset_index()).mark_line().encode(x='index', y='predictions').properties(width = 600, height = 300)
                    container_data = st.empty()
                    container_data.write(chart_data)

                # # confidence chart of the targetted region
                with col2:
                    chart_conf = alt.Chart(confidence.iloc[start_interval:(end_interval+1)].reset_index()).mark_line().encode(x='index', y='confidence').properties(width = 600, height = 300)
                    conf_container = st.empty()
                    conf_container.write(chart_conf)

                # show video (either use st.video or cv2) 
                # need to see which one allows the video range to be changed continuously 

                
                
                # we would then need to ask if anything corrections need to be made 
                # ie if the user need to add new classification labels
                new_label = st.text_input("What label would you like to add to this region?", value = "")


                if new_label != "":
                    data.loc[start_interval:end_interval+1, labels] = new_label
                    st.write("New label data at the targetted region")
                    with st.empty():
                        st.write(data[labels].iloc[start_interval - 2:end_interval+3])
                
            # save new data into a new csv file


            model_name = st.text_input("Model Name")
            if model_name != "":
                save_model(model_name, model)
                st.markdown('<p class="small-font">Model is saved.', unsafe_allow_html= True)