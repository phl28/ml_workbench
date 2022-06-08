# make predictions
from functions2 import *
import streamlit as st


def app():
    # code where we allow users to choose trained models to do predictions here
    # File uploader widget
    st.sidebar.header("Initialisation")
    with st.sidebar.expander("Upload Files"):
        # add accept_multiple_files = True when we want to accept more files here
        # but we would then also need to change the code below as now we would need many more variables
        uploaded_files = st.file_uploader("Upload files to be analyzed", type = ['csv'], accept_multiple_files = True)
        st.markdown('<p class="small-font">**Your excel files have to be saved as csv files first.', unsafe_allow_html=True)
        uploaded_videos = st.file_uploader("Upload video", type = ['mp4', 'avi', 'mov'], accept_multiple_files = False)
        # path = ['-', "best_model.16-0.43.h5"] # this should not be empty
        # real_path_1 = st.selectbox('Choose the model you would like to use', path)
        st.write("or")
        real_path = st.file_uploader("Choose the model you would like to use", type = ['h5'])
    
    # if real_path_2 is None and real_path_1 != '-':
    #     real_path = real_path_1
    #     model = load_dnn_model(real_path)

    # this flag is used to determine if a real path has been chosen and the subsequent code can be run
    
    flag = False
    if real_path is not None:
        # real_path = real_path_2
        model = load_model(real_path.name)
        flag = True
    else:
        st.sidebar.write("Alert: You have not chosen a model.")  

    if flag:
        if len(uploaded_files) > 0:
            column_labels = []
            li = []
            for files in uploaded_files:
                df = pd.read_csv(files, index_col = None, header = 0)
                li.append(df)
                column_labels += list(df.columns)
            data = pd.concat(li, axis = 0, ignore_index = True)
            data = drop_index(data)
            column_labels = list(set(column_labels))
            column_labels.remove("index")

            features = st.multiselect("Which columns do you want for prediction: ", column_labels)
            labels = st.selectbox("What do you want to predict? ", column_labels)   
            st.markdown('<p class="small-font">**All the features and labels chosen here should be the same as when you trained the model', unsafe_allow_html= True)
            sampling_rate = st.number_input(label = "Sampling rate (Hz)", min_value = 0.0, max_value = None, value = 20.0, step = 0.1)

            # creating a temp df just to calculate the accurate time based on sampling rate
            max_t = 0
            for i in range(len(data.columns)):
                df_temp = pd.DataFrame(data[data.columns[i]])
                df_temp['idx'] = df_temp.index / sampling_rate
            
                max_t = df_temp['idx'].values[-1] if df_temp['idx'].values[-1] > max_t else max_t


            if features is not None:
                # normalise data
                normalised_data = normalise(data[features])


                st.write("Normalised features data preview")
                with st.empty():
                    st.write(normalised_data.head(10))
                # plot the normalised features data
                st.line_chart(normalised_data[:int(0.1*len(normalised_data))])

                    
                # make predictions with the uploaded data
                predictions = model.predict(normalised_data)

                # plot confidence and predictions
                
                conf = []
                pred = np.argmax(predictions, axis = -1)
                for i in predictions:
                    conf.append(max(i))
                pred = pd.DataFrame(pred, columns = ["predictions"])
                conf = pd.DataFrame(conf, columns = ["confidence"])
                chart_width = 1300
                chart_height = 300
                
                c1, c2 = st.columns((1,1))
                with c1:
                    # show video
                    st.video(uploaded_videos)
                
                with c2:
                    # incomplete raw data plot 
                    variables = st.multiselect('Select features data:', [column_labels[i] for i in range(len(column_labels))], [column_labels[i] for i in range(len(column_labels))])

                    container_data = st.empty()
                    chart_data = alt.Chart(pd.DataFrame({'x':np.arange(0, int(max_t),1), 'y':np.zeros(int(max_t) - 0)})).mark_line(color = 'white').encode(x = alt.X('x', scale = alt.Scale(domain = [0, int(max_t)])), y = alt.Y('y'))

                    # incomplete 
                    # for i in range(len(column_labels)):
                    #     if column_labels[i] in variables:
                    #         chart_data += alt.Chart(data_dict[column_labels[i]]).mark_line().encode(x = alt.X('idx', axis = alt.Axis(title = 'Time (s)')), y = alt.Y(column_labels[i], axis = alt.Axis(title = 'Features')), color = 'key').properties(width = chart_width, height = chart_height)
                    
                    container_data.write(chart_data)
                
                # predictions plot
                st.write("Predictions")
                chart_pred = alt.Chart(pred.reset_index()).mark_line().encode(x='index', y='predictions').properties(width = chart_width, height = chart_height)
                container_pred = st.empty()
                container_pred.write(chart_pred)

                # confidence level chart here 
                st.write("Confidence")
                chart_conf = alt.Chart(conf.reset_index()).mark_line().encode(x='index', y='confidence').properties(width = chart_width, height = chart_height)
                conf_container = st.empty()
                conf_container.write(chart_conf)

               


 

