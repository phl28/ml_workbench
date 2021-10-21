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

from tensorflow.keras.models import load_model

def load_dnn_model(path):
    """
    (Transfer learning) Loads the chosen model weights file, pre-trained on the WIDSM dataset.
    """
    model = load_model(path)
    return model

# Import saved (pre-trained) model
path = "best_model.16-0.43.h5"
model = load_dnn_model(path)

# Set global sampling frequency
freq = 20

# Fixed parameters used to train the WISDM dataset
time_period = 40
step_length = 40

def create_segments_and_labels(df, time_period, step_length):
    """
    Segments the data.
    df: dataframe
    time_period: length of each segment
    step_length: distance between adjacent segments
    """
    n_axis = 3
    segments = []
    for i in range(0, len(df) - time_period, step_length):
        xs = df[df.columns[df.columns.str.contains("x")]].values[i: i + time_period]
        ys = df[df.columns[df.columns.str.contains("y")]].values[i: i + time_period]
        zs = df[df.columns[df.columns.str.contains("z")]].values[i: i + time_period]

        segments.append([xs, ys, zs])

    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_period, n_axis)

    return reshaped_segments

def predict(df, time_period, step_length):
    """
    Makes predictions about activity based on input dynamics data.

    df (pd.DataFrame): dataframe
    time_period (int): length of each segment
    step_length (int): distance between adjacent segments

    """
    df = normalise(df)

    X = create_segments_and_labels(df, time_period, step_length)
    input_shape = time_period*3
    X = X.reshape(X.shape[0], input_shape)

    predictions = model.predict(X)
    label = np.zeros(len(predictions))
    conf = np.zeros(len(predictions))
    for i in range(len(predictions)):
        label[i] = np.argmax(predictions[i])
        conf[i] = np.max(predictions[i])

    return conf, label

def drop_index(df):
    if 'index' in df.columns:
        df = df.drop(['index'], axis=1)
    return df

def normalise(df):
    """
    Function that normalises a dataset to take values within -1 and 1.

    df (pd.DataFrame): Input dataset.

    """
    x = df.columns[df.columns.str.contains("x")]
    df_x = df[x]

    y = df.columns[df.columns.str.contains("y")]
    df_y = df[y]

    z = df.columns[df.columns.str.contains("z")]
    df_z = df[z]

    df_x = df_x / abs(df_x).max()
    df_y = df_y / abs(df_y).max()
    df_z = df_z / abs(df_z).max()

    return pd.concat([df_x, df_y, df_z], axis=1)

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

# Helper function for the main page
def main():
    st.set_page_config(layout="wide")

    # Set standard frame width and height
    frame_width = 600
    frame_height = 175

    # File uploader widget
    with st.sidebar.expander("Upload Files"):
        file_dyn = st.file_uploader("Upload Dynamics Data", type=['csv'])
        file_vit = st.file_uploader("Upload Vitals Data", type=['csv'])
        file_vid = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])

    if file_dyn is not None and file_vit is not None and file_vid is not None:
            # Process input data
            df_dyn = pd.read_csv(file_dyn, sep=',')
            df_vit = pd.read_csv(file_vit, sep=',')

            # Clean data (if required)
            df_dyn = drop_index(df_dyn)
            df_vit = drop_index(df_vit)

            # Create arrays of column headers from both input dynamics and vitals data
            dyn_cols = df_dyn.columns
            vit_cols = df_vit.columns

            st.write('')
            st.write('')

            with st.sidebar.expander("Select sampling rate"):

                st.write("Dynamics (Hz)")

                dyn_freq_dict = {}
                dyn_dict = {}
                dyn_max_t = 0

                rs = 1

                for i in range(len(dyn_cols)):
                    # Prompt user to input sampling rates
                    user_input = st.number_input("{}".format(dyn_cols[i]), min_value=None, max_value=None, value=20.0, step=0.01, key="{}".format(dyn_cols[i]))
                    dyn_freq_dict.update({"{}".format(dyn_cols[i]): user_input})
                    # Create a new dictionary which stores a dataframe for each vitals parameter
                    dyn_dict.update({"{}".format(dyn_cols[i]): ""})

                    if "acc" in dyn_cols[i]:
                        if dyn_freq_dict[dyn_cols[i]]>20:
                            rs = 1
                            st.write('Note: {} will be down-sampled'.format(dyn_cols[i]))
                        elif dyn_freq_dict[dyn_cols[i]]<20:
                            rs = 1
                            st.write('Note: {} will be up-sampled'.format(dyn_cols[i]))

                    df_temp = pd.DataFrame(df_dyn[dyn_cols[i]])

                    # Add a new column with the actual timestamp
                    df_temp['idx'] = df_temp.index/user_input
                    df_temp['key'] = dyn_cols[i]

                    # Adding dataframe as key to dictionary
                    dyn_dict["{}".format(dyn_cols[i])] = df_temp

                    if df_temp['idx'].values[-1] > dyn_max_t:
                        dyn_max_t = df_temp['idx'].values[-1]

                st.write("Vitals (Hz)")

                vit_freq_dict = {}
                vit_dict = {}
                vit_max_t = 0

                for i in range(len(vit_cols)):
                    # Prompt user to input sampling rate for each column (parameter) in vitals file
                    user_input = st.number_input("{}".format(vit_cols[i]), min_value=None, max_value=None, value=20.0, step=0.01, key="{}".format(vit_cols[i]))
                    vit_freq_dict.update({"{}".format(vit_cols[i]): user_input})

                    # Create a new dictionary which stores a dataframe for each vitals parameter
                    vit_dict.update({"{}".format(vit_cols[i]): ""})

                    df_temp = pd.DataFrame(df_vit[vit_cols[i]])

                    # Add a new column with the actual timestamp
                    df_temp['idx'] = df_temp.index/user_input
                    df_temp['key'] = vit_cols[i]

                    # Adding dataframe as key to dictionary
                    vit_dict["{}".format(vit_cols[i])] = df_temp

                    if df_temp['idx'].values[-1] > vit_max_t:
                        vit_max_t = df_temp['idx'].values[-1]

            # Resampling operation
            if rs == 1:
                for key, value in dyn_dict.items():
                    if "acc" in key:
                        df_temp = value
                        str_temp = df_temp.columns[0]

                        df_temp1 = pd.DataFrame({'idx':np.arange(0, df_temp['idx'].max()+1/20, 1/20),
                                                 })
                        df_temp1[str_temp] = np.interp(df_temp1.idx, df_temp.idx, df_temp[str_temp])
                        df_temp1['key'] = key
                        dyn_dict.update({key: df_temp1})


            ### NEURAL NETWORK PREDICTIONS

            # Extract only columns with acceleration data
            df_dyn_acc = pd.DataFrame({})

            for key, value in dyn_dict.items():
                if "acc" in key:
                    df_temp = value
                    for i in range(len(df_temp.columns)):
                        if "acc" in df_temp.columns[i]:

                            #acc_cols = pd.concat(acc_cols,df_temp[df_temp.columns[i]])
                            df_dyn_acc[key] = df_temp[df_temp.columns[i]]

            # Predictions based on pre-trained model
            conf, predictions = predict(df_dyn_acc, time_period, step_length)

            d = {0.0:'Down',
                 1.0:'Run',
                 2.0:'Sit',
                 3.0:'Up',
                 4.0:'Walk'
                 }

            activity_label = [d[predictions[i]] for i in range(len(predictions))]

            code = np.repeat(predictions, time_period)
            label = np.repeat(activity_label, time_period)
            confidence = np.repeat(conf, time_period)


            max_t = max(dyn_max_t, vit_max_t)

            c0, c1 = st.columns((1, 1))

            with c0:
                st.image('graphic_wide.png', width=600)
                st.write('Platform for clinicians to visualise and analyse dizziness-related data.')

            with c1:
                values = st.slider('Select a time range (s)', 0, int(max_t), (0, int(max_t)))
                st.write('Selected time range:', values, 's')
                start = st.button('Start Playback')

            for i in range(len(dyn_cols)):
                df_temp = dyn_dict["{}".format(dyn_cols[i])]
                df_temp = df_temp[(df_temp['idx'] >= values[0])  & (df_temp['idx'] <= values[1])]
                dyn_dict["{}".format(dyn_cols[i])] = df_temp

            for i in range(len(vit_cols)):
                df_temp = vit_dict["{}".format(vit_cols[i])]
                df_temp = df_temp[(df_temp['idx'] >= values[0])  & (df_temp['idx'] <= values[1])]
                vit_dict["{}".format(vit_cols[i])] = df_temp

            c2, c3 = st.columns((1, 1))

            with c2:

                # Video processing
                tfile = tempfile.NamedTemporaryFile(delete=True)
                tfile.write(file_vid.read())

                cap = cv2.VideoCapture(tfile.name)

                number_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                fps = cap.get(cv2.CAP_PROP_FPS)

                df_video = pd.DataFrame({'idx':np.arange(0,number_of_frames/int(fps),1/int(fps)), 'num':np.arange(0,number_of_frames,1)})
                df_video = df_video.set_index(['idx'])

                cap.set(cv2.CAP_PROP_POS_FRAMES, df_video.loc[int(values[0])].num)
                ret, frame = cap.read()

                st.markdown('<center><h3>Environmental Video</h3></center>', unsafe_allow_html=True)
                st.write('')
                stframe = st.image(frame, width=frame_width, height=frame_height+50)

            with c3:

                #Dynamics

                dyn_sel = st.multiselect('Select dynamics data:', [dyn_cols[i] for i in range(len(dyn_cols))], [dyn_cols[i] for i in range(len(dyn_cols))])

                container_dyn = st.empty()
                chart_dyn = alt.Chart(pd.DataFrame({'x':np.arange(int(values[0]),int(values[1]),1), 'y':np.zeros(int(values[1])-int(values[0]))})).mark_line(color='white').encode(x=alt.X('x', scale=alt.Scale(domain=[int(values[0]),int(values[1])])), y=alt.Y('y'))

                for i in range(len(dyn_cols)):
                    if dyn_cols[i] in dyn_sel:
                        chart_dyn += alt.Chart(dyn_dict[dyn_cols[i]]).mark_line().encode(x=alt.X('idx', axis=alt.Axis(title='Time (s)')), y=alt.Y(dyn_cols[i], axis=alt.Axis(title='Dynamics')), color='key').properties(width=frame_width, height=frame_height)

                        # , scale=alt.Scale(domain=[int(values[0]),int(values[1])])

                        #chart_dyn += alt.Chart(dyn_dict[dyn_cols[i]]).mark_line().encode(x=alt.X('idx', axis=alt.Axis(title='Time (s)'), scale=alt.Scale(domain=[int(values[0]),int(values[1])])), y=alt.Y(dyn_cols[i], axis=alt.Axis(title='Dynamics')), color='key').properties(width=frame_width, height=frame_height)

                container_dyn.write(chart_dyn)


                # Vitals

                vit_sel = st.multiselect('Select vitals data:', [vit_cols[i] for i in range(len(vit_cols))], [vit_cols[i] for i in range(len(vit_cols))])

                container_vit = st.empty()
                chart_vit = alt.Chart(pd.DataFrame({'x':np.arange(int(values[0]),int(values[1]),1), 'y':np.zeros(int(values[1])-int(values[0]))})).mark_line(color='white').encode(x=alt.X('x', scale=alt.Scale(domain=[int(values[0]),int(values[1])])), y=alt.Y('y'))

                for i in range(len(vit_cols)):
                    if vit_cols[i] in vit_sel:
                        chart_vit += alt.Chart(vit_dict[vit_cols[i]]).mark_line().encode(x=alt.X('idx',axis=alt.Axis(title='Time (s)')), y=alt.Y(vit_cols[i], axis=alt.Axis(title='Vitals')), color='key').properties(width=frame_width, height=frame_height)

                container_vit.write(chart_vit)


            # Activity classifier

            df_dyn_acc = df_dyn_acc[:len(label)]
            df_dyn_acc['activity_label'] = label
            df_dyn_acc['confidence'] = confidence
            df_dyn_acc['idx'] = df_dyn_acc.index/20
            df_dyn_acc = df_dyn_acc[(df_dyn_acc['idx'] >= values[0])  & (df_temp['idx'] <= values[1])]

            st.markdown('<center><h3>Neural network (Activity Classifier)</h3></center>', unsafe_allow_html=True)
            st.write("")

            chart_act = alt.Chart(df_dyn_acc).mark_point(color='green').encode(x=alt.X('idx', axis=alt.Axis(title='Time (s)'), scale=alt.Scale(domain=[int(values[0]),int(values[1])])), y=alt.Y('activity_label', axis=alt.Axis(title='Activity'))).properties(width=frame_width*2, height=frame_height*1.3)

            container_act = st.empty()
            container_act.write(chart_act.configure_title(fontSize=18).configure_axis(labelFontSize = 15, titleFontSize = 15))

            chart_act_conf = alt.Chart(df_dyn_acc).mark_line(color='red').encode(x=alt.X('idx', axis=alt.Axis(title='Time (s)'), scale=alt.Scale(domain=[int(values[0]),int(values[1])])), y=alt.Y('confidence', axis=alt.Axis(title='Confidence'))).properties(width=frame_width*2, height=frame_height*1.3)

            container_act_conf = st.empty()
            container_act_conf.write(chart_act_conf.configure_title(fontSize=18).configure_axis(labelFontSize = 15, titleFontSize = 15))


            # Download data

            st.markdown('<h3>Download data:</h3>', unsafe_allow_html=True)
            st.write('Selected time range:', values, 's')

            patient_name = st.text_input('Enter patient name:')

            dyn_sel_1 = st.multiselect('Select dynamics data to download:', [dyn_cols[i] for i in range(len(dyn_cols))], [dyn_cols[i] for i in range(len(dyn_cols))])

            df_dyn_download = pd.DataFrame({})
            for i in range(len(dyn_cols)):
                if dyn_cols[i] in dyn_sel_1:
                    df_dyn_download = pd.concat([df_dyn_download, dyn_dict[dyn_cols[i]]], axis=1)

            if st.button('Download dynamics data as CSV'):
                tmp_download_link = download_link(df_dyn_download, "{}_dynamics.csv".format(patient_name), 'Click here to download your file')
                st.markdown(tmp_download_link, unsafe_allow_html=True)


            vit_sel_1 = st.multiselect('Select vitals data to download:', [vit_cols[i] for i in range(len(vit_cols))], [vit_cols[i] for i in range(len(vit_cols))])

            df_vit_download = pd.DataFrame({})
            for i in range(len(vit_cols)):
                if vit_cols[i] in vit_sel_1:
                    df_vit_download = pd.concat([df_vit_download, vit_dict[vit_cols[i]]], axis=1)

            if st.button('Download vitals data as CSV'):
                tmp_download_link = download_link(df_vit_download, "{}_vitals.csv".format(patient_name), 'Click here to download your file')
                st.markdown(tmp_download_link, unsafe_allow_html=True)

            # Download classification report

            df_dyn_acc_download = df_dyn_acc.drop(['idx'], axis=1)

            st.write("")
            st.markdown('<h3>Download classification results:</h3>', unsafe_allow_html=True)
            file_name_class = st.text_input('Enter file name here:')
            #st.write(file_name)

            if st.button('Download classification results as CSV'):
                tmp_download_link = download_link(df_dyn_acc_download, "{}_activity.csv".format(patient_name), 'Click here to download your file')
                st.markdown(tmp_download_link, unsafe_allow_html=True)


            ### PLAYBACK

            frame_current = int(df_video.loc[values[0]].num)
            frame_end = int(values[1])*30

            while start and frame_current <= frame_end:

                df_dyn['a'] = frame_current/30
                vline = alt.Chart(df_dyn).mark_rule().encode(x='a')

                chart_d = chart_dyn + vline
                container_dyn.write(chart_d.configure_title(fontSize=18).configure_axis(labelFontSize = 15,
                                                                                    titleFontSize = 15))

                chart_v = chart_vit + vline
                container_vit.write(chart_v.configure_title(fontSize=18).configure_axis(labelFontSize = 15,
                                                                                    titleFontSize = 15))

                chart_a = chart_act + vline
                container_act.write(chart_a.configure_title(fontSize=18).configure_axis(labelFontSize = 15,
                                                                                    titleFontSize = 15))

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_current)
                ret, frame = cap.read()

                stframe.image(frame, width=frame_width, height=frame_height)

                frame_current += 1

                time.sleep(1/2000)

if __name__ == "__main__":
    main()
