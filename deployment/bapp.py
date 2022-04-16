import streamlit as st
# import tensorflow as tf
import numpy as np
import pandas as pd
import json
import altair as alt
import PIL
# from tensorflow.keras.models import load_model
# from tensorflow.keras.utils import CustomObjectScope
# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.preprocessing import image



st.title('Love Dolphins & Whales?')
st.markdown('This super classifier here will classify your marine spotting photo as one of a total of 30 different whale and dolphin species!')


img_file_buffer = st.file_uploader("Choose a file")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    image = np.array(tf.io.decode_image(bytes_data, channels=3))
    #st.write(type(image))
    #st.write(image.shape)
    
    input_shape = (224, 224)
    image = tf.keras.preprocessing.image.smart_resize( image, input_shape, interpolation='bilinear')
    #st.write(type(image))
    #st.write(image.shape)

    #st.write(image[0])
    
    image = image/255
   # st.write(image[0])
    image = np.expand_dims(image, axis=0)
    #st.write(image.shape)

with CustomObjectScope(
    {'GlorotUniform': glorot_uniform()}):
    model = load_model('./mobilenet_transfer_model.h5')

with open("./prediction_dict.json", "r") as file:
    dictionary = json.load(file)

if st.button('Submit'):
    preds = model.predict(image)
    pred = dictionary[str(np.argmax(preds))]
    pred = ' '.join(pred.split('_')[1:])
    st.write(f"This is likely a picture of a {pred}!")