import streamlit as st #streamlit==1.8.1
import PIL # Pillow==9.0.1
import tensorflow as tf
from tensorflow.keras.models import load_model # tensorflow==2.8.0
import numpy as np # numpy==1.21.5



st.title('hi there. Gimme a pic of the beach')
img_file_buffer = st.file_uploader("Choose a file")

model = load_model('deployment/model.h5')

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    array = tf.io.decode_image(bytes_data, channels=3)/255 # the is what read_image_p would give out. 
    pred = model.predict(np.expand_dims(array, axis=0))
    image = PIL.Image.open('deployment/pic.png') # 
    st.image(image)



st.markdown('ok, the md goes here')


# st.subheader('Jane Austen or Edgar Alan Poe?')

# txt = st.text_area('Write your prose here')


# if st.button('Submit!!!! '):
#      st.write(f'{model.predict([txt])[0]}')


# st.title('Where the trash at?')

# st.subheader('Give me a picture of the beach and I\'ll show ya!')

# img_file_buffer = st.file_uploader('Upload your image of the beach here.')


# if st.button('Submit!!!! '):
#      st.write(f'{model.predict([txt])[0]}')

# for uploading a file: 
# st.file_uploader(label, type=None, accept_multiple_files=False, 
#                 key=None, help=None, on_change=None, args=None, 
#                 kwargs=None, *, disabled=False)

# then run model on the image and use output

# Displays an image:
