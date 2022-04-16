import streamlit as st #streamlit==1.8.1
import PIL # Pillow==9.0.1
from tensorflow.keras.models import load_model # tensorflow==2.8.0
st.title('hi there')

model = load_model('deployment/model.h5')

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
image = PIL.Image.open('deployment/pic.png')
st.image(image)