import streamlit as st #streamlit==1.8.1
# import pickle
# import PIL
# import tensorflow.keras # tensorflow==2.8.0 
import pandas as pd #pandas==1.3.5


# model = pickle.load(open('model1.p', 'rb')) # meets zsh: illegal hardware instruction 

# with open('model1.p', 'rb') as f: # meets zsh: illegal hardware instruction 
#     model = pickle.load(f)

# with open('author_pipe.pkl', 'rb') as f:
#     model = pickle.load(f)

st.title('hi there')

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
# image = PIL.Image.open('pic.png')
# st.image(image)