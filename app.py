import streamlit as st
from os import path, listdir, remove
from utils import get_trained_model, predict

def clear_dir(folder):
    files = listdir(folder)
    for file in files:
        file_path = path.join(folder, file)
        if path.isfile(file_path):
            # deletes file
            remove(file_path)

UPLOAD_FOLDER='uploads'
model = get_trained_model()

st.title('Cataract Detection using Transfer Learning Approach')
st.subheader('Fullname: Ojo Oluwaseun Emmanuel')
st.subheader('Matric Number: 2018/1/00021CS')

ct = st.container(border=True)

col1, col2 = ct.columns([0.6, 0.4])
ct1 = col1.container(border=True, height=300)
ct2 = col2.container(border=True, height=300)

uploaded_image = ct1.file_uploader('Upload the Image to Classify', type=['jpg','png'], accept_multiple_files=False)
btn_classify = ct1.button('Classify Image', type='primary')

ct2.write('#### Classification Results')

if btn_classify:
    clear_dir(UPLOAD_FOLDER)
    file_path = path.join(UPLOAD_FOLDER, uploaded_image.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_image.getvalue())

    if path.exists(file_path) and path.isfile(file_path):
        predicted_result = predict(model, file_path, show_image=False)
        ct2.write(predicted_result)
        ct2.image(file_path, use_column_width=True)
