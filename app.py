import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False


def callback():
    st.session_state.button_clicked = True


def tb_home():
    st.title("Prediction of Tuberculosis using Chest X-Ray Images")
    image = Image.open("img_path")
    st.image(image, caption=None, width=700)
    st.write("")

    if (st.button('Predict Tuberculosis', key='Tuberculosis', on_click=callback) or st.session_state.button_clicked):
        predict_tuberculosis()


def predict_tuberculosis():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    @st.cache(allow_output_mutation=True)
    cnn = load_model(
        "/home/ayushman/Desktop/MLStuff/cnn_tuberculosis_model.h5")

    file = st.file_uploader("Upload X-Ray Image")

    buffer = file
    upload_file = NamedTemporaryFile(delete=False)
    if buffer:
        upload_file.write(buffer.getvalue())
        st.write(image.load_img(upload_file.name))

    if buffer is None:
        st.text("Please upload a file! ")

    else:
        image = image.load_img(upload_file.name, target_size=(
            500, 500), color_mode='grayscale')
    # Preprocessing the image
        pp_img = image.img_to_array(img)
        pp_img = pp_img/255
        pp_img = np.expand_dims(pp_img, axis=0)

    # prediction
        pred = cnn.predict(pp_img)
        if pred >= 0.5:
            output = (
                'I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(pred[0][0]))

        else:
            output = (
                'I am {:.2%} percent confirmed that this is a Normal case'.format(1-pred[0][0]))

        st.success(output)

        image = Image.open(upload_file)
        st.image(image, use_column_width=True)
