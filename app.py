import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf
from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image


if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False


def callback():
    st.session_state.button_clicked = True


def __main__():
    st.title("Prediction of Tuberculosis using Chest X-Ray Images")
    image = Image.open("pic.jpg")
    st.image(image, caption=None, width=700)

    st.write("")

    if (st.button('Upload Image', key='Tuberculosis', on_click=callback) or st.session_state.button_clicked):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        predict_tuberculosis()


def loading_model():
    fp = "cnn_tuberculosis_model.h5"
    model_loader = load_model(fp)
    return model_loader


def predict_tuberculosis():
    st.set_option('deprecation.showfileUploaderEncoding', False)

    # @st.cache(allow_output_mutation=True)
    cnn = loading_model()

    #cnn = load_model("/content/cnn_tuberculosis_model.h5")

    upload = st.file_uploader("Upload X-Ray Image")

    buffer = upload
    temp_file = NamedTemporaryFile(delete=False)
    if buffer:
        temp_file.write(buffer.getvalue())
        st.write(image.load_img(temp_file.name))

    if buffer is None:
        st.text("Oops! that doesn't look like an image. Try again.")

    else:
        my_img = image.load_img(temp_file.name, target_size=(
            500, 500), color_mode='grayscale')

        # Preprocessing the image
        prep_img = image.img_to_array(my_img)
        prep_img = prep_img/255
        prep_img = np.expand_dims(prep_img, axis=0)

        # predict
        preds = cnn.predict(prep_img)
        if preds >= 0.5:
            out = ('I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(
                preds[0][0]))

        else:
            out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(
                1-preds[0][0]))

        st.success(out)

        image = Image.open(temp)
        st.image(image, use_column_width=True)

    # file = st.file_uploader("Upload X-Ray Image")

    # buffer = file
    # upload_file = NamedTemporaryFile(delete=False)
    # if buffer:
    #     upload_file.write(buffer.getvalue())
    #     st.write(image.load_img(upload_file.name))

    # if buffer is None:
    #     st.text("Please upload a file! ")

    # else:
    #     image = image.load_img(upload_file.name, target_size=(
    #         500, 500), color_mode='grayscale')
    # # Preprocessing the image
    #     pp_img = image.img_to_array(image)
    #     pp_img = pp_img/255
    #     pp_img = np.expand_dims(pp_img, axis=0)

    # # prediction
    #     pred = cnn.predict(pp_img)
    #     if pred >= 0.5:
    #         output = (
    #             'I am {:.2%} percent confirmed that this is a Tuberculosis case'.format(pred[0][0]))

    #     else:
    #         output = (
    #             'I am {:.2%} percent confirmed that this is a Normal case'.format(1-pred[0][0]))

    #     st.success(output)

    #     image = Image.open(upload_file)
    #     st.image(image, use_column_width=True)


__main__()
