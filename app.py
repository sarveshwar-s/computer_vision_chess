import streamlit as st
import pandas as pd
import numpy as np

st.title('Predicting Chess Positions')

def main():
    st.title("File Upload Tutorial")

    menu = ["Image","Dataset","DocumentFiles","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Image":
        st.subheader("Image")

    elif choice == "About":
        st.subheader("About")

    return choice

from PIL import Image

def load_image(image_file):
	img = Image.open(image_file)
	return img

from skimage.util.shape import view_as_blocks
from skimage import io, transform

piece_symbols = 'prbnkqPRBNKQ'

def process_image(img):
    SQUARE_SIZE = 40
    downsample_size = SQUARE_SIZE*8
    square_size = SQUARE_SIZE
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)

def fen_from_onehot(one_hot):
    output = ''
    for j in range(8):
        for i in range(8):
            if(one_hot[j][i] == 12):
                output += ' '
            else:
                output += piece_symbols[one_hot[j][i]]
        if(j != 7):
            output += '-'

    for i in range(8, 0, -1):
        output = output.replace(' ' * i, str(i))

    return output

def get_prediction(image_file):
    import tensorflow as tf
    loaded_model = tf.keras.models.load_model("models/model_2_chess_position.h5")
    print("Model loaded and ready to predict..")
    image_path = "images/" + image_file
    live_prediction_fen = loaded_model.predict(process_image(image_path)).argmax(axis=1).reshape(-1, 8, 8)
    final = np.array([fen_from_onehot(one_hot) for one_hot in live_prediction_fen])
    
    return final

if main() == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    st.write(image_file.name)
    if image_file is not None:

            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
            st.write(file_details)

            # To View Uploaded Image
            st.image(load_image(image_file),width=250)
            import os
            with open(os.path.join("images",image_file.name),"wb") as f:
                f.write((image_file).getbuffer())

            st.success("File Saved")
            
            st.write("Model Performing prediction..")

            predicted_fen = get_prediction(image_file.name)

            if predicted_fen:
                st.write("Predicted FEN for your image is: ", predicted_fen[0])

