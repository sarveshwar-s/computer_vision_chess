import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import shuffle
from skimage.util.shape import view_as_blocks
from skimage import io, transform
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D
import warnings
warnings.filterwarnings('ignore')


import os
DATA_PATH='../input/chess-positions'
TRAIN_IMAGE_PATH=os.path.join(DATA_PATH, 'train')
TEST_IMAGE_PATH=os.path.join(DATA_PATH, 'test')

def get_image_filenames(image_path:str, image_type:str)-> np.ndarray:
    if(os.path.exists(image_path)):
        return glob.glob(os.path.join(image_path, '*.'+image_type))
    return

train = get_image_filenames(TRAIN_IMAGE_PATH, "jpeg")
test = get_image_filenames(TEST_IMAGE_PATH, "jpeg")
piece_symbols = 'prbnkqPRBNKQ'
train_size = 10000
test_size = 3000
train = train[:train_size]
test = test[:test_size]

def fen_extraction(filename:str):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0]

def onehot_from_fen(fen_name: str)-> np.ndarray:
    eye = np.eye(13) # returns a matrix with diagonal values as 1 
    output = np.empty((0,13))
    fen = fen_name.replace("-", "")
    for everychar in fen:
        if everychar in '12345678':
            output = np.append(output, np.tile(eye[12], (int(everychar), 1)), axis=0)
        else:
            idx = piece_symbols.index(everychar)
            output = np.append(output, eye[idx].reshape((1, 13)), axis=0)
    return output

def fen_from_onehot(one_hot: np.ndarray) -> str:
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

def process_image(img):
    SQUARE_SIZE = 40
    downsample_size = SQUARE_SIZE*8
    square_size = SQUARE_SIZE
    img_read = io.imread(img)
    img_read = transform.resize(img_read, (downsample_size, downsample_size), mode='constant')
    tiles = view_as_blocks(img_read, block_shape=(square_size, square_size, 3))
    tiles = tiles.squeeze(axis=2)
    return tiles.reshape(64, square_size, square_size, 3)

def train_gen(features, batch_size):
    for i, img in enumerate(features):
        y = onehot_from_fen(fen_extraction(img))
        x = process_image(img)
        yield x, y
    
def pred_gen(features, batch_size):
    for i, img in enumerate(features):
        yield process_image(img)
    
model = Sequential()
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(13))
model.add(Activation('softmax'))
model.compile(
  loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

chess_position_model = model.fit(train_gen(train, 64), steps_per_epoch=1000, verbose=1)

res = (
  model.predict(pred_gen(test, 64), steps=3000)
  .argmax(axis=1)
  .reshape(-1, 8, 8)
)

pred_fens = np.array([fen_from_onehot(one_hot) for one_hot in res])
test_fens = np.array([fen_extraction(fn) for fn in test])

final_accuracy = (pred_fens == test_fens).astype(float).mean()

print("Final Accuracy: {:1.5f}%".format(final_accuracy))