
## develop/train model

# import libs
import numpy as np 
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils

training_flag =  0

# load training data
text = (open(r'C:\Users\tonyr\Dropbox\Companies + Projects\Visio Data Science\Projects\Text_generator\AI_text_gen\Mary_shelly.txt').read())
text = text.lower()

# create char/word maps 
characters = sorted(list(set(text)))
n_to_char = {n:char for n, char in enumerate(characters)}
char_to_n = {char:n for n, char in enumerate(characters)}


# data preporcessing/prepare training data 
X = []
Y = []
length = len(text)
seq_length = 100 

for i in range(0, length-seq_length, 1):
    sequence = text[i:i+seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

print(Y[0])

X_modified= np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

n_width = 500 #number of neurons/model width
epochs = 1
batch_size = 50

#build model
model = Sequential()
model.add(LSTM(n_width, input_shape = (X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_width))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
#train

if training_flag == 0:

    model.fit(X_modified, Y_modified, epochs = epochs, batch_size = batch_size)
    model.save_weights('mary_shelly_text_Gen_v2.h5')

else: 

    model.load_weights(r'C:\Users\tonyr\Dropbox\Companies + Projects\Visio Data Science\Projects\Text_generator\AI_text_gen\mary_shelly_text_Gen_v2.h5')
    ## use model to gen text
    string_mapped = X[99]
    full_string = [n_to_char[value] for value in string_mapped]

    for i in range(400):
        x = np.reshape(string_mapped, (1, len(string_mapped),1))
        x = x / float(len(characters))

        pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        full_string.append(n_to_char[pred_index])

        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]

    txt = ""
    for char in full_string:
        txt = txt + char

    print(txt)