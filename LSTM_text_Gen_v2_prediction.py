# predicting text from .h5 trained model

import numpy as np
import LSTM_text_gen_v2 as trained_model

#load saved weights 
trained_model.model.load_weights("mary_shelly_text_Gen.h5")
string_mapped = trained_model.X[99]
full_string = [trained_model.n_to_char[value] for value in string_mapped]

for i in range(400):
    x = np.reshape(string_mapped,(1 ,len(string_mapped), 1))
    x = x / float(len(trained_model.characters))

    pred_index = np.argmax(trained_model.model.predict(x, verbose=0))
    seq = [trained_model.n_to_char[value] for value in string_mapped]
    full_string.append(trained_model.n_to_char[pred_index])

    string_mapped.append(pred_index)
    string_mapped = string_mapped[1:len(string_mapped)]

txt = ""
for char in full_string:
    txt = txt + char

print(txt)