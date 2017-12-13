import re
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import LSTM
import numpy as np

sequence_length = 15 # max number of words to consider at a time.
                    # this means that each trainig set (training pattern) will be comprised of 20 time steps
step_window = 3

def load_file(data_path):

    raw_text = open(data_path).read().lower()

    # generate list of unique characters, but only include words and some punctuation marks
    pattern = re.compile('[a-z]+|\!|\n|\.|,|;')
    all_words = re.findall(pattern, raw_text)

    unique_words = sorted(set(all_words))

    word_to_int = dict((c, i) for i, c in enumerate(unique_words))

    # later used to make outputs more readable by converting ints back to characters
    int_to_word = dict((i, c) for i, c in enumerate(unique_words))
    
    return all_words, unique_words

def create_model(num_layers, drop_out_rate, len_vocab):
    
    if num_layers < 1:
        print("There needs to be at least 1 LSTM layer in your model.")
        sys.exit(1)
    
    num_memory_units = 256

    model = Sequential()

    if num_layers == 1:

        model.add(LSTM(num_memory_units, input_shape=(sequence_length, len_vocab)))
        model.add(Dropout(drop_out_rate))       
    
    elif num_layers == 2:
        
        model.add(LSTM(num_memory_units, return_sequences=True, input_shape=(sequence_length, len_vocab)))
        model.add(Dropout(drop_out_rate))
        model.add(LSTM(num_memory_units))
        model.add(Dropout(drop_out_rate))  
        
    else:
        
        model.add(LSTM(num_memory_units, return_sequences=True, input_shape=(sequence_length, len_vocab)))
        model.add(Dropout(drop_out_rate))
        
        for i in range(num_layers):
            model.add(LSTM(num_memory_units))
            model.add(Dropout(drop_out_rate))   
    
    model.add(Dense(len_vocab))
    model.add(Activation('softmax'))
    
    return model

def add_temperature(predictions, temperature):

    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)