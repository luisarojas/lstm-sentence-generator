### ---------- Handle Command Line Arguments ----------

import argparse
import helper

a = argparse.ArgumentParser(description="Generate sentences in English using LSTM Networks.")
a.add_argument("-v", "--version", action='version', version='%(prog)s 1.0.0')
a.add_argument("--data", help="path to training data (default: sample_data_long.txt)", default='sample_data_long.txt')
a.add_argument("--seed", help="provide a word or sentence to seed the program with")
a.add_argument("--nwords", help="number of words to generate (default: 400)", type=int, default=400)
a.add_argument("--temp", help="select temperature value for prediction diversity (default: 1.0)", type=float, default=1.0)
a.add_argument("--slen", help="maximum length for a training sequence (default: 15)", type=int, default=15)
a.add_argument("-win", help="select sliding window for text iteration and data collection for training (default: 3)", type=int, default=3)
args = a.parse_args()

helper.sequence_length = args.slen
helper.step_window = args.win

### ---------- Import Relevand Libraries ----------

import numpy as np
from keras.optimizers import RMSprop
import sys
import random
import re
from helper import load_file, create_model, sequence_length, add_temperature

### ---------- Load Text File and Build Vocabulary ----------

# note that the data length for training (train.py) and predictions (predict.py) must be the same

all_words, unique_words = load_file(args.data)
total_num_words = len(all_words)
len_vocab = len(unique_words)

print('\n----------------------------')
print("> Total number of words:\t" + str(total_num_words))
print("> Length of vocabulary:\t\t" + str(len_vocab))
print('----------------------------')

word_to_int = dict((c, i) for i, c in enumerate(unique_words))
int_to_word = dict((i, c) for i, c in enumerate(unique_words))

### ---------- Define Model ----------

num_layers = 1
drop_out_rate = 0.2
learning_rate = 0.01
optimizer = RMSprop(lr=learning_rate)

model = create_model(num_layers, drop_out_rate, len_vocab)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

## ---------- Predict ----------

def get_random_word(n=1, array=unique_words):

    random_words = []
    
    random_indices = random.sample(range(0, len(array)), n)
    
    # in-place shuffle
    random.shuffle(array)

    # take the first n elements of the now randomized array
    return array[:n]

# load weights
weights_path = '_weights.hdf5'
    
model.load_weights(weights_path)
model.compile(loss='categorical_crossentropy', optimizer='adam')


### ---------- Handle User Input ----------

# user_input = "this is some string entered by a user this is some string entered by a user this is some string entered by a user"
words_to_generate = args.nwords
seed_sentence = ["\n"] * sequence_length

if args.seed:
    
    user_input = args.seed
    
    pattern = re.compile('[a-z]+|\!|\n|\.|,|;')
    user_input_edit = user_input.lower()
    user_input_edit = re.findall(pattern, user_input)

    if (len(user_input_edit) > sequence_length):
        # need to truncate
        
        seed_sentence = user_input_edit[:sequence_length]
        
    elif (len(user_input_edit) < sequence_length):
        # need to pad
        
        # get number of elements missing in user input
        missing_elems = sequence_length - len(user_input_edit)
        
        seed_sentence[0:missing_elems+1] = get_random_word(missing_elems)
        seed_sentence[-(len(user_input_edit)):] = user_input_edit
        
    # check all words provided are in the vocabulary
    for i, word in enumerate(seed_sentence):
        # if it doesnt exist, replace it with one that does
        if word not in word_to_int.keys():
            seed_sentence[i] = get_random_word()[0]
    
    print('\n-> seed: "' + user_input + '" ...\n')

else:

    # pick a random seed
    random_index_start = np.random.randint(0, total_num_words - sequence_length - 1)
    seed_sentence = all_words[random_index_start : random_index_start + sequence_length]
    
    print('-> seed: "' + ' '.join(seed_sentence) + '" ...')
    
### ---------- Generate Words ----------

for i in range(words_to_generate):
    
    # convert to one-hot    
    x_input = np.zeros((1, sequence_length, len_vocab))
    for t, word in enumerate(seed_sentence):
        x_input[0, t, word_to_int[word]] = 1.

    predictions = model.predict(x_input, verbose=0)[0]
    predicted_word_index = add_temperature(predictions, args.temp)
    # predicted_word_index = np.argmax(predictions)
    predicted_word = int_to_word[predicted_word_index]

    seed_sentence = seed_sentence[1:] + list([predicted_word])

    if re.match('[a-z]', predicted_word):
        sys.stdout.write(" " + predicted_word)
    else:
        sys.stdout.write(predicted_word)

    sys.stdout.flush()
print()