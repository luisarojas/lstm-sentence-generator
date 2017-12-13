### ---------- Handle Command Line Arguments ----------

import argparse
import helper

a = argparse.ArgumentParser(prog='Distracted Driver Detection', description="Train a model that, using LSTM Networks, is to generate sentences in English.")
a.add_argument("-v", "--version", action='version', version='%(prog)s 1.0.0')
a.add_argument("--data", help="path to training data (default: sample_data_short.txt)", default='sample_data_short.txt')
a.add_argument("--epochs", help="number of epochs to train for (default: 50)", type=int, default=50)
a.add_argument("--temp", help="select temperature value for prediction diversity (default: 1.0)", type=float, default=1.0)
a.add_argument("--slen", help="maximum length for a training sequence (default: 15)", type=int, default=15)
a.add_argument("-win", help="select sliding window for text iteration and data collection for training (default: 3)", type=int, default=3)
args = a.parse_args()

helper.sequence_length = args.slen
helper.step_window = args.win

### ---------- Import Relevand Libraries ----------

import re
import numpy as np
from keras.optimizers import RMSprop
import math
import random
import sys
import matplotlib.pyplot as plt
from keras.utils import plot_model
from helper import load_file, create_model, sequence_length, step_window, add_temperature
import os

### ---------- Load File and Build Vocabulary ----------

all_words, unique_words = load_file(args.data)

total_num_words = len(all_words)
len_vocab = len(unique_words)

print()
print("> Total number of words:\t" + str(total_num_words))
print("> Length of vocabulary:\t\t" + str(len_vocab))

word_to_int = dict((c, i) for i, c in enumerate(unique_words))
int_to_word = dict((i, c) for i, c in enumerate(unique_words))

### ---------- Create Training Data from Text File ----------

# set up x and y
x_data = [] # list of lists
y_data = []

for i in range(0, total_num_words - sequence_length, step_window):
    
    # extract the first n words (length sequence_length): our "x"
    sequence_in = all_words[i : i+sequence_length]
    
    # extract last word for this window: our "y" (target)
    word_out = all_words[i+sequence_length]
        
    # store corresponding integer for each character in the input sequence
    x_data.append(sequence_in)
    y_data.append(word_out)

num_train_patters = len(x_data)
print('> Total patterns:\t\t' + str(num_train_patters))

### ---------- Prepare Training Data ----------

x = np.zeros((num_train_patters, sequence_length, len_vocab))
y = np.zeros((num_train_patters, len_vocab))

# encode all data into one-hot vectors
for i, sentence in enumerate(x_data):
    for t, word in enumerate(sentence):
        x[i, t, word_to_int[word]] = 1
    y[i, word_to_int[y_data[i]]] = 1

### ---------- Define Model ----------

num_layers = 1
drop_out_rate = 0.2
learning_rate = 0.01
optimizer = RMSprop(lr=learning_rate)

model = create_model(num_layers, drop_out_rate, len_vocab)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


### ---------- Train Model ----------

num_iterations = args.epochs
batch_size = 128
words_to_generate = 300

prev_loss = math.inf
loss_history = []
accuracy_history = []

val_loss_history = []
val_accuracy_history = []

# train the model, output generated text after each iteration
for i in range(num_iterations):
    
    print('\n' + '-'*10 + ' epoch ' + str(i+1) + '/' + str(num_iterations) + ' ' + '-'*10)
        
    history = model.fit(x, y, batch_size=batch_size, epochs=1)
    
    curr_loss = history.history['loss'][0]
    loss_history.append(curr_loss)
    
    # save weights if loss improves
    if (curr_loss < prev_loss):
        print("Loss improved from " + str(prev_loss) + " to " + str(curr_loss) + ". Saving weights.")
        
        weights_dir = 'saved_weights/'
        if not os.path.exists(weights_dir):
            os.makedirs(weights_dir)
    
        model.save_weights(weights_dir + 'weights_epoch-{}_loss-{}.hdf5'.format(i, curr_loss))
        prev_loss = curr_loss
    
    start_index = random.randint(0, total_num_words - sequence_length - 1)
    # start_index = 0

    seed_sentence = all_words[start_index : start_index + sequence_length]

    print('\n-> seed: "' + ' '.join(seed_sentence) + '" ...\n')

    for i in range(words_to_generate):
        
        x_input = np.zeros((1, sequence_length, len_vocab))
        for t, word in enumerate(seed_sentence):
            x_input[0, t, word_to_int[word]] = 1.

        predictions = model.predict(x_input, verbose=0)[0]
        
        if i == num_iterations-1:
            final_predicted = predictions
        
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

### ---------- Evaluate and Graph ----------
loss = model.evaluate(x, y, batch_size=batch_size, verbose=1)
print("loss: ", loss)

print('loss history:')
print(loss_history)

# plt.figure(figsize=(15,8))
# plt.rc('font', size=20)
# plt.plot(loss_history, lw=3, c='orange')
# plt.title('Cross Entropy Loss of LSTM Model over Epoch Iterations', fontsize=25)
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.savefig("loss.png")
# plt.grid()
# plt.show()

# print(model.summary())
# plot_model(model, to_file='model_plot.png')