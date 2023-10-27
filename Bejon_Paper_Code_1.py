import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
def plot_graph(history, string, model_id, title):
    plt.plot(history.history[string], 'b-o', label=string)
    plt.plot(history.history['val_' + string], 'r-o', label='val_' + string)
    plt.xlabel(string + ' vs val ' + string)
    plt.legend()
    plt.title(title)
    plt.savefig('learning_curve_' + string + '_' + model_id + '.png')
    plt.show()
    
actions=['hello','iloveyou','thanks']
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

# Folder start
start_folder = 30
label_map = {label:num for num, label in enumerate(actions)}
DATA_PATH=r'G:\Other_Than_PhD\Abdur Rahim Vi\Bejon Paper\Code\Dataset\Dataset'
sequences, labels = [], []
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])    

y = to_categorical(labels).astype(int)
X = np.array(sequences)
print(X.shape)  #dfjlf  180*30*1680


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
history =model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
model.summary()
'''



import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense

# Define your model

model = Sequential()

# Add a 1D convolutional layer
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(21,3)))
model.add(MaxPooling1D(pool_size=2))
# Add LSTM layer(s)
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
# You can add more LSTM layers as needed

# Flatten the output
model.add(Flatten())
# Add dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 output classes, use softmax for multi-class classification
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()


epochs = 1000
batch_size = 32

model= tf.keras.Sequential([
      tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(30,1662)),
      tf.keras.layers.MaxPooling1D(2),
      tf.keras.layers.Conv1D(128, 3, activation='relu'),
      tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'),
      tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'),
      #tf.keras.layers.MaxPooling1D(2),
      #tf.keras.layers.LSTM(64),
      #tf.keras.layers.LSTM(64, return_sequences=True),
      tf.keras.layers.Dense(256, activation='relu'),
      tf.keras.layers.Dropout(0.4),
      tf.keras.layers.Dense(3, activation='softmax')
])

model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=16, kernel_size=2, padding='valid', activation='relu', input_shape=(30,1662)),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.Conv1D(filters=64, kernel_size=4, padding='valid', activation='relu'),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
history = model.fit(X_train,  y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
'''



# Train the model

plot_graph(history, 'accuracy', '2', 'LSTM-CNN')
plot_graph(history, 'loss', '2', 'LSTM-CNN')

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
