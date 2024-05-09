import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from preprocess import ts_songs_pre_precessed
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping
import os

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(666)

# Set the random seed in tensorflow at graph level
tf.random.set_seed(666)

# Disable the tensorflow GPU deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Disable the tensorflow GPU non deterministic collective operations
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# from src.results import save_metrics

# Load your data
df = ts_songs_pre_precessed()

# Preprocess your data
# Assuming 'song_vector' is your feature and 'album' is your label
X = np.array(df['song_vector'].tolist())

# Pad sequences
X = pad_sequences(X, padding='post')

# Normalize vectors
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = df['Album']

# Convert your labels to integers
le = LabelEncoder()
y = le.fit_transform(y)

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Define your model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(300, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(150, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(75, activation='relu'),    
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

# Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Train your model
model.fit(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[early_stopping])

# Predict values for test set
y_pred = model.predict(X_test)

# Convert predictions to label indices
y_pred = np.argmax(y_pred, axis=1)

# Print actual vs predicted values
for actual, predicted in zip(y_test, y_pred):
    print(f'Actual: {le.inverse_transform([actual])[0]} x Predicted: {le.inverse_transform([predicted])[0]}')

# Evaluate your model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Accuracy: {accuracy*100}%')