from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

from preprocess import songs_pre_precessed


df = songs_pre_precessed()
df.to_excel('taylor_swift_lyrics_preprocessed.xlsx', index=False)

# Split the data into training and testing sets

X = np.stack(df['song_vector'].to_numpy())
y = df['Album'].to_numpy()

# Get the indices for training and testing
train_indices, test_indices = train_test_split(df.index, test_size=0.4, random_state=9137)

# Create your training and testing data
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Now you can use test_indices to create df_test
df_test = df.loc[test_indices].copy()

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit the encoder and transform y_train
y_train_encoded = label_encoder.fit_transform(y_train.astype(str))

# Transform y_test
y_test_encoded = label_encoder.transform(y_test.astype(str))

# Initialize and train the SVM
svm = SVC(kernel='rbf', C=1, probability=True, random_state=13, class_weight='balanced')
svm.fit(X_train, y_train_encoded)

# Predict the album for each song in the test set
y_pred = svm.predict(X_test)
y_pred_decoded = label_encoder.inverse_transform(y_pred)
y_test_decoded = label_encoder.inverse_transform(y_test_encoded)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(y_test_decoded, y_pred_decoded) # This represents how many of the predictions were correct -> TP + TN / (TP + TN + FP + FN)
report = classification_report(y_test_decoded, y_pred_decoded, output_dict=True, zero_division=1) 
precision = report['weighted avg']['precision'] # This represents the proportion of positive identifications that were actually correct -> TP / (TP + FP)
recall = report['weighted avg']['recall'] # This represents the proportion of actual positive cases which were correctly identified -> TP / (TP + FN)
f1 = report['weighted avg']['f1-score'] # This represents the balance between precision and recall -> 2 * (precision * recall) / (precision + recall)

# Add predicted album to the DataFrame
df_test['predicted_album'] = y_pred_decoded

# Display the actual and predicted albums for each song
df_test.to_excel('predicitions.xlsx', index=False)
print(df_test[['Song Name', 'Album', 'predicted_album']])
print(f'SVM Accuracy: {accuracy}')
print(f'SVM Precision: {precision}')
print(f'SVM Recall: {recall}')
print(f'SVM F1-score: {f1}')