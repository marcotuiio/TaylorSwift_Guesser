from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from src.preprocess import ts_songs_pre_precessed

# get args to see if we want to use KNN or SVM
model = sys.argv[1]

df = ts_songs_pre_precessed()

# Split the data into training and testing sets
X = np.stack(df['song_vector'].to_numpy())
y = df['Album'].to_numpy()

# Get the indices for training and testing
train_indices, test_indices = train_test_split(df.index, test_size=0.3, random_state=31, stratify=y)

# Initialize LeaveOneOut
loo = LeaveOneOut()

# Lists to store predictions and ground truth
predictions = []
ground_truth = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the model (either KNN or SVM)
    if model == 'knn':
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

    elif model == 'svm':
        svm = SVC(kernel='linear', C=1, probability=True, random_state=31, class_weight='balanced')
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
    else:
        print("Invalid model")
        sys.exit(1)

    # Store the prediction and ground truth
    predictions.append(y_pred[0])
    ground_truth.append(y_test[0])

    # Print the input data, real label, and predicted label
    print(f"Real Label: {y_test[0]}")
    print(f"Predicted Label: {y_pred[0]}\n")

# Convert the lists to arrays for easier calculation
predictions = np.array(predictions)
ground_truth = np.array(ground_truth)

# Calculate accuracy, precision, recall, and F1-score
accuracy = accuracy_score(ground_truth, predictions)
report = classification_report(ground_truth, predictions, output_dict=True, zero_division=1)
precision = report['weighted avg']['precision']
recall = report['weighted avg']['recall']
f1 = report['weighted avg']['f1-score']

# Print the metrics
print(f'\n{model.upper()} Accuracy: {accuracy}')
print(f'{model.upper()} Precision: {precision}')
print(f'{model.upper()} Recall: {recall}')
print(f'{model.upper()} F1-score: {f1}')
