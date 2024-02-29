from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.realpath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)

from src.preprocess import ts_songs_pre_precessed
from src.results import save_metrics

# get args to see if we want to use KNN or SVM
model = 'svm'  # default value
test_size = 0.3  # default value

if len(sys.argv) > 1:
    model = sys.argv[1]

if len(sys.argv) > 2:
    try:
        test_size = float(sys.argv[2])
        if test_size < 0 or test_size > 1:
            print("Invalid test size, setting to default value (0.3)")
            test_size = 0.3
    except ValueError:
        print("Invalid test size, setting to default value (0.3)")
        test_size = 0.3

df = ts_songs_pre_precessed()

# Split the data into training and testing sets
X = np.stack(df['song_vector'].to_numpy())
y = df['Album'].to_numpy()


# Get the indices for training and testing
train_indices, test_indices = train_test_split(df.index, test_size=test_size, random_state=31, stratify=y)

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

y_pred = None
if model == 'knn':
    # Fit the KNN model
    n = 5  # default value

    if len(sys.argv) > 3:
        try:
            n = int(sys.argv[3])
            if n < 1 or n > 15:
                print("Invalid k, must be between 1 and 15. Setting to default value (5)")
                n = 5
        except ValueError:
            print("Invalid k, must be an integer. Setting to default value (5)")
            n = 5

    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train_encoded)

    # Predict the album for each song in the test set
    y_pred = knn.predict(X_test)


elif model == 'svm':
    # Initialize and train the SVM
    svm = SVC(kernel='linear', C=1, probability=True, random_state=13, class_weight='balanced')
    svm.fit(X_train, y_train_encoded)

    # Predict the album for each song in the test set
    y_pred = svm.predict(X_test)

else:
    print("Invalid model")
    sys.exit(1)

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
dest_file_path = os.path.join(project_root, 'assets', 'predictions.xlsx')
df_test = df_test[['Song Name', 'Album', 'predicted_album']]
df_test.to_excel(dest_file_path, index=False)
# print(df_test[['Song Name', 'Album', 'predicted_album']])

print(f'\n{model.upper()} Accuracy: {accuracy}')
print(f'{model.upper()} Precision: {precision}')
print(f'{model.upper()} Recall: {recall}')
print(f'{model.upper()} F1-score: {f1}')

# Save the metrics to the results file
if model == 'knn':
    save_metrics('album', 'KNN', accuracy, precision, recall, f1, test_size, n)
elif model == 'svm':
    save_metrics('album', 'SVM', accuracy, precision, recall, f1, test_size, 0)
else:
    print("Invalid model")
    sys.exit(1)