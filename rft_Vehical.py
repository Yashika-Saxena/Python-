import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the training data
X_train = pd.read_csv('vehicular_communication_data_train.csv')

# Encode the categorical 'modulation_and_coding_scheme' column
#Label encoding assigns a unique integer value to each category
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(X_train['modulation_and_coding_scheme'])

# Encode the categorical 'traffic_requirements' column using one-hot encoding
# One-hot encoding creates a binary vector for each category
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
traffic_requirements_encoded = onehot_encoder.fit_transform(X_train[['traffic_requirements']])

# Combine the encoded 'traffic_requirements' with other features
X_train = np.column_stack((X_train['channel_conditions'], traffic_requirements_encoded))

# Train the model RANDOM FOREST
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train_encoded)

# Load the testing data
X_test = pd.read_csv('vehicular_communication_data_test1.csv')

# Encode the categorical 'modulation_and_coding_scheme' column of the testing data
y_test_encoded = label_encoder.transform(X_test['modulation_and_coding_scheme'])

# Encode the categorical 'traffic_requirements' column of the testing data using one-hot encoding
traffic_requirements_encoded = onehot_encoder.transform(X_test[['traffic_requirements']])

# Combine the encoded 'traffic_requirements' with other features of the testing data
X_test = np.column_stack((X_test['channel_conditions'], traffic_requirements_encoded))

# Evaluate the model on the test set
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test_encoded)

    # Print the accuracy
print('Accuracy:', accuracy)

