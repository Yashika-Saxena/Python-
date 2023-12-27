import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
warnings.filterwarnings('ignore')
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred

def main():
    # Load the training data
    X_train = pd.read_csv('_vehicular_communication_data_train20data.csv')

    # Encode the categorical 'modulation_and_coding_scheme' column
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(X_train['modulation_and_coding_scheme'])
    #print(y_train_encoded)

    # Encode the categorical 'traffic_requirements' column using one-hot encoding
    onehot_encoder = OneHotEncoder(sparse=False, drop='first')
    traffic_requirements_encoded = onehot_encoder.fit_transform(X_train[['traffic_requirements']])
    #print( traffic_requirements_encoded)

    # Combine the encoded 'traffic_reqauirements' with other features
    X_train = np.column_stack((X_train['channel_conditions'], traffic_requirements_encoded))
    #print(X_train)
    # Train the model
    model = train_model(X_train, y_train_encoded)
    

    # Load the testing data
    X_test = pd.read_csv('_vehicular_communication_data_testnew1.csv')

    # Encode the categorical 'modulation_and_coding_scheme' column of the testing data
    y_test_encoded = label_encoder.transform(X_test['modulation_and_coding_scheme'])
    print(y_test_encoded)

    # Encode the categorical 'traffic_requirements' column of the testing data using one-hot encoding
    traffic_requirements_encoded = onehot_encoder.transform(X_test[['traffic_requirements']])
    #print(traffic_requirements_encoded)
    # Combine the encoded 'traffic_requirements' with other features of the testing data
    X_test = np.column_stack((X_test['channel_conditions'], traffic_requirements_encoded))

    # Evaluate the model on the test set
    y_pred = predict(model, X_test)
    print(y_pred)
    accuracy = np.mean(y_pred == y_test_encoded)

    # Print the accuracy
    print('Accuracy:', accuracy)

if __name__ == '__main__':
    main()