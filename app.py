import numpy as np
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from PIL import Image


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('encoded_dataset.csv') 

# Preprocess data
def preprocess_data(data):
    # Drop rows with missing values
    data.dropna(inplace=True)
    # Define features and target variable
    X = data.drop(['PlacedOrNot'], axis=1)  # Exclude 'name' column
    y = data['PlacedOrNot']
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Train the model
@st.cache_data
def train_model(X_train_scaled, X_test_scaled, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    return model

# Main function to run the Streamlit app
def main():
    img = Image.open('Job-Placement-Agency.jpg')
    st.image(img,width=650)
    st.title('Job Placement Prediction Model')
    # Load data
    data = load_data()
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(data)  # Retrieve scaler
    

    for _ in range(10):  # 10 iterations for example
        # Train model
        model = train_model(X_train_scaled, X_test_scaled, y_train)
        # Evaluate on training data
        train_pred = model.predict(X_train_scaled)
        # Evaluate on testing data
        test_pred = model.predict(X_test_scaled)


    st.sidebar.subheader('User Input')
    # Collect user input features
    user_input = {}
    for feature in data.drop(['PlacedOrNot'], axis=1).columns:  
        user_input[feature] = st.sidebar.number_input(f'Enter {feature}', min_value=0, max_value=100)

    # Make predictions
    st.write('Prediction:')
    if st.sidebar.button('Predict'):
        user_data = pd.DataFrame([user_input])
        scaled_user_data = scaler.transform(user_data)
        prediction = model.predict(scaled_user_data)
        if prediction[0] == 1:
            st.write('Congratulations! The model predicts that you will get placed.')
        else:
            st.write('Sorry! The model predicts that you will not get placed.')
    
if __name__ == '__main__':
    main()
