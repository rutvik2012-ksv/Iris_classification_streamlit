import streamlit as st
import joblib
import pandas as pd

# Load the KNN model from the joblib file
model_filename = 'knn_model.joblib'
loaded_knn = joblib.load(model_filename)

# Streamlit app
st.title('Iris Species Prediction using KNN')

# Input fields for Iris dataset features
st.header('Enter the Iris flower measurements:')
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.1, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.5, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=1.4, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# When the user clicks the Predict button
if st.button('Predict'):
    # Create a DataFrame from the user input
    input_data = {
        'sepal length (cm)': [sepal_length],
        'sepal width (cm)': [sepal_width],
        'petal length (cm)': [petal_length],
        'petal width (cm)': [petal_width]
    }
    input_df = pd.DataFrame(input_data)
    
    # Make a prediction using the loaded KNN model
    prediction = loaded_knn.predict(input_df)
    predicted_species = loaded_knn.target_names[prediction[0]]
    
    # Display the prediction
    st.success(f'The predicted species is: {predicted_species}')

# Run the app: streamlit run iris_knn_app.py
