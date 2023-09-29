import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from collections import OrderedDict

# Define the file path for the saved model
model_file = 'extra_trees_model.pkl'

# Load the trained model
model = joblib.load(model_file)

# Create a Streamlit web app
st.title('Accident Severity Prediction')

# Create input widgets for feature values
number_of_casualties = st.number_input('Number of Casualties', min_value=0)
minute = st.slider('Minute', 0, 59)
age_band_of_driver = st.selectbox('Age Band of Driver', ['0-5', '6-10', '11-15', '16-20', '21-25', '26-35', '36-45', '46-55', '56-65', '66-75', 'Over 75'])
number_of_vehicles_involved = st.number_input('Number of Vehicles Involved', min_value=1)
light_conditions = st.selectbox('Light Conditions', ['Daylight', 'Darkness - lights lit', 'Darkness - lights unlit', 'Darkness - no lighting', 'Darkness - lighting unknown'])
day_of_week = st.selectbox('Day of Week', ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
types_of_junction = st.selectbox('Types of Junction', ['Not at junction or within 20 metres', 'Roundabout', 'Mini-roundabout', 'T or staggered junction', 'Crossroads', 'More than 4 arms (not roundabout)', 'Private drive or entrance', 'Other junction', 'Roundabout and mini-roundabout'])
session = st.selectbox('Session', ['Morning', 'Afternoon', 'Evening', 'Night'])
hour = st.slider('Hour', 0, 23)
lanes_or_medians = st.number_input('Number of Lanes or Medians', min_value=1)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'Number_of_casualties': [number_of_casualties],
    'minute': [minute],
    'Age_band_of_driver': [age_band_of_driver],
    'Number_of_vehicles_involved': [number_of_vehicles_involved],
    'Light_conditions': [light_conditions],
    'Day_of_week': [day_of_week],
    'Types_of_Junction': [types_of_junction],
    'session': [session],
    'hour': [hour],
    'Lanes_or_Medians': [lanes_or_medians]
})

# Predict Accident Severity
if st.button('Predict Accident Severity'):
    # Preprocess input data (e.g., one-hot encoding for categorical variables)
    # Make sure to preprocess the input data in the same way as during model training

    # Perform prediction using the loaded model
    prediction = model.predict(input_data)

    # Display the predicted Accident Severity
    st.subheader('Predicted Accident Severity:')
    st.write(prediction[0])
