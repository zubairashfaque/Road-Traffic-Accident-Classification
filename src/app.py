import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import f1_score

# Load the trained ExtraTreesClassifier model
loaded_model = joblib.load('model/extra_trees_model.pkl')

# Load ordinal encoder mapping
loaded_encoder = joblib.load('data/processed/ordinal_encoder_final.pkl')

# Load the test dataset (assuming you have a test dataset)
test_data = pd.read_csv('data/processed/sample_data/oversampler_adasyn_0_test.csv')

# Define a dictionary to map numerical labels to injury severity labels
severity_mapping = {
    1: "Slight injury",
    2: "Serious injury",
    3: "Fatal injury"
}

# Define a dictionary to map numerical predictions to categories or symbols
severity_mapping_display = {
    1: "⚠️",  # Display a warning symbol for "Slight injury"
    2: "⚕️",  # Display a medical symbol for "Serious injury"
    3: "☠️"   # Display a skull symbol for "Fatal injury"
}

# Function to perform the preprocessing steps
def preprocess_data(input_data):
    # Define a function to categorize the 'hour' into sessions
    def divide_day(x):
        if (x > 4) and (x <= 8):
            return 'Early Morning'
        elif (x > 8) and (x <= 12):
            return 'Morning'
        elif (x > 12) and (x <= 16):
            return 'Noon'
        elif (x > 16) and (x <= 20):
            return 'Evening'
        elif (x > 20) and (x <= 24):
            return 'Night'
        elif x <= 4:
            return 'Late Night'

    input_data['session'] = input_data['hour'].apply(divide_day)

    columns_to_encode = ['Age_band_of_driver', 'Light_conditions', 'Day_of_week', 'Types_of_Junction',
                         'Lanes_or_Medians', 'session']

    # Apply ordinal encoding to the selected columns
    input_data[columns_to_encode] = loaded_encoder.transform(input_data[columns_to_encode])

    print(input_data)

    return input_data

# Create a Streamlit web app with a white background
st.set_page_config(page_title='Accident Severity Prediction', page_icon='🚗', layout='wide',
                   initial_sidebar_state='auto')

# Set the background color to white
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a title and description
st.markdown("<h1 style='text-align: center;'>Accident Severity Prediction App 🚧</h1>", unsafe_allow_html=True)
st.write("<hr>", unsafe_allow_html=True)
st.write("<div style='text-align: center;'>This app predicts the accident severity based on user input.</div>", unsafe_allow_html=True)

# Create a Streamlit sidebar for input widgets
st.sidebar.title('Input Features')

# Define input widgets for feature values (with descriptions)
number_of_casualties = st.sidebar.number_input('Number of Casualties', min_value=0, help="Enter the number of casualties.")
minute = st.sidebar.slider('Minute', 0, 59, help="Select the minute of the accident.")
Age_band_of_driver = st.sidebar.selectbox('Age Band of Driver', ['18-30', '31-50', 'Under 18', 'Over 51'],
                                         help="Select the age band of the driver.")
number_of_vehicles_involved = st.sidebar.number_input('Number of Vehicles Involved', min_value=1,
                                                     help="Enter the number of vehicles involved.")
Light_conditions = st.sidebar.selectbox('Light Conditions',
                                        ['Daylight', 'Darkness - lights lit', 'Darkness - no lighting',
                                         'Darkness - lights unlit'], help="Select the light conditions.")
Day_of_week = st.sidebar.selectbox('Day of Week',
                                   ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                   help="Select the day of the week.")
Types_of_Junction = st.sidebar.selectbox('Types of Junction',
                                         ['Crossing', 'No junction', 'O Shape', 'Other', 'T Shape', 'Unknown',
                                          'X Shape', 'Y Shape'], help="Select the type of junction.")
session = st.sidebar.selectbox('Session', ['Early Morning', 'Morning', 'Noon', 'Evening', 'Night', 'Late Night'],
                              help="Select the session of the day.")
hour = st.sidebar.slider('Hour', 0, 23, help="Select the hour of the accident.")
Lanes_or_Medians = st.sidebar.selectbox('Lanes or Medians',
                                        ['Undivided Two way', 'other', 'Double carriageway (median)', 'One way',
                                         'Two-way (divided with solid lines road marking)',
                                         'Two-way (divided with broken lines road marking)'],
                                        help="Select the lanes or medians type.")

# Define a function to make predictions
def predict_severity(number_of_casualties, minute, Age_band_of_driver, number_of_vehicles_involved,
                     Light_conditions, Day_of_week, Types_of_Junction, session, hour, Lanes_or_Medians):
    # Create a DataFrame with user input
    user_input = pd.DataFrame({
        'Number_of_casualties': [number_of_casualties],
        'minute': [minute],
        'Age_band_of_driver': [Age_band_of_driver],
        'Number_of_vehicles_involved': [number_of_vehicles_involved],
        'Light_conditions': [Light_conditions],
        'Day_of_week': [Day_of_week],
        'Types_of_Junction': [Types_of_Junction],
        'session': [session],
        'hour': [hour],
        'Lanes_or_Medians': [Lanes_or_Medians]
    })

    # Preprocess the user input data
    user_input = preprocess_data(user_input)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(user_input)

    return prediction[0]

# Add a button to trigger prediction
if st.sidebar.button('Predict Severity'):
    severity_num = predict_severity(number_of_casualties, minute, Age_band_of_driver, number_of_vehicles_involved,
                                    Light_conditions, Day_of_week, Types_of_Junction, session, hour, Lanes_or_Medians)

    # Map numerical severity to labels and symbols
    severity_label = severity_mapping.get(severity_num, "Unknown")
    severity_symbol = severity_mapping_display.get(severity_num, "❓")

    # Display the prediction label and symbol
    st.write("<div style='text-align: center; font-size: 24px;'>Predicted Severity:</div>", unsafe_allow_html=True)
    st.write(f"<div style='text-align: center; font-size: 36px; color: blue; font-weight: bold;'>{severity_label} {severity_symbol}</div>", unsafe_allow_html=True)

    # Calculate and display the F1 weighted score on the test set
    test_X = test_data.drop(['Accident_severity', 'kfold'], axis=1)
    selected_features = [
        'Number_of_casualties',
        'minute',
        'Age_band_of_driver',
        'Number_of_vehicles_involved',
        'Light_conditions',
        'Day_of_week',
        'Types_of_Junction',
        'session',
        'hour',
        'Lanes_or_Medians'
    ]
    test_X = test_X[selected_features]
    test_y = test_data['Accident_severity']  # Replace 'target_column' with the actual target column name

    test_predictions = loaded_model.predict(test_X)
    f1_weighted = f1_score(test_y, test_predictions, average='weighted')

    st.write(f"<div style='text-align: center; font-size: 18px;'>F1 Weighted Score on Test Set: {f1_weighted:.4f}</div>", unsafe_allow_html=True)

# Add a reset button
if st.sidebar.button('Reset'):
    # Add a confirmation dialog
    if st.sidebar.checkbox("Are you sure?"):
        # Clear input values by setting them to default values
        number_of_casualties = 0
        minute = 0
        Age_band_of_driver = '18-30'
        number_of_vehicles_involved = 1
        Light_conditions = 'Daylight'
        Day_of_week = 'Monday'
        Types_of_Junction = 'Crossing'
        session = 'Early Morning'
        hour = 0
        Lanes_or_Medians = 'Undivided Two way'

        # Rerun the app to reflect the updated input values
        st.experimental_rerun()
