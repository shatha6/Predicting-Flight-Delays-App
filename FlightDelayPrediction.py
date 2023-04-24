import streamlit as st
import pandas as pd
import pickle
from datetime import datetime, time

# Load the trained model
with open("flightdelay.pkl","rb") as f: 
    MODEL = pickle.load(f)


# creating a function for Prediction
def model_pred(departure_date_time, origin, destination, model=MODEL):
    """_summary_

    Args:
        departure_date_time (_type_): _description_
        origin (_type_): _description_
        destination (_type_): _description_

    Returns:
        _type_: _description_
    """

    try:
        departure_date_time_parsed = datetime.strptime(departure_date_time, '%Y-%m-%d %H:%M:%S')
    except ValueError as e:
        return 'Error parsing date/time - {}'.format(e)

    month = departure_date_time_parsed.month
    day = departure_date_time_parsed.day
    day_of_week = departure_date_time_parsed.isoweekday()
    hour = departure_date_time_parsed.hour

    origin = origin.upper()
    destination = destination.upper()

    input = [{'MONTH': month,
              'DAY_OF_MONTH': day,
              'DAY_OF_WEEK': day_of_week,
              'CRS_DEP_TIME': hour,
              'ORIGIN_ATL': 1 if origin == 'ATL' else 0,
              'ORIGIN_DTW': 1 if origin == 'DTW' else 0,
              'ORIGIN_JFK': 1 if origin == 'JFK' else 0,
              'ORIGIN_MSP': 1 if origin == 'MSP' else 0,
              'ORIGIN_SEA': 1 if origin == 'SEA' else 0,
              'DEST_ATL': 1 if destination == 'ATL' else 0,
              'DEST_DTW': 1 if destination == 'DTW' else 0,
              'DEST_JFK': 1 if destination == 'JFK' else 0,
              'DEST_MSP': 1 if destination == 'MSP' else 0,
              'DEST_SEA': 1 if destination == 'SEA' else 0 }]

    return model.predict_proba(pd.DataFrame(input))[0][0]

# Define the app
def main():
    st.title('Flight Delay Prediction')
    st.write('Enter the following information about your flight to predict the likelihood of a delay.')
    
    # Create the form for user input
    form = st.form(key='flight_form')
    origin = form.selectbox('Origin Airport', ['ATL', 'DTW', 'SEA', 'MSP', 'JFK'])
    dest = form.selectbox('Destination Airport', ['SEA', 'MSP', 'DTW', 'ATL', 'JFK'])
    date_input = form.date_input("Select a date", datetime.today())
    time_input = form.time_input("Select a time", time(hour=0, minute=0, second=0))
    dep_time = datetime.combine(date_input, time_input)
    # dep_time = form.date_input('Departure Date')
    form_submit = form.form_submit_button(label='Predict')
    
    # When the user submits the form
    if form_submit:
        # Make a prediction using the trained model
        departure_date_time = dep_time.strftime('%Y-%m-%d %H:%M:%S')
        prediction = model_pred(departure_date_time, origin, dest, model=MODEL)

        
        # Show the prediction to the user
        st.write(f'The probability of a delay for your flight is {prediction:.0%}.')
        st.success(prediction)

if __name__ == '__main__':
    main()
