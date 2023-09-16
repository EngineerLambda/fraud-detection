import streamlit as st
import numpy as np
import joblib
from geopy.geocoders import Nominatim
from datetime import datetime, time


st.set_page_config(page_title="Fraud Detetion webapp", page_icon="💸", layout="wide")
st.title("Fraud Detection Web Application", )
st.info("An interractive web interface to determine whether a transaction is to considered fraudulent or not")
def get_lat_and_long(user_city):
    geolocator = Nominatim(user_agent="MyApp")
    location = geolocator.geocode(user_city)
    return float(location.latitude), float(location.longitude)

def predict_and_score():
    pred = model.predict([full_feats])
    if pred == 1:
        st.error("This transcation is most likely fradulent")
    elif pred == 0:
        st.success("This transaction is determined to be likely safe")

# Loading the external resources saved from the project notebook
city_details = joblib.load("city_details.pkl")
cities = city_details.keys()

categories = joblib.load("categories.pkl")
jobs = joblib.load("jobs.pkl")
states = joblib.load("states.pkl")


scaler = joblib.load("robust_scaler.pkl")
encoder =joblib.load("label_encoder.pkl")

model = joblib.load("log_reg.pkl")
# Taking inputs from the web app
amt = int(st.number_input("Enter the transaction amount here", step=100))

col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.title("Personal info")
    category = st.selectbox("Select the transaction category", categories)
    gender = st.selectbox("Choose your Gender", ["M", "F"])
    job = st.selectbox("What is your job/occupation", jobs)
    state = st.selectbox("Which state are you loated?", states)

with col2:
    st.title("Location info")
    city = st.selectbox("Which city are you located?", cities)
    zipcode = int(st.number_input("Type in your zip code", step=1
                                  ))
    st.info("The app is using the default latitude and longitude of your selected city")
    latitude, longitude = get_lat_and_long(city)
    city_pop = int(city_details[city])
    st.success(f"For your selected city, the corresponding lat and long are {latitude} and {longitude} respectively. Having a population of {city_pop}")

with col3:
    st.title("More details")
    date = st.date_input("What day did you execute the transaction?")

    time = st.time_input("And what time of the day?:", value=time(12, 0), step=60)
    
    date_time = f"{date} {time}"
    date_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S")
    unix_time = int(datetime.timestamp(date_time))

    merch_lat = latitude
    merch_long = longitude

    numerical = [amt, zipcode, latitude, longitude, city_pop, unix_time, merch_lat, merch_long]
    categorical = [category, gender, city, state, job]

    numerical_scaled = scaler.transform([numerical])
    categorical_encoded = encoder.fit_transform(categorical)

    full_feats = np.hstack([numerical_scaled[0], categorical_encoded])
    
    if st.button("Click to Determine"):
        predict_and_score()