import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")
logistic_regression_model = joblib.load("logistic_regression_model.joblib")
knn_model = joblib.load("knn_model.joblib")
random_forest_model = joblib.load("random_forest_model.joblib")
st.title("Forest Cover Type")
elevation = st.slider('Elevation', min_value=1859, max_value=3858, value=2850)
aspect = st.slider('Aspect', min_value=0, max_value=360, value=180)
slope = st.slider('Slope', min_value=0, max_value=66, value=33)
horizontal_distance_to_hydrology = st.slider('Horizontal Distance To Hydrology', min_value=0, max_value=1397, value=700)
vertical_distance_to_hydrology = st.slider('Vertical Distance To Hydrology', min_value=-173, max_value=601, value=200)
horizontal_distance_to_roadways = st.slider('Horizontal Distance To Roadways', min_value=0, max_value=7117, value=3550)
hillshade_9am = st.slider('Hillshade 9am', min_value=0, max_value=254, value=127)
hillshade_noon = st.slider('Hillshade Noon', min_value=0, max_value=254, value=127)
hillshade_3pm = st.slider('Hillshade 3pm', min_value=0, max_value=254, value=127)
horizontal_distance_to_fire_points = st.slider('Horizontal Distance To Fire Points', min_value=0, max_value=7173, value=3600)
wilderness_area1 = st.slider('Wilderness_Area1', min_value=0, max_value=1,value=0)
wilderness_area2 = st.slider('Wilderness_Area2', min_value=0, max_value=1,value=0)
wilderness_area3 = st.slider('Wilderness_Area3', min_value=0, max_value=1,value=0)
wilderness_area4 = st.slider('Wilderness_Area4', min_value=0, max_value=1,value=0)

if st.button("Predict Forest Cover Type"):
    input_data = np.array([[elevation, aspect, slope, horizontal_distance_to_hydrology, vertical_distance_to_hydrology,horizontal_distance_to_roadways, hillshade_9am, hillshade_noon,hillshade_3pm,horizontal_distance_to_fire_points, wilderness_area1, wilderness_area2, wilderness_area3, wilderness_area4]])
    
    lr_pred = logistic_regression_model.predict(input_data)
    knn_pred = knn_model.predict(input_data)
    rf_pred = random_forest_model.predict(input_data)   
    predictions = [
        ("Logistic Regression", lr_pred[0]),
        ("KNN", knn_pred[0]),
        ("Random Forest", rf_pred[0]),
    ]

    st.subheader("Prediction Results")
    for model_name, pred in predictions:
        st.write(f"{model_name} Prediction: {pred}")
    