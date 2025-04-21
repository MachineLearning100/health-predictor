import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Sample data
data = {
    'EatApple': [1, 2, 3, 4, 5, 2, 3, 5, 1, 4],
    'SleepHours': [5, 6, 7, 6, 8, 7, 8, 9, 5, 7],
    'ExerciseMinutes': [10, 20, 30, 15, 40, 25, 35, 50, 10, 20],
    'WaterCups': [2, 4, 5, 3, 6, 5, 6, 8, 2, 5],
    'HealthScore': [50, 60, 70, 65, 85, 68, 80, 95, 45, 75]
}
df = pd.DataFrame(data)

# Train model
X = df[['EatApple', 'SleepHours', 'ExerciseMinutes', 'WaterCups']]
y = df['HealthScore']
model = LinearRegression()
model.fit(X, y)

# Streamlit app
st.title("🍎 Health Score Predictor")
st.write("Estimate your health score based on your daily habits!")

eat_apples = st.slider("How many apples do you eat daily?", 0, 10, 3)
sleep_hours = st.slider("How many hours do you sleep?", 0, 12, 7)
exercise = st.slider("How many minutes do you exercise?", 0, 120, 30)
water = st.slider("How many cups of water do you drink?", 0, 12, 4)

input_data = [[eat_apples, sleep_hours, exercise, water]]
prediction = model.predict(input_data)[0]

st.metric("🩺 Predicted Health Score", f"{round(prediction, 1)} / 100")
