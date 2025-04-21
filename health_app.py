import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

# UI
st.title("🍎 Health Score Predictor 2.0")
st.write("Now with 📊 data visualizations!")

# Sliders for user input
eat_apples = st.slider("🍏 Apples per day", 0, 10, 3)
sleep_hours = st.slider("😴 Sleep hours", 0, 12, 7)
exercise = st.slider("🏃 Exercise minutes", 0, 120, 30)
water = st.slider("💧 Water cups", 0, 12, 4)

# Prediction
user_input = [[eat_apples, sleep_hours, exercise, water]]
prediction = model.predict(user_input)[0]
st.metric("🩺 Your Predicted Health Score", f"{round(prediction, 1)} / 100")

# 📊 Bar chart of user's inputs
st.subheader("🔍 Your Daily Habits Breakdown")
user_df = pd.DataFrame({
    'Habit': ['Apples', 'Sleep', 'Exercise', 'Water'],
    'Value': [eat_apples, sleep_hours, exercise, water]
})
st.bar_chart(user_df.set_index('Habit'))

# 📈 Scatter plot: Exercise vs Health (colored by apples)
st.subheader("📈 Sample Data: Exercise vs Health Score")
fig, ax = plt.subplots()
scatter = ax.scatter(df['ExerciseMinutes'], df['HealthScore'],
                     c=df['EatApple'], cmap='YlGn', edgecolor='k')
ax.set_xlabel("Exercise Minutes")
ax.set_ylabel("Health Score")
ax.set_title("Training Data: Health vs Exercise")
plt.colorbar(scatter, label='🍎 Apples Eaten')
st.pyplot(fig)
