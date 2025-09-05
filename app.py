import streamlit as st
import pandas as pd
import pickle

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

# Load trained model
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>🏆 IPL Win Predictor 🏆</h1>", unsafe_allow_html=True)
st.write("---")

teams = sorted([
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals'
])

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('🏏 Select Batting Team', teams)
with col2:
    bowling_team = st.selectbox('🎯 Select Bowling Team', teams)

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

selected_city = st.selectbox('🌆 Match Venue', sorted(cities))

target = st.number_input('🎯 Target Score', min_value=0)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('📊 Current Score', min_value=0)
with col4:
    wickets = st.number_input('❌ Wickets Down', min_value=0, max_value=9)
with col5:
    overs = st.number_input('⏱ Overs Completed', min_value=0, max_value=20)

if st.button('🔥 Predict Winning Probability'):
    runs_left = target - score
    balls_left = 120 - overs * 6
    wickets_remaining = 10 - wickets
    crr = score / overs if overs > 0 else 0
    rrr = (runs_left * 6 / balls_left) if balls_left > 0 else 0

    df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [selected_city],
        'runs_left': [runs_left],
        'balls_left': [balls_left],
        'wickets': [wickets_remaining],
        'total_runs_x': [target],
        'crr': [crr],
        'rrr': [rrr]
    })

    result = pipe.predict_proba(df)
    r_1 = round(result[0][0] * 100)
    r_2 = round(result[0][1] * 100)

    st.write("---")
    st.markdown("<h2 style='text-align: center;'>📊 Winning Probability</h2>", unsafe_allow_html=True)

    # Show progress bars and metrics
    col6, col7 = st.columns(2)
    with col6:
        st.metric(label=f"{batting_team}", value=f"{r_2}%")
        st.progress(r_2 / 100)
    with col7:
        st.metric(label=f"{bowling_team}", value=f"{r_1}%")
        st.progress(r_1 / 100)

    # Winner celebrations
    if r_2 > r_1:
        st.success(f"🎉 {batting_team} are more likely to win! 🎉")
        st.balloons()
    elif r_1 > r_2:
        st.error(f"😭 {bowling_team} are dominating the match! 😭")
        st.snow()
    else:
        st.warning("🤝 It's looking like a tie! Super Over loading...")

    st.write("---")
    st.markdown("<h4 style='text-align: center; color: gray;'>⚡ Powered by Machine Learning & AI ⚡</h4>", unsafe_allow_html=True)
