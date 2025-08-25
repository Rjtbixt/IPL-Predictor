import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

# Load trained in-match model
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Sunrisers Hyderabad','Mumbai Indians','Royal Challengers Bangalore',
    'Kolkata Knight Riders','Kings XI Punjab','Chennai Super Kings',
    'Rajasthan Royals','Delhi Capitals'
]

cities = [
    'Hyderabad','Bangalore','Mumbai','Indore','Kolkata','Delhi','Chandigarh',
    'Jaipur','Chennai','Cape Town','Port Elizabeth','Durban','Centurion',
    'East London','Johannesburg','Kimberley','Bloemfontein','Ahmedabad','Cuttack',
    'Nagpur','Dharamsala','Visakhapatnam','Pune','Raipur','Ranchi','Abu Dhabi',
    'Sharjah','Mohali','Bengaluru'
]

# App Title
st.title("🏏 IPL Predictor")

# Input Columns
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

# Extra Features
col6, col7 = st.columns(2)
with col6:
    match_stage = st.selectbox('Match Stage', ['Powerplay (1-6)', 'Middle Overs (7-15)', 'Death Overs (16-20)'])
with col7:
    home_adv = st.selectbox('Home Advantage', ['Yes', 'No'])

target = st.number_input('Target', min_value=0)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('Score', min_value=0)
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

# Base validation
predict_disabled = False
if batting_team == bowling_team:
    st.error("🚨 Batting team and Bowling team cannot be the same. Please select different teams.")
    predict_disabled = True

# Derived values
balls_left = 120 - int(overs * 6)
runs_left = target - score
wickets_left = 10 - wickets
crr = score / overs if overs > 0 else 0
rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

# Maximum possible runs and wickets left
max_possible_runs = balls_left * 6
max_possible_wickets = wickets_left

# Impossible scenario check
impossible_scenario = False
warning_msg = ""

if runs_left < 0:
    impossible_scenario = True
    warning_msg = "🚨 Score already exceeds target!"
elif wickets < 0 or wickets > 10:
    impossible_scenario = True
    warning_msg = "🚨 Invalid number of wickets!"
elif balls_left < 0:
    impossible_scenario = True
    warning_msg = "🚨 Overs exceeded maximum (20)!"
elif runs_left > max_possible_runs:
    impossible_scenario = True
    warning_msg = f"🚨 Impossible scenario: Cannot score {runs_left} runs in {balls_left} balls."
elif wickets_left > max_possible_wickets:
    impossible_scenario = True
    warning_msg = f"🚨 Impossible scenario: Cannot lose {wickets_left} wickets in remaining balls."

predict_disabled = predict_disabled or impossible_scenario
if impossible_scenario:
    st.error(warning_msg)

# Predict Button
if st.button('Predict Probability', disabled=predict_disabled):
    if overs == 0:
        st.warning("⚠️ Overs cannot be 0 when calculating run rate.")
    else:
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr],
            'match_stage': [match_stage],
            'home_advantage': [home_adv]
        })

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        st.markdown("### 📋 Match Summary")
        st.info(f"Target: {target} | Score: {score}/{wickets} in {overs} overs\n\n"
                f"Runs Left: {runs_left} | Balls Left: {balls_left}\n\n"
                f"CRR: {round(crr,2)} | RRR: {round(rrr,2)}")

        if rrr > crr:
            commentary = f"⚡ Pressure on {batting_team}! They need {runs_left} in {balls_left} balls."
        else:
            commentary = f"✅ {batting_team} are cruising! Run chase looks comfortable."
        st.markdown(f"### 🎙️ Commentary\n{commentary}")

        st.markdown("### 📊 Win Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(win*100),
            title={'text': f"{batting_team} Win %"},
            delta={'reference': 50},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "green"},
                   'steps': [
                       {'range': [0, 50], 'color': "#ff9999"},
                       {'range': [50, 100], 'color': "#99ff99"}]
                  }
        ))
        st.plotly_chart(fig)

        st.success(f"{batting_team} – {round(win * 100)}% chance to win")
        st.error(f"{bowling_team} – {round(loss * 100)}% chance to win")
