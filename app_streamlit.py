import streamlit as st
from utils import load_model, load_encoders, preprocess_input

st.set_page_config(page_title="IPL Winner Predictor", layout="centered")

st.title("🏏 IPL Winner Prediction App")
st.markdown("Predict the winning team based on match details.")

# Load encoders for dropdowns
encoders = load_encoders()

team_names = encoders["team1"].classes_
venues = encoders["venue"].classes_

col1, col2 = st.columns(2)
team1 = col1.selectbox("Select Team 1", team_names)
team2 = col2.selectbox("Select Team 2", team_names)

venue = st.selectbox("Select Venue", venues)
toss_winner = st.selectbox("Toss Winner", team_names)
toss_decision = st.selectbox("Toss Decision", encoders["toss_decision"].classes_)

if st.button("Predict Winner"):
    input_data = {
        "team1": team1,
        "team2": team2,
        "venue": venue,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision
    }
    model = load_model()
    processed = preprocess_input(input_data)
    prediction = model.predict(processed)[0]
    winner = encoders["winner"].inverse_transform([prediction])[0]
    st.success(f"🏆 Predicted Winner: {winner}")
