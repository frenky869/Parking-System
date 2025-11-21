import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from pandasai import SmartDataframe
from pandasai.llm import Groq
import random

# --- 1. Configuration & Setup ---
st.set_page_config(page_title="Nairobi Smart Parking", page_icon="🚗", layout="wide")

# Initialize LLM for Admin Queries
def get_llm():
    try:
        return Groq(api_token=st.secrets["GROQ_API_KEY"], model="llama3-8b-8192")
    except:
        return None

llm = get_llm()

# --- 2. Data Simulation (The "Digital Twin" of Nairobi CBD) ---
@st.cache_data
def generate_parking_data():
    """
    Simulates 40 parking spots in Nairobi CBD with historical data features.
    """
    # Center: Kenyatta Ave / Kimathi St intersection approx
    center_lat, center_lon = -1.285790, 36.821835
    
    spots = []
    streets = ['Kenyatta Ave', 'Kimathi St', 'Moi Ave', 'Mama Ngina St', 'Banda St']
    
    for i in range(40):
        # Randomize location slightly around CBD
        lat = center_lat + random.uniform(-0.002, 0.002)
        lon = center_lon + random.uniform(-0.002, 0.002)
        
        # Simulate features for ML
        is_occupied = random.choice([True, False])
        base_fee = random.choice([100, 200]) # KES per hour
        
        spots.append({
            'spot_id': f'CBD-{i+1:03d}',
            'street': random.choice(streets),
            'latitude': lat,
            'longitude': lon,
            'is_occupied': is_occupied,
            'base_fee': base_fee,
            'occupancy_history_avg': random.uniform(0.3, 0.9), # 30% to 90% avg usage
            'hour_of_day': datetime.now().hour
        })
    
    return pd.DataFrame(spots)

# Load or initialize data
if 'parking_data' not in st.session_state:
    st.session_state.parking_data = generate_parking_data()

if 'reservations' not in st.session_state:
    st.session_state.reservations = []

# --- 3. Data Science Algorithms ---

def dynamic_pricing_engine(base_fee, occupancy_rate, duration_hours):
    """
    DS Algorithm: Calculates price based on demand elasticity.
    If the area is busy (>80%), price surges to discourage long stays.
    """
    demand_multiplier = 1.0
    if occupancy_rate > 0.8:
        demand_multiplier = 1.5  # Surge pricing
    elif occupancy_rate < 0.3:
        demand_multiplier = 0.8  # Discount for low demand
        
    # Fairness logic: Cheaper rate per hour if parking longer? 
    # Or expensive to discourage hoarding? Let's use linear for now.
    total_cost = base_fee * duration_hours * demand_multiplier
    return round(total_cost, 0), demand_multiplier

def train_availability_model(df):
    """
    DS Algorithm: Trains a simplified ML model to predict if a spot 
    will be free in the next hour based on history.
    """
    # Mock training data creation
    X = df[['latitude', 'longitude', 'base_fee', 'occupancy_history_avg']]
    y = [1 if x < 0.7 else 0 for x in df['occupancy_history_avg']] # 1 = Likely Free, 0 = Likely Full
    
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

# --- 4. UI: Driver Interface ---

def driver_view():
    st.header("🚗 Driver Portal: Find & Reserve Parking")
    
    df = st.session_state.parking_data
    
    # Metrics
    total_spots = len(df)
    free_spots = len(df[~df['is_occupied']])
    occupancy = (total_spots - free_spots) / total_spots
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Available Spots", free_spots, delta=f"{free_spots/total_spots:.0%} Free")
    col2.metric("Current CBD Congestion", f"{occupancy:.0%}", delta_color="inverse")
    
    # --- Map Visualization ---
    st.subheader("📍 Live Parking Map (Nairobi CBD)")
    
    # Create Folium Map
    m = folium.Map(location=[-1.285790, 36.821835], zoom_start=16)
    
    # Add spots to map
    for idx, row in df.iterrows():
        color = "red" if row['is_occupied'] else "green"
        status = "Occupied" if row['is_occupied'] else "Free"
        
        # Predictive Insight
        ml_model = train_availability_model(df)
        prediction = ml_model.predict([[row['latitude'], row['longitude'], row['base_fee'], row['occupancy_history_avg']]])[0]
        pred_text = "High chance of being free" if prediction == 1 else "High demand area"
        
        popup_html = f"""
        <b>Spot:</b> {row['spot_id']}<br>
        <b>Street:</b> {row['street']}<br>
        <b>Status:</b> {status}<br>
        <b>AI Insight:</b> {pred_text}
        """
        
        folium.Marker(
            [row['latitude'], row['longitude']],
            popup=popup_html,
            icon=folium.Icon(color=color, icon="car", prefix="fa")
        ).add_to(m)

    # Display map
    st_map = st_folium(m, width=None, height=400)

    # --- Reservation Section ---
    st.subheader("📱 Reserve a Space")
    
    # Filter only free spots
    free_df = df[~df['is_occupied']]
    
    if not free_df.empty:
        selected_spot_id = st.selectbox("Select an Available Spot", free_df['spot_id'])
        duration = st.slider("How long will you park? (Hours)", 0.5, 12.0, 1.0)
        
        selected_spot = df[df['spot_id'] == selected_spot_id].iloc[0]
        
        # Calculate Dynamic Price
        price, multiplier = dynamic_pricing_engine(selected_spot['base_fee'], occupancy, duration)
        
        st.info(f"💰 **Estimated Fee:** KES {price}")
        if multiplier > 1.0:
            st.warning(f"⚠️ High Demand Zone! Pricing increased by {int((multiplier-1)*100)}%.")
        elif multiplier < 1.0:
            st.success(f"✅ Low Demand Zone! Discount applied.")
            
        if st.button("Confirm Reservation & Pay via M-Pesa"):
            # Update state
            idx = df.index[df['spot_id'] == selected_spot_id][0]
            st.session_state.parking_data.at[idx, 'is_occupied'] = True
            
            st.session_state.reservations.append({
                "spot": selected_spot_id,
                "time": datetime.now().strftime("%H:%M:%S"),
                "fee": price,
                "duration": duration
            })
            st.success(f"Reservation confirmed for {selected_spot_id}. SMS sent to your phone.")
            st.rerun()
    else:
        st.error("No spots available right now. Our AI predicts spots opening in Kimathi St in 15 mins.")

# --- 5. UI: Admin/Authority Interface (The "DataSense" Part) ---

def admin_view():
    st.header("👮 Authority Dashboard: Management & Analytics")
    
    df = st.session_state.parking_data
    
    # KPI Row
    kpi1, kpi2, kpi3 = st.columns(3)
    revenue_est = sum([r['fee'] for r in st.session_state.reservations])
    kpi1.metric("Total Revenue (Today)", f"KES {revenue_est}")
    kpi2.metric("Violations Detected", random.randint(0, 5))
    kpi3.metric("Sensor Health", "98% Online")

    st.markdown("---")
    
    # Natural Language Querying
    st.subheader("🧠 AI Analytics Assistant")
    st.markdown("Ask questions about occupancy patterns, revenue, or violations.")
    
    if llm:
        sdf = SmartDataframe(df, config={"llm": llm})
        query = st.text_area("Ask a question:", placeholder="e.g., 'Which street has the highest occupancy?' or 'Plot a bar chart of base fees by street'")
        
        if st.button("Generate Insight"):
            with st.spinner("AI is analyzing parking data..."):
                response = sdf.chat(query)
                st.write(response)
    else:
        st.warning("Configure Groq API Key in secrets.toml to enable AI Analytics.")

    # Raw Data View
    with st.expander("View Raw Parking Sensor Data"):
        st.dataframe(df)

# --- 6. Main Navigation ---

def main():
    st.sidebar.title("Nairobi Smart Parking")
    role = st.sidebar.radio("Select User Mode", ["Driver", "City Authority (Admin)"])
    
    if role == "Driver":
        driver_view()
    else:
        admin_view()
    
    st.sidebar.markdown("---")
    st.sidebar.info("Final Year Project | Data Science & Analytics | JKUAT")

if __name__ == "__main__":
    main()
