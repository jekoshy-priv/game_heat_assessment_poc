import streamlit as st
import pandas as pd
import numpy as np
import datetime
from zoneinfo import ZoneInfo 
#from databricks import sql

st.set_page_config(
    page_title="NRL Heat Assessment",
    layout="wide"    # optional
)
st.header("NRL | Game Heat Assessment")

PLAYER_DATA = pd.DataFrame([
    ["Hit-Up Forward", 122.0, 1.94, 24.8, 1.5],
    ["Wide-Running Forwards", 115.0, 1.90, 25.3, 1.5],
    ["Adjustables", 85.0, 1.80, 25.8, 1.5],
    ["Outside Backs", 100.0, 1.90, 23.5, 1.4],
], columns=[
    "Player", "Weight", "Height",
    "Rate_of_Oxygen_Uptake", "vself"
])

# Club list
clubs = [
    "","Broncos","Raiders","Bulldogs","Sharks","Dolphins","Titans",
    "Sea Eagles","Storm","Warriors","Knights","Cowboys","Eels",
    "Panthers","Rabbitohs","Dragons","Roosters","Wests Tigers"
]

def float_input(label, default=""):
    value = st.text_input(label, value=default)
    try:
        return float(value) if value != "" else None
    except ValueError:
        st.error(f"{label} must be a number")
        return None

# Use a form so everything submits together
with st.form("heat_assessment_form"):
    col1, col2 = st.columns(2)

    with col1:
        club_name = st.selectbox("Club Name", clubs)
        record_type = st.selectbox("Record Type", ["","Training", "Game Day"])
        venue = st.text_input("Venue")
        gender = st.selectbox("Gender", ["","Male", "Female"])

    with col2:
        air_temp = float_input("Air Temperature (°C)")
        globe_temp = float_input("Globe Temperature (°C)")
        humidity = float_input("Humidity (%)")
        air_speed = float_input("Air Speed (m/s)")

    calculate = st.form_submit_button("Calculate")

def calculate_heat_metrics(
    air_temp, globe_temp, humidity, air_speed,
    gender, record_type, club, venue
):
    df = PLAYER_DATA.copy()

    # -----------------------------
    # SUBJECT
    # -----------------------------
    df["Mean_Radiant_Temperature"] = (
        ((globe_temp + 273) ** 4 +
         (2.5e8 * air_speed ** 0.6 * (globe_temp - air_temp))) ** 0.25
    ) - 273

    df["Barometric_Pressure"] = 101.9
    df["Air_Velocity"] = air_speed + 1.5
    df["VO2_per_l"] = df["Rate_of_Oxygen_Uptake"] * df["Weight"] / 1000
    df["RER"] = 0.95
    df["Tcl"] = 36
    df["Tsk"] = 36
    df["Emissivity"] = 0.95
    df["Ar"] = 0.35
    df["Icl"] = 0.4
    df["Recl"] = 0.012

    df["Body_Surface_Area"] = (
        0.202 * (df["Weight"] ** 0.425) * (df["Height"] ** 0.725)
    )

    # -----------------------------
    # ENVIRONMENT
    # -----------------------------
    df["Ambient_Temp_K"] = air_temp + 273.15
    df["Ambient_Vapour_Pressure"] = (
        np.exp(18.956 - (4030.18 / (air_temp + 235))) / 10
    ) * humidity / 100

    # -----------------------------
    # METABOLIC
    # -----------------------------
    df["Metabolic_W"] = np.where(
        df["RER"] < 1,
        df["VO2_per_l"] * ((0.23 * df["RER"]) + 0.77) * 5.88 * 60,
        df["VO2_per_l"] * 5.88 * 60
    )

    df["Metabolic_W_m2"] = df["Metabolic_W"] / df["Body_Surface_Area"]

    # -----------------------------
    # DRY HEAT EXCHANGE
    # -----------------------------
    df["fcl"] = 1 + (0.31 * df["Icl"])

    df["hc"] = np.where(
        df["Air_Velocity"] < 0.2,
        3.16006,
        0.7 * 8.3 * (df["Air_Velocity"] ** 0.6)
    )

    df["hr"] = (
        4 * df["Emissivity"] * 5.67e-8 * df["Ar"] *
        (273.2 + ((df["Tcl"] + df["Mean_Radiant_Temperature"]) / 2)) ** 3
    )

    df["h"] = df["hc"] + df["hr"]

    df["To"] = (
        (df["hr"] * df["Mean_Radiant_Temperature"] +
         df["hc"] * air_temp) / df["h"]
    )

    df["Rcl"] = 0.155 * df["Icl"]

    df["Dry_Heat_W_m2"] = (
        (df["Tsk"] - df["To"]) /
        (df["Rcl"] + (1 / (df["fcl"] * df["h"])))
    )

    # -----------------------------
    # RESPIRATORY
    # -----------------------------
    df["Resp_Heat_m2"] = (
        (0.0014 * df["Metabolic_W_m2"] * (34 - air_temp)) +
        (0.0173 * df["Metabolic_W_m2"] * (5.86618428 - df["Ambient_Vapour_Pressure"]))
    )

    # -----------------------------
    # EVAPORATIVE
    # -----------------------------
    df["Ereq_m2"] = (
        df["Metabolic_W_m2"] -
        df["Dry_Heat_W_m2"] -
        df["Resp_Heat_m2"]
    )

    df["Esk_max_m2"] = (
        (np.exp(18.956 - (4030.18 / (df["Tsk"] + 235))) / 10 -
         df["Ambient_Vapour_Pressure"]) /
        (df["Recl"] + (1 / (df["fcl"] * (16.5 * df["hc"]))))
    )


    # --- SWEAT RATE (cte_sweat_rate + cte_hsi) ---

    # Convert m2 terms to kg terms (matches SQL)
    df["Ereq_kg"] = (df["Ereq_m2"] * df["Body_Surface_Area"]) / df["Weight"]
    df["Esk_max_kg"] = (df["Esk_max_m2"] * df["Body_Surface_Area"]) / df["Weight"]

    # Skin wettedness ratio
    df["Skin_Wettedness"] = df["Ereq_kg"] / df["Esk_max_kg"]

    # HSI (exact SQL)
    df["HSI"] = df["Skin_Wettedness"] * 100

    # Sweating efficiency (exact SQL IF)
    df["Sweating_Efficiency"] = np.where(
        df["Skin_Wettedness"] < 1,
        1 - (df["Skin_Wettedness"] ** 2) / 2,
        0.6
    )

    # Heat loss equivalent (W/m²)
    df["Heat_Loss_Equivalent"] = df["Ereq_m2"] / df["Sweating_Efficiency"]

    # Sweat rate (kg/hr → L/hr)
    df["Sweat_Rate_g_hr"] = (
        df["Heat_Loss_Equivalent"]
        * df["Body_Surface_Area"]
        * 3600
        / 2427
    )

    df["Sweat_Rate"] = df["Sweat_Rate_g_hr"] / 1000

    df["HSI"] = df["HSI"].round(0)

    # -----------------------------
    # ASSESSMENT
    # -----------------------------
    if gender.lower() == "male":
        df["Assessment"] = np.select(
            [df["HSI"] > 250, df["HSI"] > 200, df["HSI"] > 150],
            ["Delay/Suspend Play",
             "Caution: Extended breaks recommended",
             "Cooling breaks recommended"],
            default="No cooling breaks required"
        )
    else:
        df["Assessment"] = np.select(
            [df["HSI"] > 225, df["HSI"] > 180, df["HSI"] > 135],
            ["Delay/Suspend Play",
             "Caution: Extended breaks recommended",
             "Cooling breaks recommended"],
            default="No cooling breaks required"
        )

    df["records_type"] = record_type
    df["club"] = club
    df["venue"] = venue
    df["gender"] = gender
    df["created_at"] = datetime.datetime.now(ZoneInfo("Australia/Sydney")).strftime("%Y-%m-%d %H:%M:%S")

    # Final output
    return df[[
        "records_type",
        "club",
        "venue",
        "gender",
        "Player",
        "HSI",
        "Assessment",
        "Sweat_Rate",
        "created_at"
    ]].round({
        "HSI": 0,
        "Sweat_Rate": 2
    })

# Action after button press
if calculate:
    results = calculate_heat_metrics(
        air_temp=air_temp,
        globe_temp=globe_temp,
        humidity=humidity,
        air_speed=air_speed,
        gender=gender,
        record_type=record_type,
        club=club_name,
        venue=venue
    )
    st.dataframe(results, use_container_width=True)

    #try:
        #insert_to_databricks_with_id(results)
        #st.success("Inserted into <table>")
    #except Exception as e:
     #   st.error(f"Insert failed: {e}")