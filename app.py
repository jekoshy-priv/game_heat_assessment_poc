import streamlit as st
import pandas as pd
import numpy as np
import datetime
from zoneinfo import ZoneInfo 
import os
import msal 
import requests

TENANT_ID = st.secrets["TENANT_ID"]
CLIENT_ID = st.secrets["CLIENT_ID"]
CLIENT_SECRET = st.secrets["CLIENT_SECRET"]

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]

#CSV_PATH = "heat_assessment_log.csv"

SITE_ID = st.secrets["SITE_ID"]
LIST_ID = st.secrets["LIST_ID"]

def insert_to_sharepoint(log_df):
    token = get_graph_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    for _, row in log_df.iterrows():
        payload = {
                    "fields": {
                        "Title": row["player"],
                        "Club": row["club"],
                        "Record_x0020_Type": row["records_type"],
                        "Venue": row["venue"],
                        "Gender": row["gender"],
                        "AirTemperature_x0028_C_x0029_": float(row["air_temp"]),
                        "GlobeTemperature_x0028_C_x0029_": float(row["globe_temp"]),
                        "Humidity_x0028__x0025__x0029_": float(row["humidity"]),
                        "AirSpeed_x0028_m_x002f_s_x0029_": float(row["air_speed"]),
                        "Player": row["player"],
                        "Assessment": row["assessment"],
                        "HSI": int(row["HSI"]),
                        "SweatRate": float(row["sweat_rate"]),
                        "CreatedAt": row["created_at"]
                    }
                }

        url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/lists/{LIST_ID}/items"
        r = requests.post(url, headers=headers, json=payload)

        if r.status_code not in (200, 201):
            raise Exception(r.text)

st.set_page_config(
    page_title="NRL Game Heat Assessment",
    layout="wide"    
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

clubs = [
    "","Broncos","Raiders","Bulldogs","Sharks","Dolphins","Titans",
    "Sea Eagles","Storm","Warriors","Knights","Cowboys","Eels",
    "Panthers","Rabbitohs","Dragons","Roosters","Wests Tigers"
]

def get_graph_token():
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET
    )
    result = app.acquire_token_for_client(scopes=SCOPE)
    if "access_token" not in result:
        raise Exception("Could not acquire Graph token")
    return result["access_token"]

def float_input(label, default=""):
    value = st.text_input(label, value=default)
    try:
        return float(value) if value != "" else None
    except ValueError:
        st.error(f"{label} must be a number")
        return None
    
def all_fields_filled(fields: dict):
    """Return True if all values in the dictionary are not None or empty."""
    for key, value in fields.items():
        if value is None or value == "":
            return False
    return True

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

    calculate = st.form_submit_button("Submit")

def calculate_heat_metrics(
    air_temp, globe_temp, humidity, air_speed,
    gender, record_type, club, venue
):
    df = PLAYER_DATA.copy()

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

    df["HSI"] = df["HSI"].round(0).astype(int)
    df["Sweat_Rate"] = df["Sweat_Rate"].round(2)

    if gender.lower() == "male":
        df["Assessment"] = np.select(
            [df["HSI"] > 250, df["HSI"] > 200, df["HSI"] > 150],
            ["Delay/Suspend Play",
             "Caution: Implement Full Heat Policy Strategies",
             "Cooling breaks recommended"],
            default="No cooling breaks required"
        )
    else:
        df["Assessment"] = np.select(
            [df["HSI"] > 225, df["HSI"] > 180, df["HSI"] > 135],
            ["Delay/Suspend Play",
             "Caution: Implement Full Heat Policy Strategies",
             "Cooling breaks recommended"],
            default="No cooling breaks required"
        )

    df["records_type"] = record_type
    df["club"] = club
    df["venue"] = venue
    df["gender"] = gender
    df["created_at"] = datetime.datetime.now(ZoneInfo("Australia/Sydney")).isoformat()

    full_df = df.copy()
    

    return full_df, df[[
        "Player",
        "Assessment",
        "HSI",
        "Sweat_Rate"
    ]].round({
        "HSI": 0,
        "Sweat_Rate": 2
    })

def assessment_color(val):

    if val == "Delay/Suspend Play":
        # Grey
        return "background-color: #9e9e9e; color: white;"

    elif val == "Caution: Implement Full Heat Policy Strategies":
        # Red
        return "background-color: #d32f2f; color: white;"

    elif val == "Cooling breaks recommended":
        # Yellow
        return "background-color: #fbc02d; color: black;"

    elif val == "No cooling breaks required":
        # Green
        return "background-color: #388e3c; color: white;"

    else:
        # Fallback (in case of unexpected text)
        return ""

# Action after button press
if calculate:

    form_values = {
        "air_temp": air_temp,
        "globe_temp": globe_temp,
        "humidity": humidity,
        "air_speed": air_speed,
        "gender": gender,
        "record_type": record_type,
        "club": club_name,
        "venue": venue
    }

    if not all_fields_filled(form_values):
        st.warning("⚠️ Please fill in all fields before submitting.")
    else:
        # -----------------------------
        # CALCULATION FIRST
        # -----------------------------
        full_df, results = calculate_heat_metrics(
            air_temp=air_temp,
            globe_temp=globe_temp,
            humidity=humidity,
            air_speed=air_speed,
            gender=gender,
            record_type=record_type,
            club=club_name,
            venue=venue
        )

        # Display results with color
        styled_results = results.copy()
        styled_results["Assessment"] = styled_results["Assessment"].apply(
            lambda x: f'<div style="{assessment_color(x)} padding:4px; text-align:center">{x}</div>'
        )
        html_table = styled_results.to_html(index=False, escape=False)
        st.markdown(f'<div style="overflow-x:auto; width:100%">{html_table}</div>',
                    unsafe_allow_html=True)

        # -----------------------------
        # THEN TRY LOGGING TO SHAREPOINT
        # -----------------------------
        log_df = pd.DataFrame({
            "club": full_df["club"],
            "records_type": full_df["records_type"],
            "venue": full_df["venue"],
            "gender": full_df["gender"],
            "air_temp": air_temp,
            "air_speed": air_speed,
            "globe_temp": globe_temp,
            "humidity": humidity,
            "player": full_df["Player"],
            "assessment": full_df["Assessment"],
            "HSI": full_df["HSI"],
            "sweat_rate": full_df["Sweat_Rate"],
            "created_at": full_df["created_at"],
        })

        try:
            insert_to_sharepoint(log_df)
            st.success("✅ Data successfully sent to SharePoint")
        except Exception as e:
            st.warning(f"⚠️ Could not send data to SharePoint: {e}")

    #st.success(f"CSV written to: {os.path.abspath(CSV_PATH)}")

    #try:
        #insert_to_databricks_with_id(results)
        #st.success("Inserted into <table>")
    #except Exception as e:
     #   st.error(f"Insert failed: {e}")