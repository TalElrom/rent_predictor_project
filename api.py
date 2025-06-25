from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from assets_data_prep import prepare_data
from datetime import datetime

# טען את המודל
with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)
    
app = Flask(__name__)

@app.route("/", methods=["GET"])
def form():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # קבלת ערכי הטופס מהמשתמש
    form_data = request.form

    # יצירת DataFrame משורת טופס אחת
    input_dict = {
        "room_num": float(form_data.get("room_num")),
        "area": float(form_data.get("area")),
        "floor": float(form_data.get("floor") or 0),
        "total_floors": float(form_data.get("total_floors") or 0),
        "property_type": form_data.get("property_type"),
        "neighborhood": form_data.get("neighborhood"),
        "days_to_enter": max(
            (datetime.strptime(form_data.get("entrance_date"), "%Y-%m-%d") - datetime.today()).days
            if form_data.get("entrance_date") else 0, 0 ),                           
        "num_of_payments": float(form_data.get("num_of_payments") or 0),
        "monthly_arnona": float(form_data.get("monthly_arnona") or 0),
        "building_tax": float(form_data.get("building_tax") or 0),
        "garden_area": float(form_data.get("garden_area") or 0),
        "has_balcony": int("has_balcony" in form_data),
        "has_parking": int("has_parking" in form_data),
        "elevator": int("elevator" in form_data),
        "has_safe_room": int("has_safe_room" in form_data),
        "is_renovated": int("is_renovated" in form_data),
        "ac": int("ac" in form_data),
        "is_furnished": int("is_furnished" in form_data),
        "handicap": int("handicap" in form_data),
        "has_storage": int("has_storage" in form_data),
        "has_bars": int("has_bars" in form_data),
    }

    input_df = pd.DataFrame([input_dict])

    # הכנה עם prepare_data
    df_prepared = prepare_data(input_df, dataset_type="test")
    # Ensure correct columns and order
    df_prepared = df_prepared.reindex(columns=model_columns, fill_value=0)

    # חיזוי
    prediction = model.predict(df_prepared)[0]
    prediction_rounded = round(prediction)

    return render_template("index.html", prediction=prediction_rounded)

if __name__ == "__main__":
    app.run(debug=True)
