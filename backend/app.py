from flask import Flask, render_template, request
import joblib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from flask import send_file
import matplotlib
matplotlib.use("Agg")


app = Flask(__name__, template_folder="../frontend")

# ---------- Load Model ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "model", "xgboost_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
feature_order = joblib.load(os.path.join(BASE_DIR, "model", "feature_order.pkl"))

CLASS_MAP = {
    0: "Non-Habitable",
    1: "Potentially Habitable",
    2: "Highly Habitable"
}

# ---------- Store Predictions ----------
planet_rankings = []


# ---------- Habitability Score ----------
def compute_final_habitability_score(d):
    hsi = (
        0.25 * d["pl_rade"] +
        0.20 * d["pl_bmasse"] +
        0.20 * d["pl_dens"] +
        0.20 * d["pl_eqt"] +
        0.15 * d["pl_orbper"]
    )

    sci = (
        0.45 * d["st_spectral_score"] +
        0.25 * (1 - abs(d["st_teff"] - 0.5)) +
        0.20 * d["st_lum"] +
        0.10 * d["st_met"]
    )

    final = 0.8 * hsi + 0.2 * sci
    return round(min(max(final, 0), 1), 3)

def generate_feature_importance():
    importances = model.feature_importances_
    features = feature_order

    df_imp = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    plt.figure()
    sns.barplot(data=df_imp, x="Importance", y="Feature")
    plt.title("Feature Importance")
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "backend", "static", "feature_importance.png")
    plt.savefig(path)
    plt.close()

    return "feature_importance.png"

def generate_distribution():
    preds = model.predict(scaler.transform(
        np.random.rand(200, len(feature_order))
    ))

    df = pd.DataFrame(preds, columns=["Class"])

    plt.figure()
    df["Class"].value_counts().plot(kind="bar")
    plt.title("Habitability Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()

    path = os.path.join(BASE_DIR, "backend", "static", "distribution.png")
    plt.savefig(path)
    plt.close()

    return "distribution.png"


# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    score = None
    top_two = []

    if request.method == "POST":
        try:
            data = {f: float(request.form[f]) for f in feature_order}

            X = np.array([data[f] for f in feature_order]).reshape(1, -1)
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0].max()

            result = CLASS_MAP[int(pred)]
            score = compute_final_habitability_score(data)

            # -------- Store Planet Automatically --------
            planet_rankings.append({
                "name": f"Planet {len(planet_rankings) + 1}",
                "score": score,
                "class": result
            })

            # Sort highest first
            planet_rankings.sort(key=lambda x: x["score"], reverse=True)

            # Get only Top 2
            top_two = planet_rankings[:2]

        except Exception as e:
            result = f"Error: {e}"

    return render_template("index.html", result=result, score=score, rankings=top_two)

@app.route("/dashboard")
def dashboard():
    imp_img = generate_feature_importance()
    dist_img = generate_distribution()

    return render_template(
        "dashboard.html",
        imp_img=imp_img,
        dist_img=dist_img
    )

@app.route("/export_pdf")
def export_pdf():
    static_path = os.path.join(BASE_DIR, "backend", "static")
    path = os.path.join(static_path, "habitability_report.pdf")

    c = canvas.Canvas(path)

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 800, "ExoHabitAI Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(100, 770, "Generated Successfully")

    # Image paths
    feature_img = os.path.join(static_path, "feature_importance.png")
    dist_img = os.path.join(static_path, "distribution.png")

    # Insert images if they exist
    if os.path.exists(feature_img):
        c.drawImage(feature_img, 100, 450, width=400, height=250)

    if os.path.exists(dist_img):
        c.drawImage(dist_img, 100, 150, width=400, height=250)

    c.save()

    return send_file(path, as_attachment=True)



# ---------- Run ----------
if __name__ == "__main__":
    app.run()
