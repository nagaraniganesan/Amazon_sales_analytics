from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

# ===== load models / artifacts =====
with open("order_issue_model.pkl", "rb") as f:
    order_issue_model = pickle.load(f)

with open("recommender.pkl", "rb") as f:
    rec_art = pickle.load(f)

products = rec_art["products"]
product_to_index = rec_art["product_to_index"]
sim_matrix = rec_art["sim_matrix"]

app = Flask(__name__)

# ===== helper: similar items =====
def similar_items_api(item_name: str, top_n: int = 5):
    if item_name not in product_to_index:
        return []
    idx = product_to_index[item_name]
    sims = sim_matrix[idx]
    similar_idx = sims.argsort()[::-1][1 : top_n + 1]
    recs = products.loc[similar_idx, ["ProductName", "Category", "Brand"]]
    return recs.to_dict(orient="records")

# ===== routes =====

@app.route("/", methods=["GET"])
def landing():
    return "WELCOME TO AMAZON ANALYTICS API"

@app.route("/predict_order_issue", methods=["POST"])
def predict_order_issue():
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    try:
        df_in = pd.DataFrame([data])
        pred = order_issue_model.predict(df_in)[0]
        proba = order_issue_model.predict_proba(df_in)[0][1]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    return jsonify({
        "order_issue": int(pred),
        "probability_issue": float(proba)
    })

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    # GET: simple demo for browser
    if request.method == "GET":
        recs = similar_items_api("Smartphone Case", top_n=5)
        return jsonify({"query": "Smartphone Case", "recommendations": recs})

    # POST: use JSON from client
    data = request.get_json()
    if not isinstance(data, dict):
        return jsonify({"error": "Invalid JSON"}), 400

    item = data.get("product_name")
    top_n = data.get("top_n", 5)

    if not item:
        return jsonify({"error": "product_name is required"}), 400

    try:
        top_n = int(top_n)
    except:
        top_n = 5

    recs = similar_items_api(item, top_n=top_n)
    return jsonify({
        "query": item,
        "recommendations": recs
    })

if __name__ == "__main__":
    app.run(debug=True)
