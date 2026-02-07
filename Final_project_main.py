import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,silhouette_score,mean_squared_error
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
import re
import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# 1.Load_dataset
df = pd.read_csv("Amazon.csv")
print("Raw Shape:", df.shape)
df.columns = df.columns.str.strip()

print(df.head())
print(df.dtypes)

# 2.Basic_cleaning_of_the_dataset
df = df.drop_duplicates()

# 3.Type_conversions
df["OrderDate"] = pd.to_datetime(df["OrderDate"], errors="coerce")

for col in ["Quantity", "UnitPrice", "Discount", "Tax", "ShippingCost", "TotalAmount"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 4.Drop_rows_with_missing_key_fields
df = df.dropna(subset=["OrderDate", "TotalAmount"])
print("After dropping NaNs in date/TotalAmount:", df.shape)

# 5.Feature_engineering
df["OrderYear"] = df["OrderDate"].dt.year
df["OrderMonth"] = df["OrderDate"].dt.month

# 6.Target: OrderIssue_if1 = Returned/Cancelled,0 = normal)
df["OrderIssue"] = df["OrderStatus"].isin(["Returned", "Cancelled"]).astype(int)
print("OrderIssue_counts:")
print(df["OrderIssue"].value_counts())

# 7.Select_features_and_target
target_col = "OrderIssue"

#Features:
feature_cols = ["Quantity","UnitPrice","Discount","Tax","ShippingCost","TotalAmount","OrderYear","OrderMonth",
    "Category","Brand","PaymentMethod","City","State","Country",]

# Keep_only_columns_that_actually_exist
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols]
y = df[target_col]

print("Feature_columns_used:", feature_cols)
print("X shape:", X.shape, "y shape:", y.shape)

# 8.Identify_numeric_vs_categorical
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = [c for c in X.columns if c not in numeric_features]

print("Numeric_features:", numeric_features)
print("Categorical_features:", categorical_features)

# 9.Preprocessing_pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

# 10.Train/test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# 11.XG_Boosting
xgb_clf = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", xgb_clf)
])

# 12.Fit_model
model_pipeline.fit(X_train, y_train)
print("XGBoosting_model_trained_on_OrderIssue!")

# 13.Evaluate
y_pred = model_pipeline.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)

print("\n=== Classification metrics===")
print("Accuracy:", f"{acc:.3f}")
print("Precision:", f"{prec:.3f}")
print("Recall:", f"{rec:.3f}")
print("F1:", f"{f1:.3f}")

print("\n=== Full classification report ===")
print(classification_report(y_test, y_pred, zero_division=0))

# ============================================
# Customer_level_clustering_with_KMeans

# 1.Build_customer_level_features
cust = df.copy()

# Basic_aggregations_per_customer
cust_group = cust.groupby("CustomerID").agg({
    "OrderID": "nunique",
    "TotalAmount": "sum",
    "Quantity": "sum",
    "Discount": "mean",
    "OrderIssue": "mean",
    "Category": "nunique",
})

cust_group.rename(columns={
    "OrderID": "NumOrders",
    "TotalAmount": "TotalSpend",
    "Quantity": "TotalQuantity",
    "Discount": "AvgDiscount",
    "OrderIssue": "IssueRate",
    "Category": "NumCategories"
}, inplace=True)

print("Customer_feature_table_listed:", cust_group.shape)
print(cust_group.head())

# 2.Fill_any_remaining_NaNs
cust_group = cust_group.fillna(0)

# 3.Scale_numeric_features_for_clustering
scaler = StandardScaler()
cust_scaled = scaler.fit_transform(cust_group)

# 4.KMeans_clustering
range_n_clusters = range(2, 5)
best_k = None
best_sil = -1
best_labels = None

print("\nSilhouette_scores_for_different_k:")
for k in range_n_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cust_scaled)
    sil = silhouette_score(cust_scaled, labels)
    print(f"k = {k}, silhouette = {sil:.4f}")

    if sil > best_sil:
        best_sil = sil
        best_k = k
        best_labels = labels

print(f"\nBest_k_by_silhouette: {best_k}, best_silhouette: {best_sil:.4f}")

# Best_k_and_labels
cust_group["Cluster"] = best_labels

print("Cluster_counts:\n", cust_group["Cluster"].value_counts().sort_index())

# 6.Cluster_summaries
cluster_profiles = cust_group.groupby("Cluster").agg({
    "NumOrders": ["mean", "min", "max"],
    "TotalSpend": ["mean", "min", "max"],
    "TotalQuantity": ["mean"],
    "AvgDiscount": ["mean"],
    "IssueRate": ["mean"],
    "NumCategories": ["mean"]
})

print("\n=== Customer_cluster_profiles ===")
print(cluster_profiles)

# ============================================
# Monthly_Sales_Forecasting_with_ARIMA

print("============================================")
print("TIME_SERIES_ANALYSIS")
# 1.Build_monthly_total_sales_series_from_cleaned_df
ts_monthly = (
    df.set_index("OrderDate")
      .resample("M")["TotalAmount"].sum()
      .sort_index()
)

print("\nMonthly_series_length:", len(ts_monthly))
print(ts_monthly.head())

# 2.Train/test_split
n_test = 6
train = ts_monthly.iloc[:-n_test]
test = ts_monthly.iloc[-n_test:]

print("Train_period:", train.index.min(), "to", train.index.max())
print("Test_period:", test.index.min(), "to", test.index.max())

# 3.Fit_ARIMA_model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# 4.Forecast_for_the_test_horizon
forecast = model_fit.forecast(steps=n_test)

# 5.Evaluate
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print("Test_RMSE (monthly TotalAmount):", rmse)

# 6.Plot_train, test, forecast
plt.figure(figsize=(10, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="orange")
plt.plot(forecast.index, forecast, label="Forecast", color="green")
plt.title("Monthly_Total_Sales - ARIMA Forecast")
plt.xlabel("Month")
plt.ylabel("Total_Sales")
plt.legend()
plt.tight_layout()
plt.show()

# 7.Forecast_next_6_future_months_beyond_last_data_point
future_steps = 6
future_forecast = model_fit.forecast(steps=len(test) + future_steps)

# Last_6_points_of_that_forecast = future_beyond_original_data
future_only = future_forecast.iloc[-future_steps:]

print("\nNext_6months_forecast:")
for date, value in future_only.items():
    print(date.strftime("%Y-%m"), "->", round(value, 2))


print("============== NLP-enhanced_item_similarity ==============")

products = (
    df[["ProductID", "ProductName", "Category", "Brand"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

products["text"] = (
    products["ProductName"].fillna("") + " " +
    products["Category"].fillna("") + " " +
    products["Brand"].fillna("")
)
products["text"] = products["text"].apply(clean_text)

tfidf = TfidfVectorizer(ngram_range=(1, 2))  # unigrams + bigrams
tfidf_matrix = tfidf.fit_transform(products["text"])
sim_matrix = cosine_similarity(tfidf_matrix)

product_to_index = {name: i for i, name in enumerate(products["ProductName"])}

def similar_items(item_name, top_n=5):
    if item_name not in product_to_index:
        print(f"'{item_name}' not found in ProductName.")
        return []
    idx = product_to_index[item_name]
    sims = sim_matrix[idx]
    similar_idx = sims.argsort()[::-1][1 : top_n + 1]
    return products.loc[similar_idx, ["ProductName", "Category", "Brand"]]

print("Similar products to 'Smartphone Case':")
print(similar_items("Smartphone Case", top_n=5))

#pickle_file_writing
import os

print("Current working directory:", os.getcwd())

with open("order_issue_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)
print("Saved order_issue_model.pkl")

with open("customer_clusters.pkl", "wb") as f:
    pickle.dump(cust_group, f)
print("Saved customer_clusters.pkl")

with open("arima_sales_model.pkl", "wb") as f:
    pickle.dump(model_fit, f)
print("Saved arima_sales_model.pkl")

recommender_artifacts = {
    "products": products,
    "product_to_index": product_to_index,
    "sim_matrix": sim_matrix
}
with open("recommender.pkl", "wb") as f:
    pickle.dump(recommender_artifacts, f)
print("Saved recommender.pkl")


# === Precompute_EDA_summaries ===
total_revenue = df["TotalAmount"].sum()
date_min = df["OrderDate"].min()
date_max = df["OrderDate"].max()

top_cat = (
    df.groupby("Category")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(5)
)

top_prod = (
    df.groupby("ProductName")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(5)
)

def get_recommendations():
    item_name = entry_item.get().strip()
    text_output.delete("1.0", tk.END)

    if not item_name:
        text_output.insert(tk.END, "Please enter a product name.\n")
        return

    recs = similar_items(item_name, top_n=5)

    if isinstance(recs, list) and not recs:
        text_output.insert(tk.END, f"No recommendations for '{item_name}'.")
        return

    text_output.insert(tk.END, f"Similar products to '{item_name}':\n\n")
    for _, row in recs.iterrows():
        line = f"- {row['ProductName']} | {row['Category']} | {row['Brand']}\n"
        text_output.insert(tk.END, line)

# ----- plotting_helpers -----
plot_frame = None
canvas = None

def show_figure_in_gui(fig):
    global canvas
    for widget in plot_frame.winfo_children():
        widget.destroy()
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

def show_eda_plots():
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Pie: payment methods
    pay_counts = (df["PaymentMethod"]
                  .value_counts(normalize=True)
                  .sort_values(ascending=False) * 100)

    axes[0].pie(
        pay_counts.values,
        labels=pay_counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    axes[0].set_title("Distribution of Payment Methods")

    # Barh: sales by category
    cat_sales = (df.groupby("Category")["TotalAmount"]
                   .sum()
                   .sort_values(ascending=False))

    axes[1].barh(cat_sales.index, cat_sales.values)
    axes[1].set_title("Total Sales by Product Category")
    axes[1].set_xlabel("Total Revenue")

    fig.tight_layout()
    show_figure_in_gui(fig)

def show_time_series_plot():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train.index, train, label="Train")
    ax.plot(test.index, test, label="Test", color="orange")
    ax.plot(forecast.index, forecast, label="Forecast", color="green")
    ax.set_title("Monthly Total Sales - ARIMA Forecast")
    ax.set_xlabel("Month")
    ax.set_ylabel("Total Sales")
    ax.legend()
    fig.tight_layout()
    show_figure_in_gui(fig)

# ----- GUI_layout -----
root = tk.Tk()
root.title("Amazon Retail Analytics")
root.configure(bg="#f5f5f5")
root.geometry("1050x700")

title = tk.Label(
    root,
    text="Amazon Retail Analytics Dashboard",
    fg="white",
    bg="#222831",
    font=("Segoe UI", 18, "bold"),
    pady=10
)
title.pack(fill=tk.X)

main_frame = tk.Frame(root, bg="#f5f5f5")
main_frame.pack(fill=tk.BOTH, expand=True)

btn_style = dict(width=22, height=2, font=("Segoe UI", 10), relief=tk.FLAT)

left_frame = tk.Frame(main_frame, bg="#f5f5f5")
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=15, pady=15)

tk.Button(left_frame, text="📊 EDA Charts", bg="#00adb5", fg="white",
          command=show_eda_plots, **btn_style).pack(pady=6)

tk.Button(left_frame, text="📈 Time Series Forecast", bg="#f8b400", fg="black",
          command=show_time_series_plot, **btn_style).pack(pady=6)

tk.Label(left_frame, text="Product for Similar Items:",
         bg="#f5f5f5", font=("Segoe UI", 10, "bold")).pack(pady=(20, 3))

entry_item = tk.Entry(left_frame, width=25, font=("Segoe UI", 10))
entry_item.pack(pady=3)
entry_item.insert(0, "Smartphone Case")

tk.Button(left_frame, text="💡 Get Recommendations", bg="#6a00f4", fg="white",
          command=get_recommendations, **btn_style).pack(pady=6)

right_frame = tk.Frame(main_frame, bg="#f5f5f5")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=15, pady=15)

plot_frame = tk.Frame(right_frame, bg="white", bd=1, relief=tk.SOLID, height=380)
plot_frame.pack(fill=tk.BOTH, expand=True)

text_output = tk.Text(right_frame, width=90, height=8,
                      font=("Consolas", 10), bd=1, relief=tk.SOLID)
text_output.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

root.mainloop()

