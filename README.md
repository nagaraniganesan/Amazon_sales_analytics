# Amazon Sales Analytics

End‑to‑end analytics project on Amazon‑style sales data, covering EDA, machine learning models, a GUI dashboard, and a Flask API.

## Features
- Data cleaning and exploratory data analysis (EDA).
- XGBoost classifier for issue / outcome prediction.
- Customer segmentation with KMeans.
- Sales forecasting using ARIMA.
- NLP‑based similar product recommender.
- Desktop GUI for EDA and recommendations.
- Flask API to serve model predictions.

## Files
- `Final_project_main.py`: main analysis, models, and GUI.
- `Final_project_api.py`: Flask API endpoints.
- `Amazon.csv`: sample dataset.

## Exploratory Data Analysis (EDA)

The project includes EDA on the Amazon sales data:
- Checked missing values, duplicates, and basic summary statistics.
- Analyzed sales by category, payment method, and time period.
- Created visualizations (bar charts, pie charts, line plots) to show revenue trends and top‑selling products.
- Identified key customer and product patterns to guide the ML models.

## How to run

1. Install dependencies:
    ```bash
   pip install -r requirements.txt

2.Run the main script:

```bash
python Final_project_main.py

3.Run the API:
  ```bash
   python Final_project_api.py
   
