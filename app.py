
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server-side plotting

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from flask import Flask, request, send_file, render_template
import pandas as pd
from collections import defaultdict
import joblib
import pickle

# ... rest of your imports and code

app = Flask(__name__)

# Load data and models once at startup
df = pd.read_csv("crop_yield.csv")
best_model = joblib.load("crop_yield_model.joblib")
encoder = joblib.load("onehot_encoder.joblib")
with open("harvest_time_dict.pkl", "rb") as f:
    harvest_time_dict = pickle.load(f)
with open("avg_price_per_crop.pkl", "rb") as f:
    avg_price_per_crop = pickle.load(f)

categorical = ['Crop', 'Season', 'State']

def recommend_crops_for_state(state_name, annual_rainfall_input, model, df, encoder):
    state_name = state_name.title()
    state_df = df[df['State'] == state_name]
    unique_crops = state_df['Crop'].unique()
    raw_recommendations = defaultdict(list)

    for crop in unique_crops:
        crop_data = state_df[state_df['Crop'] == crop]
        avg_features = crop_data[['Crop_Year', 'Area', 'Production', 'Fertilizer', 'Pesticide']].mean()
        weighted_rainfall = annual_rainfall_input * 2
        input_df = pd.DataFrame({
            'Crop': [crop],
            'Season': [crop_data['Season'].mode()[0] if not crop_data['Season'].mode().empty else 'Unknown'],
            'State': [state_name]
        })
        season = input_df['Season'][0]
        encoded_cat = encoder.transform(input_df[categorical])
        encoded_df = pd.DataFrame(encoded_cat, columns=encoder.get_feature_names_out())
        numeric_values = avg_features.values.tolist() + [weighted_rainfall]
        numeric_columns = list(avg_features.index) + ['Annual_Rainfall_Weighted']
        numeric_df = pd.DataFrame([numeric_values], columns=numeric_columns)
        X_pred = pd.concat([encoded_df.reset_index(drop=True), numeric_df.reset_index(drop=True)], axis=1)
        X_pred = X_pred.reindex(columns=model.feature_names_in_, fill_value=0)
        predicted_yield = model.predict(X_pred)[0]
        raw_recommendations[season].append((crop, predicted_yield))

    all_scores = [score for crops in raw_recommendations.values() for _, score in crops]
    min_score = min(all_scores)
    max_score = max(all_scores)

    def scale(score):
        if max_score == min_score:
            return 100
        return 1 + 99 * (score - min_score) / (max_score - min_score)

    scaled_recommendations = defaultdict(list)
    for season, crops in raw_recommendations.items():
        for crop, score in crops:
            scaled_score = round(scale(score), 2)
            scaled_recommendations[season].append((crop, scaled_score, round(score, 2)))
        scaled_recommendations[season].sort(key=lambda x: x[1], reverse=True)

    return scaled_recommendations

def generate_report_pdf(state_name, annual_rainfall_input, recommendations, filename="crop_report.pdf"):
    rows = []
    for season, crops in recommendations.items():
        for crop, score, est_yield in crops:
            harvest_days = harvest_time_dict.get(crop, "N/A")
            avg_price = avg_price_per_crop.get(crop, 0)
            est_profit = round(est_yield * avg_price * 1000, 2) if avg_price else None
            if est_profit is not None and harvest_days != "N/A":
                rows.append({
                    "Season": season,
                    "Crop": crop,
                    "Score": score,
                    "Estimated Yield": est_yield,
                    "Harvest Time (Days)": harvest_days,
                    "Estimated Profit": est_profit
                })

    df_plot = pd.DataFrame(rows)
    with PdfPages(filename) as pdf:
        # Table page
        fig, ax = plt.subplots(figsize=(12, max(2, len(df_plot)*0.4)))
        ax.axis('off')
        tbl = ax.table(cellText=df_plot.values,
                       colLabels=df_plot.columns,
                       cellLoc='center', loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.auto_set_column_width(col=list(range(len(df_plot.columns))))
        plt.title(f"Crop Recommendations for {state_name}", fontsize=14, pad=20)
        pdf.savefig()
        plt.close()

        # Estimated Yield
        plt.figure(figsize=(10,6))
        df_plot_sorted = df_plot.sort_values('Estimated Yield', ascending=False)
        plt.bar(df_plot_sorted['Crop'], df_plot_sorted['Estimated Yield'], color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Predicted Yield for Crops in {state_name}')
        plt.ylabel('Estimated Yield')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Estimated Profit
        plt.figure(figsize=(10,6))
        df_plot_sorted = df_plot.sort_values('Estimated Profit', ascending=False)
        plt.bar(df_plot_sorted['Crop'], df_plot_sorted['Estimated Profit'], color='lightgreen')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Estimated Profit for Crops in {state_name}')
        plt.ylabel('Estimated Profit')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Harvest Time vs Estimated Profit
        plt.figure(figsize=(8,6))
        sc = plt.scatter(df_plot['Harvest Time (Days)'], df_plot['Estimated Profit'],
                         c=df_plot['Score'], cmap='viridis', s=100)
        plt.colorbar(sc, label='Suitability Score')
        plt.title(f'Harvest Time vs Estimated Profit for Crops in {state_name}')
        plt.xlabel('Harvest Time (Days)')
        plt.ylabel('Estimated Profit')
        plt.tight_layout()
        pdf.savefig()
        plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend-crops', methods=['POST'])
def recommend_crops():
    data = request.json
    state_name = data.get('state')
    rainfall = data.get('rainfall')
    if not state_name or rainfall is None:
        return {"error": "State and rainfall must be provided."}, 400
    try:
        rainfall = float(rainfall)
    except ValueError:
        return {"error": "Invalid rainfall value."}, 400

    recommendations = recommend_crops_for_state(state_name, rainfall, best_model, df, encoder)
    pdf_filename = f"{state_name}_crop_report.pdf"
    generate_report_pdf(state_name, rainfall, recommendations, pdf_filename)
    return send_file(pdf_filename, mimetype='application/pdf', as_attachment=True, download_name=pdf_filename)

if __name__ == '__main__':
    app.run(debug=True)
