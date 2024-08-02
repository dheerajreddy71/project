from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

# Load and prepare the datasets
yield_df = pd.read_csv("D:/DESIGN PROJECT/yield_df.csv")
crop_recommendation_data = pd.read_csv("D:/DESIGN PROJECT/Crop_recommendation.csv")

# Define the preprocessors and models
yield_preprocessor = ColumnTransformer(
    transformers=[
        ('StandardScale', StandardScaler(), [0, 1, 2, 3]),
        ('OHE', OneHotEncoder(drop='first'), [4, 5]),
    ],
    remainder='passthrough'
)
yield_X = yield_df[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item']]
yield_y = yield_df['hg/ha_yield']
yield_X_train, yield_X_test, yield_y_train, yield_y_test = train_test_split(yield_X, yield_y, train_size=0.8, random_state=0, shuffle=True)
yield_X_train_dummy = yield_preprocessor.fit_transform(yield_X_train)
yield_X_test_dummy = yield_preprocessor.transform(yield_X_test)
yield_model = KNeighborsRegressor(n_neighbors=5)
yield_model.fit(yield_X_train_dummy, yield_y_train)

crop_X = crop_recommendation_data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
crop_y = crop_recommendation_data['label']
crop_X_train, crop_X_test, crop_y_train, crop_y_test = train_test_split(crop_X, crop_y, test_size=0.2, random_state=42)
crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
crop_model.fit(crop_X_train, crop_y_train)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'predict_yield' in request.form:
            features = {
                'Year': int(request.form['year']),
                'average_rain_fall_mm_per_year': float(request.form['rainfall']),
                'pesticides_tonnes': float(request.form['pesticides']),
                'avg_temp': float(request.form['temp']),
                'Area': request.form['area'],
                'Item': request.form['item'],
            }
            features_array = np.array([[features['Year'], features['average_rain_fall_mm_per_year'],
                                        features['pesticides_tonnes'], features['avg_temp'],
                                        features['Area'], features['Item']]], dtype=object)
            transformed_features = yield_preprocessor.transform(features_array)
            predicted_yield = yield_model.predict(transformed_features).reshape(1, -1)
            yield_result = f"The predicted yield is {predicted_yield[0][0]:.2f} hectograms (hg) per hectare (ha)."

        elif 'recommend_crop' in request.form:
            crop_features = [
                float(request.form['N']),
                float(request.form['P']),
                float(request.form['K']),
                float(request.form['temperature']),
                float(request.form['humidity']),
                float(request.form['ph']),
                float(request.form['rainfall'])
            ]
            recommended_crop = crop_model.predict([crop_features])[0]
            yield_result = f"Recommended Crop: {recommended_crop}"

        return render_template('index.html', yield_result=yield_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
