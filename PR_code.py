# Import necessary libraries
import pandas as pd
import folium
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import xgboost as xgb  # Import XGBoost

# Load dataset
file_path = 'database.csv'  # Update this path if necessary
data = pd.read_csv(file_path)

# Handle missing values for cost-related features
data = data.dropna(subset=['Accident Latitude', 'Accident Longitude', 'All Costs', 
                            'Property Damage Costs', 'Lost Commodity Costs', 
                            'Public/Private Property Damage Costs', 'Emergency Response Costs', 
                            'Environmental Remediation Costs', 'Other Costs'])

# Filter data to include only relevant features (Latitude, Longitude, and Costs)
data_filtered = data[['Accident Latitude', 'Accident Longitude', 'All Costs']].dropna()

# Initialize a world map centered around the mean location
m = folium.Map(location=[data_filtered['Accident Latitude'].mean(), data_filtered['Accident Longitude'].mean()], zoom_start=2)  # Global view with zoom level 2

# Add accidents as points on the map
for idx, row in data_filtered.iterrows():
    folium.CircleMarker(location=[row['Accident Latitude'], row['Accident Longitude']],
                        radius=5, 
                        weight=1,
                        popup=f"Cost: ${row['All Costs']}",
                        fill=True,
                        color='blue',
                        fill_opacity=0.6).add_to(m)

# Save the accident world map to an HTML file
m.save("world_accident_map.html")
print("World accident map saved as 'world_accident_map.html'.")

# Create a heatmap to visualize accident hotspots on a global scale
heat_map = folium.Map(location=[data_filtered['Accident Latitude'].mean(), data_filtered['Accident Longitude'].mean()], zoom_start=2)  # Global heatmap with zoom level 2
HeatMap(data_filtered[['Accident Latitude', 'Accident Longitude', 'All Costs']].values, radius=15).add_to(heat_map)
heat_map.save("world_heatmap.html")

print("Global heatmap saved as 'world_heatmap.html'.")

# DBSCAN clustering for high-risk areas
coords = data_filtered[['Accident Latitude', 'Accident Longitude']].values
db = DBSCAN(eps=0.5, min_samples=5).fit(coords)
labels = db.labels_

# Plot the clusters on the world map using Folium
cluster_map = folium.Map(location=[data_filtered['Accident Latitude'].mean(), data_filtered['Accident Longitude'].mean()], zoom_start=2)

# Add clusters to the map
for label in set(labels):
    color = 'black' if label == -1 else 'red'  # Mark noise points with black, clusters with red
    cluster_data = data_filtered[labels == label]
    
    for idx, row in cluster_data.iterrows():
        folium.CircleMarker(location=[row['Accident Latitude'], row['Accident Longitude']],
                            radius=5, 
                            weight=1,
                            popup=f"Cost: ${row['All Costs']}, Cluster: {label}",
                            fill=True,
                            color=color,
                            fill_opacity=0.7).add_to(cluster_map)

# Save the cluster map to an HTML file
cluster_map.save("world_cluster_map.html")

print("Clustered world map saved as 'world_cluster_map.html'.")

# Assign risk scores based on total accident costs for each location
risk_scores = data_filtered.groupby(['Accident Latitude', 'Accident Longitude'])['All Costs'].sum().reset_index()

# Normalize risk scores to a scale of 0 to 100
risk_scores['Risk Score'] = 100 * (risk_scores['All Costs'] / risk_scores['All Costs'].max())

# Display top 5 high-risk areas
print("\nTop 5 High-Risk Areas based on Accident Costs:")
print(risk_scores.sort_values(by='Risk Score', ascending=False).head())

# Save the risk scores to a CSV file
risk_scores.to_csv("risk_scores.csv", index=False)

print("Risk scores saved as 'risk_scores.csv'.")

# Prepare features and target variable for prediction
features = ['Property Damage Costs', 'Lost Commodity Costs', 
            'Public/Private Property Damage Costs', 'Emergency Response Costs', 
            'Environmental Remediation Costs', 'Other Costs']
target = 'All Costs'

# Prepare feature and target datasets
X = data[features]
y = data[target]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)

# Gaussian Process Regression
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gpr_model.fit(X_train_scaled, y_train)
y_pred_gpr = gpr_model.predict(X_test_scaled)

# XGBoost Regression
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)

# Ensemble Predictions - Average of RF, GPR, and XGBoost
ensemble_predictions = (rf_predictions + y_pred_gpr + xgb_predictions) / 3

# Evaluate Model Performance
ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
ensemble_r2 = r2_score(y_test, ensemble_predictions)
ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)

print("\nTesting the Ensemble Model Performance:")
print(f'Mean Squared Error: {ensemble_mse}')
print(f'R^2 Score: {ensemble_r2}')
print(f'Mean Absolute Error: {ensemble_mae}')

# Display final predictions vs actual values for Ensemble
results_ensemble = pd.DataFrame({
    'Actual': y_test,
    'Random Forest Predicted': rf_predictions,
    'GPR Predicted': y_pred_gpr,
    'XGBoost Predicted': xgb_predictions,
    'Ensemble Predicted': ensemble_predictions
})
print("\nEnsemble Predictions vs Actual:")
print(results_ensemble)

# Plotting Actual vs Predicted for each model
plt.figure(figsize=(20, 5))

# Random Forest Predictions
plt.subplot(1, 4, 1)
plt.scatter(y_test, rf_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Random Forest Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Gaussian Process Predictions
plt.subplot(1, 4, 2)
plt.scatter(y_test, y_pred_gpr, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Gaussian Process Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# XGBoost Predictions
plt.subplot(1, 4, 3)
plt.scatter(y_test, xgb_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('XGBoost Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

# Ensemble Predictions
plt.subplot(1, 4, 4)
plt.scatter(y_test, ensemble_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Ensemble Predictions')
plt.xlabel('Actual')
plt.ylabel('Predicted')

plt.tight_layout()
plt.show()

# Step 1: Predict on the full dataset
X_scaled = scaler.transform(X)  # Scale the full dataset

# Make predictions on the entire dataset using the trained models
rf_predictions_full = rf_model.predict(X_scaled)
gpr_predictions_full = gpr_model.predict(X_scaled)
xgb_predictions_full = xgb_model.predict(X_scaled)

# Ensemble predictions for the full dataset
ensemble_predictions_full = (rf_predictions_full + gpr_predictions_full + xgb_predictions_full) / 3

# Step 2: Merge the predictions back into the data_filtered DataFrame
data['Predicted All Costs'] = ensemble_predictions_full

# Step 3: Recalculate risk scores based on predicted costs
data['Predicted Risk Score'] = 100 * (data['Predicted All Costs'] / data['Predicted All Costs'].max())

# Display top 5 high-risk areas based on predicted costs
predicted_risk_scores = data[['Accident Latitude', 'Accident Longitude', 'Predicted All Costs', 'Predicted Risk Score']]
print("\nTop 5 Predicted High-Risk Areas based on Predicted Costs:")
print(predicted_risk_scores.sort_values(by='Predicted Risk Score', ascending=False).head())

# Save the predicted risk scores (including Predicted All Costs) to a CSV file
predicted_risk_scores.to_csv("predicted_risk_scores.csv", index=False)
print("Predicted risk scores saved as 'predicted_risk_scores.csv'.")
