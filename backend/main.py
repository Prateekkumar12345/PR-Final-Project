# backend.py - FastAPI Backend for Accident Risk Dashboard

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
import xgboost as xgb
import json
import io
import base64
from datetime import datetime
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

class AccidentRiskAnalyzer:
    """Main class for accident risk analysis and prediction"""
    
    def __init__(self, data_path: str = 'database.csv'):
        self.data_path = data_path
        self.data = None
        self.data_filtered = None
        self.models = {}
        self.scaler = StandardScaler()
        self.risk_scores = None
        self.predicted_risk_scores = None
        
        # Model performance metrics
        self.model_metrics = {}
        
        # Load and prepare data
        self.load_data()
        self.prepare_data()
        
    def load_data(self):
        """Load and clean the dataset"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {len(self.data)} records")
        except FileNotFoundError:
            print(f"Warning: Data file {self.data_path} not found. Using sample data.")
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for demonstration"""
        np.random.seed(42)
        n_samples = 1000
        
        # Generate sample coordinates (focus on major regions)
        lats = np.concatenate([
            np.random.normal(40.7128, 5, n_samples//4),  # NY region
            np.random.normal(34.0522, 5, n_samples//4),  # LA region
            np.random.normal(51.5074, 3, n_samples//4),  # London region
            np.random.normal(35.6762, 4, n_samples//4),  # Tokyo region
        ])
        
        lngs = np.concatenate([
            np.random.normal(-74.0060, 5, n_samples//4),  # NY region
            np.random.normal(-118.2437, 5, n_samples//4), # LA region  
            np.random.normal(-0.1278, 3, n_samples//4),   # London region
            np.random.normal(139.6503, 4, n_samples//4),  # Tokyo region
        ])
        
        # Generate cost data
        property_damage = np.random.exponential(50000, n_samples)
        commodity_costs = np.random.exponential(30000, n_samples)
        public_damage = np.random.exponential(20000, n_samples)
        emergency_costs = np.random.exponential(15000, n_samples)
        environmental_costs = np.random.exponential(40000, n_samples)
        other_costs = np.random.exponential(10000, n_samples)
        
        all_costs = (property_damage + commodity_costs + public_damage + 
                    emergency_costs + environmental_costs + other_costs)
        
        self.data = pd.DataFrame({
            'Accident Latitude': lats,
            'Accident Longitude': lngs,
            'All Costs': all_costs,
            'Property Damage Costs': property_damage,
            'Lost Commodity Costs': commodity_costs,
            'Public/Private Property Damage Costs': public_damage,
            'Emergency Response Costs': emergency_costs,
            'Environmental Remediation Costs': environmental_costs,
            'Other Costs': other_costs
        })
        
    def prepare_data(self):
        """Clean and prepare data for analysis"""
        # Handle missing values
        required_cols = ['Accident Latitude', 'Accident Longitude', 'All Costs', 
                        'Property Damage Costs', 'Lost Commodity Costs', 
                        'Public/Private Property Damage Costs', 'Emergency Response Costs', 
                        'Environmental Remediation Costs', 'Other Costs']
        
        self.data = self.data.dropna(subset=required_cols)
        self.data_filtered = self.data[['Accident Latitude', 'Accident Longitude', 'All Costs']].copy()
        
        print(f"Data prepared: {len(self.data_filtered)} valid records")
        
    def perform_clustering(self):
        """Perform DBSCAN clustering to identify accident hotspots"""
        coords = self.data_filtered[['Accident Latitude', 'Accident Longitude']].values
        db = DBSCAN(eps=0.5, min_samples=5).fit(coords)
        labels = db.labels_
        
        # Add cluster labels to data
        self.data_filtered['Cluster'] = labels
        
        # Calculate cluster statistics
        cluster_stats = []
        for label in set(labels):
            if label != -1:  # Ignore noise points
                cluster_data = self.data_filtered[self.data_filtered['Cluster'] == label]
                stats = {
                    'cluster_id': int(label),
                    'size': len(cluster_data),
                    'center_lat': float(cluster_data['Accident Latitude'].mean()),
                    'center_lng': float(cluster_data['Accident Longitude'].mean()),
                    'total_cost': float(cluster_data['All Costs'].sum()),
                    'avg_cost': float(cluster_data['All Costs'].mean())
                }
                cluster_stats.append(stats)
        
        return cluster_stats
        
    def calculate_risk_scores(self):
        """Calculate risk scores for different locations"""
        # Group by location and sum costs
        self.risk_scores = (self.data_filtered
                           .groupby(['Accident Latitude', 'Accident Longitude'])['All Costs']
                           .sum()
                           .reset_index())
        
        # Normalize to 0-100 scale
        max_cost = self.risk_scores['All Costs'].max()
        self.risk_scores['Risk Score'] = 100 * (self.risk_scores['All Costs'] / max_cost)
        
        return self.risk_scores.to_dict('records')
        
    def train_models(self):
        """Train ensemble of ML models for cost prediction"""
        features = ['Property Damage Costs', 'Lost Commodity Costs', 
                   'Public/Private Property Damage Costs', 'Emergency Response Costs', 
                   'Environmental Remediation Costs', 'Other Costs']
        target = 'All Costs'
        
        X = self.data[features]
        y = self.data[target]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_predictions = rf_model.predict(X_test_scaled)
        
        # Train Gaussian Process
        kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        gpr_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        gpr_model.fit(X_train_scaled, y_train)
        gpr_predictions = gpr_model.predict(X_test_scaled)
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        xgb_model.fit(X_train_scaled, y_train)
        xgb_predictions = xgb_model.predict(X_test_scaled)
        
        # Ensemble predictions
        ensemble_predictions = (rf_predictions + gpr_predictions + xgb_predictions) / 3
        
        # Store models
        self.models = {
            'random_forest': rf_model,
            'gaussian_process': gpr_model,
            'xgboost': xgb_model
        }
        
        # Calculate metrics
        self.model_metrics = {
            'ensemble': {
                'mse': float(mean_squared_error(y_test, ensemble_predictions)),
                'r2': float(r2_score(y_test, ensemble_predictions)),
                'mae': float(mean_absolute_error(y_test, ensemble_predictions))
            },
            'random_forest': {
                'mse': float(mean_squared_error(y_test, rf_predictions)),
                'r2': float(r2_score(y_test, rf_predictions)),
                'mae': float(mean_absolute_error(y_test, rf_predictions))
            },
            'gaussian_process': {
                'mse': float(mean_squared_error(y_test, gpr_predictions)),
                'r2': float(r2_score(y_test, gpr_predictions)),
                'mae': float(mean_absolute_error(y_test, gpr_predictions))
            },
            'xgboost': {
                'mse': float(mean_squared_error(y_test, xgb_predictions)),
                'r2': float(r2_score(y_test, xgb_predictions)),
                'mae': float(mean_absolute_error(y_test, xgb_predictions))
            }
        }
        
        return self.model_metrics
        
    def predict_risks(self):
        """Generate predictions for the entire dataset"""
        if not self.models:
            raise ValueError("Models not trained yet!")
            
        features = ['Property Damage Costs', 'Lost Commodity Costs', 
                   'Public/Private Property Damage Costs', 'Emergency Response Costs', 
                   'Environmental Remediation Costs', 'Other Costs']
        
        X = self.data[features]
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        rf_pred = self.models['random_forest'].predict(X_scaled)
        gpr_pred = self.models['gaussian_process'].predict(X_scaled)
        xgb_pred = self.models['xgboost'].predict(X_scaled)
        
        # Ensemble prediction
        ensemble_pred = (rf_pred + gpr_pred + xgb_pred) / 3
        
        # Add predictions to data
        self.data['Predicted All Costs'] = ensemble_pred
        self.data['Predicted Risk Score'] = 100 * (ensemble_pred / ensemble_pred.max())
        
        # Calculate predicted risk scores by location
        self.predicted_risk_scores = (self.data
                                     .groupby(['Accident Latitude', 'Accident Longitude'])
                                     ['Predicted All Costs']
                                     .sum()
                                     .reset_index())
        
        self.predicted_risk_scores['Predicted Risk Score'] = (
            100 * (self.predicted_risk_scores['Predicted All Costs'] / 
                  self.predicted_risk_scores['Predicted All Costs'].max())
        )
        
        return self.predicted_risk_scores.to_dict('records')
        
    def generate_map_data(self):
        """Generate data for map visualizations"""
        map_data = {
            'accidents': self.data_filtered.to_dict('records'),
            'clusters': self.perform_clustering(),
            'heatmap_data': self.data_filtered[['Accident Latitude', 'Accident Longitude', 'All Costs']].values.tolist()
        }
        return map_data

# FastAPI Application
app = FastAPI(title="Accident Risk Analysis Dashboard", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzer
analyzer = AccidentRiskAnalyzer()

# Response Models
class ModelMetrics(BaseModel):
    mse: float
    r2: float
    mae: float

class ClusterInfo(BaseModel):
    cluster_id: int
    size: int
    center_lat: float
    center_lng: float
    total_cost: float
    avg_cost: float

class RiskLocation(BaseModel):
    latitude: float = None
    longitude: float = None
    risk_score: float = None
    total_cost: float = None

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Accident Risk Analysis Dashboard API", "status": "running"}

@app.get("/api/overview")
async def get_overview():
    """Get overview statistics"""
    data_stats = {
        "total_accidents": len(analyzer.data_filtered),
        "total_cost": float(analyzer.data_filtered['All Costs'].sum()),
        "avg_cost": float(analyzer.data_filtered['All Costs'].mean()),
        "max_cost": float(analyzer.data_filtered['All Costs'].max()),
        "min_cost": float(analyzer.data_filtered['All Costs'].min())
    }
    return data_stats

@app.get("/api/map-data")
async def get_map_data():
    """Get data for map visualizations"""
    return analyzer.generate_map_data()

@app.get("/api/clusters")
async def get_clusters():
    """Get clustering analysis results"""
    return analyzer.perform_clustering()

@app.get("/api/risk-scores")
async def get_risk_scores():
    """Get current risk scores"""
    return analyzer.calculate_risk_scores()

@app.post("/api/train-models")
async def train_models(background_tasks: BackgroundTasks):
    """Train machine learning models"""
    try:
        metrics = analyzer.train_models()
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    if not analyzer.model_metrics:
        raise HTTPException(status_code=404, detail="Models not trained yet")
    return analyzer.model_metrics

@app.get("/api/predictions")
async def get_predictions():
    """Get risk predictions"""
    try:
        predictions = analyzer.predict_risks()
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/top-risks")
async def get_top_risks(limit: int = 10):
    """Get top risk locations"""
    risk_scores = analyzer.calculate_risk_scores()
    sorted_risks = sorted(risk_scores, key=lambda x: x['Risk Score'], reverse=True)
    return sorted_risks[:limit]

# Static file serving for frontend
@app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Accident Risk Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: 250px 1fr;
            min-height: 100vh;
        }
        
        .sidebar {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px;
            box-shadow: 2px 0 10px rgba(0,0,0,0.1);
        }
        
        .sidebar h2 {
            color: #333;
            margin-bottom: 30px;
            font-size: 24px;
        }
        
        .nav-item {
            padding: 12px 15px;
            margin: 10px 0;
            cursor: pointer;
            border-radius: 8px;
            transition: all 0.3s ease;
            color: #666;
        }
        
        .nav-item:hover, .nav-item.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            transform: translateX(5px);
        }
        
        .main-content {
            padding: 30px;
            overflow-y: auto;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 20px 30px;
            margin-bottom: 30px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .header h1 {
            color: #333;
            font-size: 32px;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 16px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            border: 1px solid rgba(255, 255, 255, 0.18);
        }
        
        .stat-card h3 {
            color: #666;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        
        .stat-card .value {
            color: #333;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-card .change {
            color: #10B981;
            font-size: 14px;
        }
        
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .chart-container h3 {
            margin-bottom: 20px;
            color: #333;
        }
        
        #mapContainer {
            height: 500px;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .section {
            display: none;
        }
        
        .section.active {
            display: block;
        }
        
        .loading {
            text-align: center;
            padding: 50px;
            color: #666;
        }
        
        .error {
            color: #EF4444;
            text-align: center;
            padding: 20px;
        }
        
        .action-buttons {
            margin-bottom: 20px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            margin-right: 10px;
            transition: transform 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        
        .table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        
        .table th,
        .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        .table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        
        .metric-card h4 {
            color: #666;
            margin-bottom: 10px;
        }
        
        .metric-card .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="sidebar">
            <h2>Risk Analytics</h2>
            <div class="nav-item active" onclick="showSection('overview')">📊 Overview</div>
            <div class="nav-item" onclick="showSection('maps')">🗺️ Risk Maps</div>
            <div class="nav-item" onclick="showSection('clusters')">🎯 Hotspots</div>
            <div class="nav-item" onclick="showSection('models')">🤖 ML Models</div>
            <div class="nav-item" onclick="showSection('predictions')">🔮 Predictions</div>
        </div>
        
        <div class="main-content">
            <div class="header">
                <h1>Accident Risk Analysis Dashboard</h1>
                <p>Real-time analysis and prediction of accident risks using machine learning</p>
            </div>
            
            <!-- Overview Section -->
            <div id="overview" class="section active">
                <div class="stats-grid" id="statsGrid">
                    <!-- Stats will be populated by JavaScript -->
                </div>
                
                <div class="chart-container">
                    <h3>Cost Distribution</h3>
                    <canvas id="costChart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <!-- Maps Section -->
            <div id="maps" class="section">
                <div class="action-buttons">
                    <button class="btn" onclick="loadMap('accidents')">Show Accidents</button>
                    <button class="btn" onclick="loadMap('heatmap')">Heat Map</button>
                    <button class="btn" onclick="loadMap('clusters')">Clusters</button>
                </div>
                <div class="chart-container">
                    <h3>Risk Visualization Map</h3>
                    <div id="mapContainer"></div>
                </div>
            </div>
            
            <!-- Clusters Section -->
            <div id="clusters" class="section">
                <div class="chart-container">
                    <h3>Accident Hotspots</h3>
                    <div id="clustersTable"></div>
                </div>
            </div>
            
            <!-- Models Section -->
            <div id="models" class="section">
                <div class="action-buttons">
                    <button class="btn" onclick="trainModels()">Train Models</button>
                </div>
                <div class="metrics-grid" id="metricsGrid">
                    <!-- Metrics will be populated by JavaScript -->
                </div>
                <div class="chart-container">
                    <h3>Model Performance</h3>
                    <canvas id="metricsChart" width="400" height="200"></canvas>
                </div>
            </div>
            
            <!-- Predictions Section -->
            <div id="predictions" class="section">
                <div class="chart-container">
                    <h3>Risk Predictions</h3>
                    <div id="predictionsTable"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let map = null;
        let currentData = null;
        
        // Navigation
        function showSection(sectionName) {
            // Hide all sections
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from nav items
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Show selected section
            document.getElementById(sectionName).classList.add('active');
            event.target.classList.add('active');
            
            // Load section-specific data
            loadSectionData(sectionName);
        }
        
        // Load data for specific sections
        async function loadSectionData(section) {
            switch(section) {
                case 'overview':
                    await loadOverview();
                    break;
                case 'maps':
                    await loadMapData();
                    break;
                case 'clusters':
                    await loadClusters();
                    break;
                case 'models':
                    await loadModelMetrics();
                    break;
                case 'predictions':
                    await loadPredictions();
                    break;
            }
        }
        
        // Load overview statistics
        async function loadOverview() {
            try {
                const response = await fetch('/api/overview');
                const data = await response.json();
                
                const statsGrid = document.getElementById('statsGrid');
                statsGrid.innerHTML = `
                    <div class="stat-card">
                        <h3>Total Accidents</h3>
                        <div class="value">${data.total_accidents.toLocaleString()}</div>
                    </div>
                    <div class="stat-card">
                        <h3>Total Cost</h3>
                        <div class="value">$${(data.total_cost / 1000000).toFixed(1)}M</div>
                    </div>
                    <div class="stat-card">
                        <h3>Average Cost</h3>
                        <div class="value">$${(data.avg_cost / 1000).toFixed(1)}K</div>
                    </div>
                    <div class="stat-card">
                        <h3>Max Cost</h3>
                        <div class="value">$${(data.max_cost / 1000).toFixed(1)}K</div>
                    </div>
                `;
                
                // Create cost distribution chart
                createCostChart();
                
            } catch (error) {
                console.error('Error loading overview:', error);
            }
        }
        
        // Create cost distribution chart
        function createCostChart() {
            const ctx = document.getElementById('costChart').getContext('2d');
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: ['Property', 'Commodity', 'Public', 'Emergency', 'Environmental', 'Other'],
                    datasets: [{
                        label: 'Average Cost ($)',
                        data: [50000, 30000, 20000, 15000, 40000, 10000], // Sample data
                        backgroundColor: [
                            '#FF6384',
                            '#36A2EB',
                            '#FFCE56',
                            '#4BC0C0',
                            '#9966FF',
                            '#FF9F40'
                        ]
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true
                        }
                    }
                }
            });
        }
        
        // Load map data
        async function loadMapData() {
            try {
                const response = await fetch('/api/map-data');
                currentData = await response.json();
                
                // Initialize map if not exists
                if (!map) {
                    map = L.map('mapContainer').setView([40.7128, -74.0060], 2);
                    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
                }
                
                // Load accidents by default
                loadMap('accidents');
                
            } catch (error) {
                console.error('Error loading map data:', error);
            }
        }
        
        // Load specific map view
        function loadMap(type) {
            if (!map || !currentData) return;
            
            // Clear existing layers
            map.eachLayer(layer => {
                if (layer instanceof L.CircleMarker || layer instanceof L.HeatLayer) {
                    map.removeLayer(layer);
                }
            });
            
            switch(type) {
                case 'accidents':
                    currentData.accidents.forEach(accident => {
                        L.circleMarker([accident['Accident Latitude'], accident['Accident Longitude']], {
                            radius: 5,
                            fillColor: '#ff0000',
                            color: '#000',
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        }).bindPopup(`Cost: ${accident['All Costs'].toLocaleString()}`).addTo(map);
                    });
                    break;
                    
                case 'heatmap':
                    if (currentData.heatmap_data.length > 0) {
                        L.heatLayer(currentData.heatmap_data, {radius: 25}).addTo(map);
                    }
                    break;
                    
                case 'clusters':
                    currentData.clusters.forEach(cluster => {
                        L.circleMarker([cluster.center_lat, cluster.center_lng], {
                            radius: Math.max(10, Math.min(50, cluster.size / 2)),
                            fillColor: '#ff0000',
                            color: '#000',
                            weight: 2,
                            opacity: 1,
                            fillOpacity: 0.6
                        }).bindPopup(`
                            Cluster ${cluster.cluster_id}<br>
                            Size: ${cluster.size} accidents<br>
                            Total Cost: ${cluster.total_cost.toLocaleString()}<br>
                            Avg Cost: ${cluster.avg_cost.toLocaleString()}
                        `).addTo(map);
                    });
                    break;
            }
        }
        
        // Load clusters data
        async function loadClusters() {
            try {
                const response = await fetch('/api/clusters');
                const clusters = await response.json();
                
                const clustersTable = document.getElementById('clustersTable');
                let tableHtml = `
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Cluster ID</th>
                                <th>Size</th>
                                <th>Center Location</th>
                                <th>Total Cost</th>
                                <th>Average Cost</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                clusters.forEach(cluster => {
                    tableHtml += `
                        <tr>
                            <td>${cluster.cluster_id}</td>
                            <td>${cluster.size}</td>
                            <td>${cluster.center_lat.toFixed(4)}, ${cluster.center_lng.toFixed(4)}</td>
                            <td>${cluster.total_cost.toLocaleString()}</td>
                            <td>${cluster.avg_cost.toLocaleString()}</td>
                        </tr>
                    `;
                });
                
                tableHtml += '</tbody></table>';
                clustersTable.innerHTML = tableHtml;
                
            } catch (error) {
                console.error('Error loading clusters:', error);
                document.getElementById('clustersTable').innerHTML = '<div class="error">Error loading clusters data</div>';
            }
        }
        
        // Train ML models
        async function trainModels() {
            try {
                const response = await fetch('/api/train-models', {
                    method: 'POST'
                });
                const result = await response.json();
                
                if (result.status === 'success') {
                    alert('Models trained successfully!');
                    await loadModelMetrics();
                } else {
                    alert('Error training models');
                }
            } catch (error) {
                console.error('Error training models:', error);
                alert('Error training models');
            }
        }
        
        // Load model metrics
        async function loadModelMetrics() {
            try {
                const response = await fetch('/api/model-metrics');
                const metrics = await response.json();
                
                const metricsGrid = document.getElementById('metricsGrid');
                let gridHtml = '';
                
                Object.keys(metrics).forEach(modelName => {
                    const modelMetrics = metrics[modelName];
                    gridHtml += `
                        <div class="metric-card">
                            <h4>${modelName.replace('_', ' ').toUpperCase()}</h4>
                            <div class="metric-value">${modelMetrics.r2.toFixed(3)}</div>
                            <div>R² Score</div>
                        </div>
                    `;
                });
                
                metricsGrid.innerHTML = gridHtml;
                
                // Create metrics chart
                createMetricsChart(metrics);
                
            } catch (error) {
                console.error('Error loading model metrics:', error);
                document.getElementById('metricsGrid').innerHTML = '<div class="error">Train models first to see metrics</div>';
            }
        }
        
        // Create model metrics chart
        function createMetricsChart(metrics) {
            const ctx = document.getElementById('metricsChart').getContext('2d');
            
            const modelNames = Object.keys(metrics);
            const r2Scores = modelNames.map(name => metrics[name].r2);
            const maeScores = modelNames.map(name => metrics[name].mae);
            
            new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: modelNames.map(name => name.replace('_', ' ').toUpperCase()),
                    datasets: [{
                        label: 'R² Score',
                        data: r2Scores,
                        backgroundColor: '#36A2EB',
                        yAxisID: 'y'
                    }, {
                        label: 'MAE (scaled)',
                        data: maeScores.map(mae => mae / 10000), // Scale for visibility
                        backgroundColor: '#FF6384',
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            grid: {
                                drawOnChartArea: false,
                            },
                        }
                    }
                }
            });
        }
        
        // Load predictions
        async function loadPredictions() {
            try {
                const response = await fetch('/api/predictions');
                const predictions = await response.json();
                
                const predictionsTable = document.getElementById('predictionsTable');
                let tableHtml = `
                    <table class="table">
                        <thead>
                            <tr>
                                <th>Latitude</th>
                                <th>Longitude</th>
                                <th>Predicted Cost</th>
                                <th>Risk Score</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                // Show top 20 predictions
                predictions.slice(0, 20).forEach(pred => {
                    tableHtml += `
                        <tr>
                            <td>${pred['Accident Latitude'].toFixed(4)}</td>
                            <td>${pred['Accident Longitude'].toFixed(4)}</td>
                            <td>${pred['Predicted All Costs'].toLocaleString()}</td>
                            <td>${pred['Predicted Risk Score'].toFixed(1)}</td>
                        </tr>
                    `;
                });
                
                tableHtml += '</tbody></table>';
                predictionsTable.innerHTML = tableHtml;
                
            } catch (error) {
                console.error('Error loading predictions:', error);
                document.getElementById('predictionsTable').innerHTML = '<div class="error">Train models first to see predictions</div>';
            }
        }
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            loadOverview();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)

# Run the application
if __name__ == "__main__":
    print("🚀 Starting Accident Risk Analysis Dashboard...")
    print("✅ Features:")
    print("   - Interactive maps with accident visualization")
    print("   - DBSCAN clustering for hotspot identification") 
    print("   - Ensemble ML models (Random Forest, GPR, XGBoost)")
    print("   - Real-time risk scoring and predictions")
    print("   - Comprehensive dashboard with charts and tables")
    print("")
    print("📊 Dashboard will be available at: http://localhost:8000/dashboard")
    print("🔧 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )