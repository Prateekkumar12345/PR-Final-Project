✨ Features
📊 Interactive Overview Dashboard - Real-time statistics and cost distribution visualizations

🗺️ Geographical Risk Maps - Heatmaps and cluster visualization for accident hotspots

🎯 Smart Clustering - DBSCAN-based accident hotspot identification with cost analysis

🤖 ML Integration - Multiple machine learning models for risk prediction

🔮 Predictive Analytics - Future risk forecasting with detailed scoring

📱 Responsive Design - Works seamlessly on desktop, tablet, and mobile devices

🎨 Modern UI - Glassmorphism design with smooth animations


🔌 API Integration
The dashboard expects a Flask backend with the following endpoints:

Endpoint	Method	Description
/api/overview	GET	Overview statistics and metrics
/api/clusters	GET	Accident cluster data
/api/map-data	GET	Geographical data for visualization
/api/model-metrics	GET	Machine learning model performance
/api/predictions	GET	Risk prediction data
/api/train-models	POST	Trigger model training

📁 Project Structure
text
src/
├── components/
│   └── AccidentRiskDashboard.jsx  # Main dashboard component
├── App.js                         # Root application component
├── index.js                       # Application entry point
└── styles/
    └── globals.css                # Global styles (if needed)
