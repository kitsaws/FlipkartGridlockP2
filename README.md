# GridLock AI: Spatio-Temporal Enforcement Priority Intelligence

An end-to-end Machine Learning intelligence system that transforms historical parking violation data into a predictive enforcement strategy. By leveraging H3 spatial target encoding, Random Forest classification, and SHAP explainability, this system predicts future gridlock hotspots and provides traffic commanders with actionable, human-readable insights.

![Dashboard Preview](outputs/explainability/v2_shap_summary.png)

---

## 🏗️ Architectural Overview

The system operates on a modern **Client-Server Architecture**:

1.  **Data Engine (`scripts 10-15`)**: Python-based pipeline that processes 300,000+ raw parking tickets, maps coordinates to H3 hex grids, pads zero-inflated hours, and calculates 3-day and 7-day momentum features to track the escalation of traffic violations.
2.  **AI Microservice (`backend/`)**: A high-performance **FastAPI** server that runs a pre-trained Multi-Class Scikit-Learn `RandomForestClassifier`. It scales base probabilities mathematically in real-time based on the Commander's UI priorities.
3.  **Command Dashboard (`frontend/`)**: A sleek **Next.js 14** web application built with **Tailwind CSS v4**, **Shadcn UI**, and **Framer Motion**. It uses `react-leaflet` to render dynamic, interactive geo-spatial intelligence.

---

## ✨ Key Features

*   **Predictive Time Travel**: Slide forward in time to map exactly where future bottlenecks will occur based on momentum and cyclical patterns.
*   **Commander Control Panel**: Dynamically adjust the priority of specific violations (e.g., crank "Double Parking" to max priority) to instantly re-calculate AI probabilities without needing to re-train the model.
*   **Historical Profiler**: The system doesn't just flag a hotspot—it extracts the specific localized history to warn officers (e.g., *"Historically accounts for 83% of incidents here"*).
*   **Human-Readable AI (SHAP)**: Translates raw mathematical Tree Explainer outputs into plain-English alerts (e.g., *"Recent surge in dangerous parking (3-Day Trend)"*).

---

## 🚀 Quickstart Guide

This repository has been configured so that anyone can run the complete system out-of-the-box. The final 2.6M row CSV and the `.pkl` models are tracked via GitHub, so you do not need to run the massive Python aggregation scripts.

### Prerequisites
*   **Node.js 18+**
*   **Python 3.9+**

### Step 1: Start the AI Engine (FastAPI Backend)

Open a terminal in the root directory.

```bash
# Create a virtual environment (optional but recommended)
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the Microservice
uvicorn backend.main:app --port 8000
```
*(Wait until you see `Application startup complete`)*

### Step 2: Start the Web Dashboard (Next.js Frontend)

Open a **second** terminal window and navigate into the `frontend` directory.

```bash
cd frontend

# Install Node modules
npm install

# Start the development server
npm run dev
```

### Step 3: View the Intelligence
Open your browser and navigate to **[http://localhost:3000](http://localhost:3000)**. 

*(Note: The map handles hundreds of highly dense H3 polygon coordinates, so give it a second to smoothly animate onto your screen!)*

---

## 🧠 Model Training (Optional)

If you want to re-train the Random Forest from scratch using the raw data, run the scripts in order:

```bash
cd scripts

# 1. Generate 2.6M row zero-inflated predictive dataset
python 10_time_series_aggregation.py

# 2. Undersample and train the Multi-Class model
python 11_risk_classification_model.py

# 3. Generate Evaluation Metrics and SHAP summaries
python 14_explain_predictions.py
python 15_evaluate_models.py
```
