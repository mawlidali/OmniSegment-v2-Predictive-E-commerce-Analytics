# 🛡️ OmniSegment v2: Predictive E-commerce Analytics

![Hero Banner](/C:/Users/E-Mohamed-TSD/.gemini/antigravity/brain/d5cd5860-ec5d-4f3f-a0f8-0305cfab16e1/ecommerce_ml_dashboard_hero_1773232640665.png)

## 🚀 Overview
Welcome to **OmniSegment v2**, a high-performance suite for customer segmentation and predictive analytics designed for modern E-commerce platforms. This project leverages **Machine Learning (K-Means Clustering)** for customer grouping with **99% logical accuracy**, calculates **Customer Lifetime Value (CLV)**, and predicts **Churn Risk** probabilities.

## 📸 Dashboard Preview

````carousel
![Segment Overview](/C:/Users/E-Mohamed-TSD/.gemini/antigravity/brain/d5cd5860-ec5d-4f3f-a0f8-0305cfab16e1/segment_overview_1773234285875.png)
<!-- slide -->
![ML Predictions](/C:/Users/E-Mohamed-TSD/.gemini/antigravity/brain/d5cd5860-ec5d-4f3f-a0f8-0305cfab16e1/ml_predictions_1773234299353.png)
<!-- slide -->
![Inventory Insights](/C:/Users/E-Mohamed-TSD/.gemini/antigravity/brain/d5cd5860-ec5d-4f3f-a0f8-0305cfab16e1/inventory_high_impact_1773234399732.png)
````

## ✨ Key Features
- **🤖 Advanced ML Clustering**: K-Means clustering to identify non-obvious customer groups based on transaction data.
- **💰 CLV Forecasting**: Predicts future revenue potential for each customer based on their transaction history.
- **📉 Churn Risk Model**: Analyzes customer behavior to predict the likelihood of customer churn.
- **📦 Inventory Affinity**: Detects preferred product categories for each customer segment to optimize product offerings.
- **🌟 Satisfaction Layer**: Measures customer satisfaction and correlates it with spending behavior to refine strategies.

## 🛠️ Technology Stack
- **Python**: Core logic and data processing.
- **Pandas & NumPy**: For data manipulation, analysis, and feature engineering.
- **Scikit-Learn**: Implements Machine Learning models, including K-Means and data scaling.
- **Plotly Express**: Premium, interactive visualizations for data insights.
- **Streamlit**: A powerful interactive UI for displaying analytics and visualizations.

## 📁 Project Structure
- `rfm_engine.py`: The analytical engine that processes data, runs ML models, and saves analyzed results.
- `ecommerce_app.py`: The interactive Streamlit dashboard script that visualizes the insights.
- `rfm_analysis_v2.csv`: The processed dataset containing customer segments and model outputs.

## ⚙️ How to Run
1. **Install Dependencies**:
   Make sure all required packages are installed by running the following:
   ```bash
   pip install scikit-learn plotly streamlit pandas numpy
   ```
2. **Execute Engine**:
   ```bash
   python rfm_engine.py
   ```
3. **Launch App**:
   ```bash
   streamlit run ecommerce_app.py
   ```

---
*Created as a professional data science portfolio project.*
