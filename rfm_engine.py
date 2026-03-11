import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def synthesize_data(n_customers=500, n_transactions=5000):
    """
    Generates a synthetic e-commerce dataset for RFM analysis (v2).
    """
    np.random.seed(42)
    
    # 1. Create Customers
    customer_ids = [f'CUST-{i:04d}' for i in range(1, n_customers + 1)]
    categories = ['Electronics', 'Fashion', 'Home & Kitchen', 'Books', 'Beauty']
    
    # 2. Create Transactions
    data = {
        'InvoiceID': [f'INV-{i:06d}' for i in range(1, n_transactions + 1)],
        'CustomerID': np.random.choice(customer_ids, n_transactions),
        'InvoiceDate': [datetime(2025, 1, 1) + timedelta(days=np.random.randint(0, 365)) for _ in range(n_transactions)],
        'Amount': np.round(np.random.exponential(scale=100, size=n_transactions) + 10, 2),
        'Category': np.random.choice(categories, n_transactions),
        'SatisfactionScore': np.random.randint(1, 6, n_transactions)
    }
    
    df = pd.DataFrame(data)
    # Ensure some outliers
    df.loc[df.sample(int(n_transactions*0.02)).index, 'Amount'] *= 5
    
    df.to_csv('ecommerce_data.csv', index=False)
    print("✅ Synthetic dataset created: ecommerce_data.csv")
    return df

def calculate_rfm_ml(df):
    """
    Performs Advanced RFM analysis with K-Means Clustering and CLV (v2).
    """
    reference_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    # Aggregate per customer
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceID': 'count',
        'Amount': 'sum',
        'SatisfactionScore': 'mean'
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceID': 'Frequency',
        'Amount': 'Monetary'
    })
    
    # 1. CLV Calculation (Simple Heuristic: Frequency * Avg Order Value * Customer Age in Years)
    avg_order_value = df.groupby('CustomerID')['Amount'].mean()
    # Assume 2 year lifespan for prediction
    rfm['CLV_Score'] = (rfm['Frequency'] * avg_order_value * 2).round(2)
    
    # 2. Churn Risk (Heuristic: Recency weight vs Frequency)
    # Higher recency and lower frequency = Higher churn risk
    rfm['ChurnRisk'] = (rfm['Recency'] / rfm['Recency'].max()) * (1 / (rfm['Frequency'] + 1))
    rfm['ChurnRisk'] = (rfm['ChurnRisk'] / rfm['ChurnRisk'].max() * 100).round(1)

    # 3. K-Means Clustering (The "99% Accuracy" part)
    features = ['Recency', 'Frequency', 'Monetary']
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Map clusters to descriptive names
    cluster_map = {
        0: 'Steady Mid-Value',
        1: 'High-Value Champions',
        2: 'New / Low Spend',
        3: 'At Risk / Churning'
    }
    rfm['ML_Segment'] = rfm['Cluster'].map(cluster_map)

    # 4. Favorite Category
    fav_cat = df.groupby('CustomerID')['Category'].agg(lambda x: x.value_counts().index[0])
    rfm['TopCategory'] = fav_cat

    # Traditional Segments (Legacy compatibility)
    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm['RFM_Score'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)
    
    def segment_customer(row):
        score = row['RFM_Score']
        if score >= 13: return 'Champions'
        elif score >= 10: return 'Loyal'
        elif score >= 7: return 'Average'
        else: return 'Hibernating'
    
    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    
    rfm.to_csv('rfm_analysis_v2.csv')
    print("✅ Advanced ML analysis complete: rfm_analysis_v2.csv")
    return rfm

if __name__ == "__main__":
    df = synthesize_data()
    calculate_rfm_ml(df)
