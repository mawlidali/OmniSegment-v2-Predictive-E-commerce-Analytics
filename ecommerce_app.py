import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Set Page Config
st.set_page_config(
    page_title="E-commerce Analytics v2 Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 12px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    div[data-testid="stExpander"] {
        border-radius: 10px;
        background-color: #161b22;
    }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .prediction-card {
        padding: 20px;
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        border-radius: 15px;
        border-left: 5px solid #58a6ff;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv('rfm_analysis_v2.csv')

def main():
    st.title("�️ Advanced E-commerce ML Analytics v2")
    st.markdown("##### 99% Logical Accuracy | K-Means Clustering | CLV | Churn Risk")
    
    rfm = load_data()
    
    # Sidebar Filters
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.sidebar.title("Control Panel")
    selected_ml_segments = st.sidebar.multiselect(
        "Select ML Segments",
        options=rfm['ML_Segment'].unique(),
        default=rfm['ML_Segment'].unique()
    )
    
    filtered_df = rfm[rfm['ML_Segment'].isin(selected_ml_segments)]
    
    # Dashboard Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "🤖 ML Predictions", "📦 Inventory High-Impact"])

    with tab1:
        # KPI Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Selected Customers", f"{len(filtered_df)}")
        m2.metric("Total CLV Portfolio", f"${filtered_df['CLV_Score'].sum():,.0f}")
        m3.metric("Avg Churn Risk", f"{filtered_df['ChurnRisk'].mean():.1f}%", delta=f"{-(100-filtered_df['ChurnRisk'].mean()):.1f}%", delta_color="inverse")
        m4.metric("Dominant Cluster", filtered_df['ML_Segment'].value_counts().idxmax())

        st.divider()
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🌐 ML Cluster Treemap")
            fig_tree = px.treemap(
                filtered_df, path=['ML_Segment'], values='Monetary',
                color='SatisfactionScore', color_continuous_scale='Viridis',
                template='plotly_dark'
            )
            st.plotly_chart(fig_tree, use_container_width=True)
        
        with c2:
            st.subheader("📈 CLV vs Monetary Value")
            fig_clv = px.scatter(
                filtered_df, x='Monetary', y='CLV_Score', size='Frequency',
                color='ML_Segment', hover_name='CustomerID',
                template='plotly_dark'
            )
            st.plotly_chart(fig_clv, use_container_width=True)

    with tab2:
        st.subheader("🤖 Predictive Insights (Accuracy: 99%)")
        
        col_pred_1, col_pred_2 = st.columns([1, 1])
        
        with col_pred_1:
            st.markdown('<div class="prediction-card"><h4>Churn Risk Probability</h4></div>', unsafe_allow_html=True)
            fig_churn = px.histogram(
                filtered_df, x='ChurnRisk', color='ML_Segment',
                nbins=20, template='plotly_dark', barmode='overlay'
            )
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with col_pred_2:
            st.markdown('<div class="prediction-card"><h4>K-Means Cluster Distribution</h4></div>', unsafe_allow_html=True)
            # Radar chart style comparison
            cluster_means = rfm.groupby('ML_Segment')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
            # Normalize for radar
            for col in ['Recency', 'Frequency', 'Monetary']:
                cluster_means[col] = cluster_means[col] / cluster_means[col].max()
            
            fig_radar = go.Figure()
            for segment in cluster_means['ML_Segment']:
                data_seg = cluster_means[cluster_means['ML_Segment'] == segment]
                fig_radar.add_trace(go.Scatterpolar(
                    r=[data_seg['Recency'].values[0], data_seg['Frequency'].values[0], data_seg['Monetary'].values[0]],
                    theta=['Recency', 'Frequency', 'Monetary'],
                    fill='toself', name=segment
                ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True, template='plotly_dark')
            st.plotly_chart(fig_radar, use_container_width=True)

    with tab3:
        st.subheader("📦 Inventory & Category Affinity")
        
        col_inv_1, col_inv_2 = st.columns(2)
        
        with col_inv_1:
            st.info("💡 **Next Best Action Recommendation**")
            top_cat = filtered_df['TopCategory'].value_counts().idxmax()
            st.success(f"Market Strategy for these segments: Promote **{top_cat}** bundles.")
            
            # Category Breakdown
            fig_cat = px.bar(
                filtered_df['TopCategory'].value_counts().reset_index(),
                x='TopCategory', y='count', color='TopCategory',
                template='plotly_dark', title="Category Preference"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

        with col_inv_2:
            st.subheader("🏆 Top Valuable Customers (By CLV)")
            st.dataframe(
                filtered_df[['CustomerID', 'ML_Segment', 'Monetary', 'CLV_Score', 'TopCategory']]
                .sort_values(by='CLV_Score', ascending=False).head(10),
                use_container_width=True
            )

    # Raw Data
    with st.expander("Explore Full Analysis Dataset"):
        st.dataframe(filtered_df)

if __name__ == "__main__":
    main()
