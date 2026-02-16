import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📊 Customer Segmentation Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Interactive K-Means Clustering with RFM Analysis")

# Function to load and clean data
@st.cache_data
def load_and_clean_data():
    """Load and clean the retail data"""
    try:
        df = pd.read_excel('data/Online Retail.xlsx')
        
        # Clean data
        df_clean = df.copy()
        df_clean = df_clean[df_clean['CustomerID'].notna()]
        df_clean = df_clean[~df_clean['InvoiceNo'].astype(str).str.startswith('C')]
        df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
        df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']
        
        return df_clean
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Function to calculate RFM
@st.cache_data
def calculate_rfm(df_clean):
    """Calculate RFM metrics"""
    reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df_clean.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalAmount': 'sum'
    }).reset_index()
    
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    return rfm

# Function to perform clustering
def perform_clustering(rfm, n_clusters):
    """Perform K-Means clustering"""
    X = rfm[['Recency', 'Frequency', 'Monetary']].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    wss = kmeans.inertia_
    silhouette = silhouette_score(X_scaled, rfm['Cluster'])
    
    return rfm, wss, silhouette, X_scaled, kmeans

# Load data
with st.spinner('Loading data...'):
    df_clean = load_and_clean_data()

if df_clean is not None:
    # Calculate RFM
    rfm = calculate_rfm(df_clean)
    
    # Sidebar controls
    st.sidebar.header("🎯 Clustering Parameters")
    
    # Number of clusters slider
    n_clusters = st.sidebar.slider(
        "Select Number of Clusters (k)",
        min_value=2,
        max_value=10,
        value=4,
        step=1,
        help="Choose the number of customer segments"
    )
    
    st.sidebar.markdown("---")
    
    # Show data info
    st.sidebar.header("📈 Data Overview")
    st.sidebar.metric("Total Customers", f"{len(rfm):,}")
    st.sidebar.metric("Total Transactions", f"{len(df_clean):,}")
    st.sidebar.metric("Total Revenue", f"${df_clean['TotalAmount'].sum():,.2f}")
    
    # Perform clustering
    rfm_clustered, wss, silhouette, X_scaled, kmeans = perform_clustering(rfm, n_clusters)
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Number of Clusters",
            value=n_clusters,
            delta=None
        )
    
    with col2:
        st.metric(
            label="WSS (Inertia)",
            value=f"{wss:.2f}",
            delta=None,
            help="Within-Cluster Sum of Squares - Lower is better"
        )
    
    with col3:
        st.metric(
            label="Silhouette Score",
            value=f"{silhouette:.4f}",
            delta=None,
            help="Cluster quality metric - Higher is better (range: -1 to 1)"
        )
    
    with col4:
        avg_customers_per_cluster = len(rfm_clustered) / n_clusters
        st.metric(
            label="Avg Customers/Cluster",
            value=f"{avg_customers_per_cluster:.0f}",
            delta=None
        )
    
    st.markdown("---")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 3D Visualization", 
        "📊 Cluster Analysis", 
        "📈 Distribution Charts",
        "📋 Data Table"
    ])
    
    # Tab 1: 3D Visualization
    with tab1:
        st.subheader("3D Customer Segmentation Visualization")
        
        # Create 3D scatter plot
        fig_3d = go.Figure()
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        
        for cluster in range(n_clusters):
            cluster_data = rfm_clustered[rfm_clustered['Cluster'] == cluster]
            
            fig_3d.add_trace(go.Scatter3d(
                x=cluster_data['Recency'],
                y=cluster_data['Frequency'],
                z=cluster_data['Monetary'],
                mode='markers',
                name=f'Cluster {cluster}',
                marker=dict(
                    size=5,
                    color=colors[cluster],
                    opacity=0.6,
                    line=dict(color='black', width=0.5)
                ),
                text=[f'Customer {cid}<br>Recency: {r}<br>Frequency: {f}<br>Monetary: ${m:.2f}'
                      for cid, r, f, m in zip(cluster_data['CustomerID'], 
                                              cluster_data['Recency'],
                                              cluster_data['Frequency'],
                                              cluster_data['Monetary'])],
                hoverinfo='text'
            ))
        
        fig_3d.update_layout(
            title=f'Customer Segments in 3D RFM Space (k={n_clusters})',
            scene=dict(
                xaxis_title='Recency (Days)',
                yaxis_title='Frequency (Purchases)',
                zaxis_title='Monetary (Amount $)',
                xaxis=dict(backgroundcolor="rgb(230, 230,230)"),
                yaxis=dict(backgroundcolor="rgb(230, 230,230)"),
                zaxis=dict(backgroundcolor="rgb(230, 230,230)"),
            ),
            width=1200,
            height=700,
            showlegend=True,
            legend=dict(x=0.7, y=0.9)
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Add cluster centroids
        st.markdown("#### 📍 Cluster Centroids")
        centroids_df = pd.DataFrame(
            kmeans.cluster_centers_,
            columns=['Recency (Scaled)', 'Frequency (Scaled)', 'Monetary (Scaled)']
        )
        centroids_df.insert(0, 'Cluster', range(n_clusters))
        st.dataframe(centroids_df, use_container_width=True)
    
    # Tab 2: Cluster Analysis
    with tab2:
        st.subheader("Cluster Profile Analysis")
        
        # Calculate cluster statistics
        cluster_stats = rfm_clustered.groupby('Cluster').agg({
            'CustomerID': 'count',
            'Recency': ['mean', 'median'],
            'Frequency': ['mean', 'median'],
            'Monetary': ['mean', 'median', 'sum']
        }).round(2)
        
        cluster_stats.columns = ['Customer_Count', 'Avg_Recency', 'Med_Recency',
                                'Avg_Frequency', 'Med_Frequency',
                                'Avg_Monetary', 'Med_Monetary', 'Total_Revenue']
        
        cluster_stats['Revenue_%'] = (cluster_stats['Total_Revenue'] / 
                                      cluster_stats['Total_Revenue'].sum() * 100).round(1)
        cluster_stats['Customer_%'] = (cluster_stats['Customer_Count'] / 
                                       cluster_stats['Customer_Count'].sum() * 100).round(1)
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Visualize cluster characteristics
        st.markdown("#### 📊 Cluster Characteristics Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart for average RFM values
            fig_bar = go.Figure()
            
            metrics = ['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary']
            
            for metric in metrics:
                fig_bar.add_trace(go.Bar(
                    name=metric.replace('Avg_', ''),
                    x=cluster_stats.index,
                    y=cluster_stats[metric],
                ))
            
            fig_bar.update_layout(
                title='Average RFM Values by Cluster',
                xaxis_title='Cluster',
                yaxis_title='Value',
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Pie chart for customer distribution
            fig_pie = go.Figure(data=[go.Pie(
                labels=[f'Cluster {i}' for i in cluster_stats.index],
                values=cluster_stats['Customer_Count'],
                hole=0.3,
                marker_colors=colors[:n_clusters]
            )])
            
            fig_pie.update_layout(
                title='Customer Distribution Across Clusters',
                height=400
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Revenue distribution
        st.markdown("#### 💰 Revenue Distribution")
        fig_revenue = go.Figure(data=[go.Pie(
            labels=[f'Cluster {i}' for i in cluster_stats.index],
            values=cluster_stats['Total_Revenue'],
            hole=0.3,
            marker_colors=colors[:n_clusters]
        )])
        
        fig_revenue.update_layout(
            title='Revenue Distribution Across Clusters',
            height=400
        )
        
        st.plotly_chart(fig_revenue, use_container_width=True)
    
    # Tab 3: Distribution Charts
    with tab3:
        st.subheader("RFM Distribution by Cluster")
        
        # Create box plots
        metric_to_plot = st.selectbox(
            "Select Metric to Visualize",
            options=['Recency', 'Frequency', 'Monetary'],
            index=0
        )
        
        fig_box = px.box(
            rfm_clustered,
            x='Cluster',
            y=metric_to_plot,
            color='Cluster',
            title=f'{metric_to_plot} Distribution by Cluster',
            color_discrete_sequence=colors[:n_clusters]
        )
        
        fig_box.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # 2D Scatter plots
        st.markdown("#### 2D Projections")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig_2d_1 = px.scatter(
                rfm_clustered,
                x='Recency',
                y='Frequency',
                color='Cluster',
                title='Recency vs Frequency',
                color_discrete_sequence=colors[:n_clusters],
                opacity=0.6
            )
            fig_2d_1.update_layout(height=400)
            st.plotly_chart(fig_2d_1, use_container_width=True)
        
        with col2:
            fig_2d_2 = px.scatter(
                rfm_clustered,
                x='Recency',
                y='Monetary',
                color='Cluster',
                title='Recency vs Monetary',
                color_discrete_sequence=colors[:n_clusters],
                opacity=0.6
            )
            fig_2d_2.update_layout(height=400)
            st.plotly_chart(fig_2d_2, use_container_width=True)
        
        with col3:
            fig_2d_3 = px.scatter(
                rfm_clustered,
                x='Frequency',
                y='Monetary',
                color='Cluster',
                title='Frequency vs Monetary',
                color_discrete_sequence=colors[:n_clusters],
                opacity=0.6
            )
            fig_2d_3.update_layout(height=400)
            st.plotly_chart(fig_2d_3, use_container_width=True)
    
    # Tab 4: Data Table
    with tab4:
        st.subheader("Customer Segmentation Data")
        
        # Filter options
        col1, col2 = st.columns(2)
        
        with col1:
            selected_clusters = st.multiselect(
                "Filter by Cluster",
                options=sorted(rfm_clustered['Cluster'].unique()),
                default=sorted(rfm_clustered['Cluster'].unique())
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by",
                options=['Recency', 'Frequency', 'Monetary'],
                index=2
            )
        
        # Filter and sort data
        filtered_data = rfm_clustered[rfm_clustered['Cluster'].isin(selected_clusters)]
        filtered_data = filtered_data.sort_values(by=sort_by, ascending=False)
        
        st.dataframe(
            filtered_data.style.format({
                'Recency': '{:.0f}',
                'Frequency': '{:.0f}',
                'Monetary': '${:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Segmented Data as CSV",
            data=csv,
            file_name=f"customer_segments_k{n_clusters}.csv",
            mime="text/csv",
        )
    
    # Sidebar - Model Evaluation
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Evaluation")
    
    # Calculate metrics for different k values
    if st.sidebar.checkbox("Show Elbow & Silhouette Analysis", value=False):
        st.markdown("---")
        st.subheader("🔍 Optimal Cluster Selection Analysis")
        
        with st.spinner('Calculating metrics for different k values...'):
            k_range = range(2, 11)
            wss_values = []
            silhouette_values = []
            
            for k in k_range:
                kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels_temp = kmeans_temp.fit_predict(X_scaled)
                wss_values.append(kmeans_temp.inertia_)
                silhouette_values.append(silhouette_score(X_scaled, labels_temp))
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Elbow plot
                fig_elbow = go.Figure()
                fig_elbow.add_trace(go.Scatter(
                    x=list(k_range),
                    y=wss_values,
                    mode='lines+markers',
                    name='WSS',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))
                
                # Highlight current k
                current_idx = n_clusters - 2
                fig_elbow.add_trace(go.Scatter(
                    x=[n_clusters],
                    y=[wss_values[current_idx]],
                    mode='markers',
                    name='Current k',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                fig_elbow.update_layout(
                    title='Elbow Method (WSS)',
                    xaxis_title='Number of Clusters (k)',
                    yaxis_title='Within-Cluster Sum of Squares',
                    height=400
                )
                
                st.plotly_chart(fig_elbow, use_container_width=True)
            
            with col2:
                # Silhouette plot
                fig_silhouette = go.Figure()
                fig_silhouette.add_trace(go.Scatter(
                    x=list(k_range),
                    y=silhouette_values,
                    mode='lines+markers',
                    name='Silhouette Score',
                    line=dict(color='green', width=3),
                    marker=dict(size=10)
                ))
                
                # Highlight current k
                fig_silhouette.add_trace(go.Scatter(
                    x=[n_clusters],
                    y=[silhouette_values[current_idx]],
                    mode='markers',
                    name='Current k',
                    marker=dict(size=15, color='red', symbol='star')
                ))
                
                fig_silhouette.update_layout(
                    title='Silhouette Score',
                    xaxis_title='Number of Clusters (k)',
                    yaxis_title='Silhouette Score',
                    height=400
                )
                
                st.plotly_chart(fig_silhouette, use_container_width=True)
            
            # Best k recommendation
            best_silhouette_k = list(k_range)[silhouette_values.index(max(silhouette_values))]
            st.info(f"💡 **Recommendation:** Based on Silhouette Score, the optimal number of clusters is **k={best_silhouette_k}** (Score: {max(silhouette_values):.4f})")

else:
    st.error("❌ Failed to load data. Please check if the data file exists in the 'data' folder.")
    st.info("Expected file path: `data/Online Retail.xlsx`")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>📊 Customer Segmentation Dashboard | Built with Streamlit & K-Means Clustering</p>
        <p>Use the sidebar to adjust the number of clusters and explore different segmentations</p>
    </div>
""", unsafe_allow_html=True)
