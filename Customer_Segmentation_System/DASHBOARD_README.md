# Customer Segmentation Streamlit Dashboard

## 🚀 Quick Start

### Running the Dashboard

1. Open a terminal in the project folder
2. Run the following command:

```bash
streamlit run streamlit_app.py
```

3. The dashboard will open in your browser at `http://localhost:8501`

## 📊 Features

### Interactive Controls
- **Cluster Selection Slider**: Choose the number of clusters (k) from 2 to 10
- **Real-time Updates**: All visualizations update instantly when you change k

### Visualizations

#### 🌐 3D Visualization Tab
- Interactive 3D scatter plot showing customers in RFM space
- Hover over points to see customer details
- Rotate, zoom, and pan the 3D view
- Color-coded clusters with legend
- Cluster centroids table

#### 📊 Cluster Analysis Tab
- Detailed cluster statistics table
- Average RFM values by cluster
- Customer distribution pie chart
- Revenue distribution across clusters
- Comparison charts

#### 📈 Distribution Charts Tab
- Box plots for Recency, Frequency, and Monetary values
- 2D scatter plots:
  - Recency vs Frequency
  - Recency vs Monetary
  - Frequency vs Monetary
- Interactive filtering and zooming

#### 📋 Data Table Tab
- Complete customer segmentation data
- Filter by cluster
- Sort by any RFM metric
- Download button for CSV export

### Advanced Features

#### 📊 Model Evaluation (Optional)
- Check "Show Elbow & Silhouette Analysis" in sidebar
- Elbow method plot (WSS vs k)
- Silhouette score plot
- Automatic recommendation for optimal k

### Metrics Displayed
- Number of Clusters
- WSS (Within-Cluster Sum of Squares)
- Silhouette Score
- Average Customers per Cluster
- Total Customers
- Total Transactions
- Total Revenue

## 🎯 How to Use

1. **Adjust k**: Use the slider in the sidebar to select the number of clusters
2. **Explore 3D View**: Navigate to the "3D Visualization" tab and interact with the plot
3. **Analyze Clusters**: Check the "Cluster Analysis" tab for detailed statistics
4. **View Distributions**: Use the "Distribution Charts" tab to understand cluster characteristics
5. **Export Data**: Go to "Data Table" tab and download the segmented customers

## 💡 Tips

- Start with k=4 as a baseline
- Use the Elbow & Silhouette analysis to find the optimal k
- Lower Recency = Better (recent customers)
- Higher Frequency = Better (loyal customers)
- Higher Monetary = Better (high-value customers)

## 📝 Cluster Interpretation Guide

Based on RFM characteristics:

- **Champions**: Low Recency, High Frequency, High Monetary
- **Loyal Customers**: High Frequency, High Monetary
- **Big Spenders**: Low Recency, High Monetary
- **At Risk**: High Recency (haven't purchased recently)
- **Low Value**: Low Frequency, Low Monetary

## 🔧 Troubleshooting

If you encounter any issues:

1. Make sure all packages are installed:
   ```bash
   pip install streamlit pandas numpy plotly scikit-learn openpyxl
   ```

2. Ensure the data file exists at: `data/Online Retail.xlsx`

3. Restart the Streamlit server if visualizations don't update

## 📦 Required Packages

- streamlit
- pandas
- numpy
- plotly
- scikit-learn
- openpyxl

All packages are already installed in your Python environment!
