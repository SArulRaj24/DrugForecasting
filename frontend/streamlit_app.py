# streamlit_app.py - Updated Frontend with Backend Integration
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import io
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="Medicine Time Series Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
API_BASE_URL = "http://localhost:8000"  # Change this to your backend URL

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTab {
        font-size: 1.2rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# API Helper Functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_categories():
    """Get available medicine categories"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/categories", timeout=10)
        if response.status_code == 200:
            return response.json()["categories"]
        return []
    except Exception as e:
        st.error(f"Error fetching categories: {str(e)}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data_from_api(start_date=None, end_date=None, categories=None):
    """Load data from API"""
    try:
        params = {}
        if start_date:
            params['start_date'] = start_date
        if end_date:
            params['end_date'] = end_date
        if categories:
            params['categories'] = ','.join(categories)
        
        response = requests.get(f"{API_BASE_URL}/api/data", params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            if not df.empty and 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df, data.get('summary_stats', {}), data.get('categories', [])
        elif response.status_code == 404:
            return pd.DataFrame(), {}, []
        else:
            st.error(f"Error loading data: {response.json().get('detail', 'Unknown error')}")
            return pd.DataFrame(), {}, []
            
    except Exception as e:
        st.error(f"Error connecting to API: {str(e)}")
        return pd.DataFrame(), {}, []

def upload_file_to_api(file):
    """Upload file to API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/api/upload", files=files, timeout=60)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"detail": str(e)}

def get_predictions_from_api(category, periods, confidence_level=0.95):
    """Get Forecast from API"""
    try:
        data = {
            "category": category,
            "periods": periods,
            "confidence_level": confidence_level
        }
        response = requests.post(f"{API_BASE_URL}/api/predict", json=data, timeout=120)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            error_msg = response.json().get('detail', 'Unknown error')
            return False, {"error": error_msg}
            
    except Exception as e:
        return False, {"error": str(e)}

def get_category_stats(category):
    """Get detailed statistics for a category"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/stats/{category}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {}
    except Exception as e:
        st.error(f"Error fetching category stats: {str(e)}")
        return {}

# Main application
def main():
    st.markdown('<h1 class="main-header">💊 Medicine Time Series Analysis & Forecasting</h1>', unsafe_allow_html=True)

    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("⚠️ Backend API is not accessible. Please ensure the backend server is running.")
        st.info("To start the backend server, run: `uvicorn main:app --reload` in the backend directory")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Control Panel")
        
        # API Status
        st.subheader("🔌 API Status")
        if api_healthy:
            st.success("✅ Backend Connected")
            if 'database' in health_data:
                db_status = "✅ Connected" if health_data['database'] == 'connected' else "❌ Disconnected"
                st.write(f"Database: {db_status}")
        else:
            st.error("❌ Backend Disconnected")
        
        st.divider()
        
        # File upload section
        st.subheader("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload new medicine sales data",
            type=['csv', 'xlsx'],
            help="Upload CSV or Excel file with date and medicine category columns"
        )
        
        if uploaded_file is not None:
            st.info(f"📄 File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            if st.button("💾 Upload to Database", type="primary"):
                with st.spinner("Uploading file..."):
                    success, result = upload_file_to_api(uploaded_file)
                    
                    if success:
                        st.success(f"✅ {result['message']}")
                        st.cache_data.clear()  # Clear cache to reload data
                        st.rerun()
                    else:
                        st.error(f"❌ Upload failed: {result.get('detail', 'Unknown error')}")
        
        st.divider()
        
        # Data refresh
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Date range filter
        st.subheader("📅 Date Range")
        use_date_filter = st.checkbox("Filter by date range")
        
        start_date = None
        end_date = None
        
        if use_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=datetime(2014, 1, 1))
            with col2:
                end_date = st.date_input("End Date", value=datetime(2017, 12, 31))

            if start_date:
                start_date = start_date.strftime('%Y-%m-%d')
            if end_date:
                end_date = end_date.strftime('%Y-%m-%d')

    # Load data with filters
    df, summary_stats, available_categories = load_data_from_api(start_date, end_date)
    
    if df.empty:
        st.warning("📭 No data available. Please upload some medicine sales data to get started.")
        
        # Show sample data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            **CSV/Excel file should contain:**
            - `date` column (YYYY-MM-DD format)
            - Medicine category columns (e.g., Antibiotics, Painkillers, Vitamins)
            - Quantity values (non-negative numbers)
            
            **Example:**
            ```
            date,Antibiotics,Painkillers,Vitamins
            2023-01-01,120,80,150
            2023-02-01,110,85,160
            2023-03-01,130,75,140
            ```
            """)
        return
    
    medicine_categories = [col for col in df.columns if col != 'date']
    
    # Main tabs
    tab1, tab2 = st.tabs(["Analysis", "Forecast"])
    
    with tab1:
        st.header("📈 Time Series Analysis")
        
        # Key metrics
        if not df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_sales = df[medicine_categories].sum().sum()
                st.metric("Total Sales (All Time)", f"{total_sales:,.0f}")
            
            with col2:
                avg_monthly = df[medicine_categories].mean().mean()
                st.metric("Avg Monthly Sales", f"{avg_monthly:.0f}")
            
            with col3:
                best_category = df[medicine_categories].sum().idxmax()
                st.metric("Top Category", best_category)
            
            with col4:
                latest_month_total = df[medicine_categories].iloc[-1].sum()
                prev_month_total = df[medicine_categories].iloc[-2].sum() if len(df) > 1 else latest_month_total
                delta = latest_month_total - prev_month_total
                st.metric("Latest Month Total", f"{latest_month_total:.0f}", delta=f"{delta:+.0f}")
        
        st.divider()
        
        # Category selection for detailed analysis
        col1, col2 = st.columns([1, 3])
        
        with col1:
            selected_categories = st.multiselect(
                "Select Medicine Categories",
                medicine_categories,
                default=medicine_categories[:3] if len(medicine_categories) >= 3 else medicine_categories,
                help="Choose categories to analyze"
            )
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis Type",
                ["Time Series", "Distribution", "Correlation", "Seasonality", "Trend Analysis"],
                help="Choose the type of analysis to perform"
            )
        
        if selected_categories:
            if analysis_type == "Time Series":
                # Time series plot
                fig = go.Figure()
                
                for category in selected_categories:
                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df[category],
                        mode='lines+markers',
                        name=category,
                        line=dict(width=2),
                        hovertemplate='<b>%{fullData.name}</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Quantity: %{y:,.0f}<br>' +
                                    '<extra></extra>'
                    ))
                
                fig.update_layout(
                    title="Medicine Sales Time Series",
                    xaxis_title="Date",
                    yaxis_title="Quantity Sold",
                    height=500,
                    hovermode='x unified',
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Distribution":
                # Distribution analysis
                fig = make_subplots(
                    rows=1, cols=len(selected_categories),
                    subplot_titles=selected_categories,
                    specs=[[{"secondary_y": False}] * len(selected_categories)]
                )
                
                for i, category in enumerate(selected_categories):
                    fig.add_trace(
                        go.Histogram(
                            x=df[category], 
                            name=category, 
                            nbinsx=20,
                            showlegend=False
                        ),
                        row=1, col=i+1
                    )
                
                fig.update_layout(
                    title="Distribution of Medicine Sales",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Correlation":
                # Correlation analysis
                if len(selected_categories) > 1:
                    corr_data = df[selected_categories].corr()
                    
                    fig = px.imshow(
                        corr_data,
                        title="Correlation Matrix of Medicine Categories",
                        color_continuous_scale="RdBu_r",
                        text_auto=True,
                        aspect="auto"
                    )
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 categories for correlation analysis.")
            
            elif analysis_type == "Seasonality":
                # Seasonality analysis
                df_seasonal = df.copy()
                df_seasonal['month'] = df_seasonal['date'].dt.month
                df_seasonal['year'] = df_seasonal['date'].dt.year
                
                monthly_avg = df_seasonal.groupby('month')[selected_categories].mean()
                
                fig = go.Figure()
                
                for category in selected_categories:
                    fig.add_trace(go.Scatter(
                        x=list(range(1, 13)),
                        y=monthly_avg[category],
                        mode='lines+markers',
                        name=category,
                        line=dict(width=3),
                        marker=dict(size=8)
                    ))
                
                fig.update_layout(
                    title="Seasonal Patterns (Monthly Averages)",
                    xaxis_title="Month",
                    yaxis_title="Average Quantity Sold",
                    height=400,
                    xaxis=dict(
                        tickmode='array', 
                        tickvals=list(range(1, 13)),
                        ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_type == "Trend Analysis":
                # Trend analysis with moving averages
                fig = go.Figure()
                
                for category in selected_categories:

                    fig.add_trace(go.Scatter(
                        x=df['date'],
                        y=df[category],
                        mode='lines',
                        name=f'{category}',
                        line=dict(width=1),
                        opacity=1,
                    ))
                    
                    # # 3-month moving average
                    # ma_3 = df[category].rolling(window=3, center=True).mean()
                    # fig.add_trace(go.Scatter(
                    #     x=df['date'],
                    #     y=ma_3,
                    #     mode='lines',
                    #     name=f'{category} (3-Month MA)',
                    #     line=dict(width=2)
                    # ))
                
                fig.update_layout(
                    title="Trend Analysis",
                    xaxis_title="Date",
                    yaxis_title="Quantity Sold",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        if selected_categories and summary_stats:
            st.subheader("📊 Summary Statistics")
            
            # Create summary DataFrame
            stats_data = {}
            for category in selected_categories:
                if category in summary_stats:
                    stats = summary_stats[category]
                    stats_data[category] = {
                        'Mean': f"{stats.get('mean', 0):.2f}",
                        'Median': f"{stats.get('median', 0):.2f}",
                        'Std Dev': f"{stats.get('std', 0):.2f}",
                        'Min': f"{stats.get('min', 0):.0f}",
                        'Max': f"{stats.get('max', 0):.0f}",
                        'Total': f"{stats.get('total', 0):,.0f}",
                        'Count': f"{stats.get('count', 0):.0f}"
                    }
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data).T
                st.dataframe(stats_df, use_container_width=True)
        
        # Category-specific details
        if selected_categories:
            st.subheader("🔍 Category Details")
            
            selected_category_detail = st.selectbox(
                "Select category for detailed view:",
                selected_categories,
                key="detail_category"
            )
            
            if selected_category_detail:
                # Get detailed stats from API
                detailed_stats = get_category_stats(selected_category_detail)
                
                if detailed_stats:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Record Count", detailed_stats.get('record_count', 'N/A'))
                        st.metric("Start Date", detailed_stats.get('start_date', 'N/A'))
                    
                    with col2:
                        st.metric("Average Quantity", f"{detailed_stats.get('avg_quantity', 0):.2f}")
                        st.metric("End Date", detailed_stats.get('end_date', 'N/A'))
                    
                    with col3:
                        st.metric("Total Quantity", f"{detailed_stats.get('total_quantity', 0):,.2f}")
                        min_max = f"{detailed_stats.get('min_quantity', 0):,.2f} - {detailed_stats.get('max_quantity', 0):,.2f}"
                        st.metric("Range (Min-Max)", min_max)
    
    with tab2:
        st.header("🔮 Sales Forecast")
        
        if not available_categories:
            st.warning("No data available for Forecast. Please upload data first.")
            return
        
        # Prediction controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_category = st.selectbox(
                "Select Category for Forecast",
                available_categories,
                help="Choose a medicine category to Forecast"
            )
        
        with col2:
            prediction_months = st.slider(
                "Forecast Period (Months)",
                min_value=1,
                max_value=24,
                value=12,
                help="Number of months to forecast"
            )
        
        with col3:
            confidence_level = 95 / 100
        # Model information expander
        with st.expander("ℹ️ About the ARIMA Model"):
            st.markdown("""
            **ARIMA (AutoRegressive Integrated Moving Average) Model Features:**
            
            - ✅ **Automatic Stationarity Testing**: Uses ADF and KPSS tests for stationarity detection
            - ✅ **Data Transformation**: Applies differencing and other transformations as needed
            - ✅ **Parameter Optimization**: Automatically selects optimal ARIMA parameters (p,d,q) using AIC
            - ✅ **Model Validation**: Performs residual analysis and calculates performance metrics
            - ✅ **Seasonality Detection**: Identifies and models seasonal patterns in the data
            - ✅ **Real-time Processing**: Models are retrained when new data is uploaded
            
            **Performance Metrics:**
            - **MAME**: Mean Absolute Error(lower is better)
            - **RMSE**: Root Mean Square Error (lower is better)
            """)
        
        # Generate Forecast button
        if st.button("🚀 Generate Forecast", type="primary"):
            with st.spinner("Running ARIMA model analysis... Please wait"):
                success, result = get_predictions_from_api(pred_category, prediction_months, confidence_level)
                
                if success:
                    # Extract prediction data
                    predictions = result['predictions']
                    dates = result['dates']
                    metrics = result['metrics']
                    
                    # Create prediction visualization
                    fig = go.Figure()
                    
                    # Historical data
                    historical_data = df[df[pred_category].notna()]
                    fig.add_trace(go.Scatter(
                        x=historical_data['date'],
                        y=historical_data[pred_category],
                        mode='lines',
                        name='Historical Data',
                        line=dict(color='blue', width=2),
                        hovertemplate='<b>Historical</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Quantity: %{y:,.0f}<br>' +
                                    '<extra></extra>'
                    ))
                    
                    # Predictions
                    pred_dates = pd.to_datetime(dates)
                    fig.add_trace(go.Scatter(
                        x=pred_dates,
                        y=predictions,
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=6),
                        hovertemplate='<b>Forecast</b><br>' +
                                    'Date: %{x}<br>' +
                                    'Quantity: %{y:,.0f}<br>' +
                                    '<extra></extra>'
                    ))
                    
                    fig.update_layout(
                        title=f"Sales Prediction for {pred_category}",
                        xaxis_title="Date",
                        yaxis_title="Quantity Sold",
                        height=600,
                        hovermode='x unified',
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Prediction summary and metrics
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📋 Forecast Summary")
                        avg_prediction = np.mean(predictions)
                        total_prediction = np.sum(predictions)
                        
                        st.metric("Average Monthly Prediction", f"{avg_prediction:.0f}")
                        st.metric(f"Total {prediction_months}-Month Prediction", f"{total_prediction:.0f}")
                        
                        # Growth analysis
                        if not df.empty:
                            historical_avg = df[pred_category].tail(12).mean()
                            growth_rate = ((avg_prediction - historical_avg) / historical_avg) * 100 if historical_avg > 0 else 0
                            predicted_nextmonth_sales = predictions[0]
                            sales_diff_predicated = predictions[0] - historical_avg

                            st.metric("Historical 12-Month Average", f"{historical_avg:.0f}")
                            st.metric("Predicted Growth Rate", f"{growth_rate:+.1f}%")
                            st.metric("Predicted Next Month",f"{predicted_nextmonth_sales:.0f}",delta=f"{sales_diff_predicated:+.0f}"
)
                    with col2:
                        st.subheader("🎯 Model Performance")

                        # ARIMA order (taken directly from result, not metrics)
                        arima_order = result.get('arima_order', [0, 0, 0])
                        st.info(f"**ARIMA Order**: ({arima_order[0]}, {arima_order[1]}, {arima_order[2]})")

                        # Performance metrics
                        mae = metrics.get('mae', 0)
                        rmse = metrics.get('rmse', 0)

                        col2_1, col2_2 = st.columns(2)
                        with col2_1:
                            st.metric("MAE", f"{mae:.2f}")

                        with col2_2:
                            st.metric("RMSE", f"{rmse:.2f}")

                        
                        # Model validation
                        model_valid = metrics.get('model_valid', True)
                        validation_status = "✅ Valid" if model_valid else "⚠️ Check residuals"
                        st.info(f"**Model Validation**: {validation_status}")
                    
                    # Detailed predictions table
                    with st.expander("📊 Detailed Forecast"):
                        pred_df = pd.DataFrame({
                            'Date': dates,
                            'Predicted Quantity': [f"{p:.2f}" for p in predictions],
                        })
                        st.dataframe(pred_df, use_container_width=True)
                    
                    # Download predictions
                    csv_data = pd.DataFrame({
                        'date': dates,
                        'predicted_quantity': predictions,
                        
                    })
                    
                    csv_string = csv_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Forecast (CSV)",
                        data=csv_string,
                        file_name=f"{pred_category}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
                else:
                    error_msg = result.get('error', 'Unknown error occurred')
                    st.error(f"❌ Error generating Forecast: {error_msg}")
                    
                    # Provide helpful suggestions based on common errors
                    if "insufficient" in error_msg.lower():
                        st.info("💡 **Tip**: ARIMA models require at least 24 data points. Please upload more historical data.")
                    elif "category" in error_msg.lower():
                        st.info("💡 **Tip**: Make sure the selected category exists in your uploaded data.")
                    elif "stationary" in error_msg.lower():
                        st.info("💡 **Tip**: The data might have complex patterns. Try uploading more recent data or check for outliers.")

if __name__ == "__main__":
    main()