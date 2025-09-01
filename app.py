import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Emission Hotspot Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E8B57;
    }
    .kpi-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('comprehensive_pollution_vehicle_dataset.csv')
        
        # Create hotspot indicator if not present
        if 'Hotspot' not in df.columns:
            hotspot_threshold = df['Pollution_Impact_Score'].quantile(0.7)
            df['Hotspot'] = (df['Pollution_Impact_Score'] >= hotspot_threshold).astype(int)
        
        return df
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure 'comprehensive_pollution_vehicle_dataset.csv' is in the working directory.")
        return None

@st.cache_resource
def load_model_artifacts():
    """Load ML model and preprocessing artifacts"""
    artifacts = {}
    model_files = {
        'model': 'models/emission_hotspot_model.pkl',
        'scaler': 'models/feature_scaler.pkl',
        'vehicle_make_encoder': 'models/vehicle_make_encoder.pkl',
        'vehicle_class_encoder': 'models/vehicle_class_encoder.pkl',
        'economic_region_encoder': 'models/economic_region_encoder.pkl'
    }
    
    for name, filepath in model_files.items():
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    artifacts[name] = pickle.load(f)
            except Exception as e:
                st.warning(f"Could not load {name}: {e}")
        else:
            st.warning(f"Model file not found: {filepath}")
    
    return artifacts

def create_kpi_cards(df):
    """Create KPI metric cards"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            label="Total Records",
            value=f"{len(df):,}",
            help="Total number of data points in the dataset"
        )
    
    with col2:
        avg_pm25 = df['PM25_Concentration'].mean()
        st.metric(
            label="Avg PM2.5 (μg/m³)",
            value=f"{avg_pm25:.1f}",
            help="Average PM2.5 concentration across all regions"
        )
    
    with col3:
        avg_co2 = df['Avg_CO2_Emissions'].mean()
        st.metric(
            label="Avg CO₂ (g/km)",
            value=f"{avg_co2:.1f}",
            help="Average CO2 emissions per kilometer"
        )
    
    with col4:
        hotspot_rate = (df['Hotspot'].sum() / len(df)) * 100 if 'Hotspot' in df.columns else 0
        st.metric(
            label="Hotspot Rate (%)",
            value=f"{hotspot_rate:.1f}%",
            help="Percentage of records classified as emission hotspots"
        )
    
    with col5:
        unique_countries = df['Country'].nunique()
        st.metric(
            label="Countries Covered",
            value=f"{unique_countries}",
            help="Number of unique countries in the dataset"
        )

def create_sidebar_filters(df):
    """Create sidebar filters"""
    st.sidebar.header("🔍 Filters")
    
    # Year range filter
    year_range = st.sidebar.slider(
        "Year Range",
        min_value=int(df['Year'].min()),
        max_value=int(df['Year'].max()),
        value=(int(df['Year'].min()), int(df['Year'].max())),
        help="Select the range of years to analyze"
    )
    
    # Region filter
    regions = ['All'] + sorted(df['Region'].unique().tolist())
    selected_region = st.sidebar.selectbox("Region", regions)
    
    # Economic Region filter
    econ_regions = ['All'] + sorted(df['Economic_Region'].unique().tolist())
    selected_econ_region = st.sidebar.selectbox("Economic Region", econ_regions)
    
    # Vehicle Make filter
    makes = ['All'] + sorted(df['Vehicle_Make'].unique().tolist())
    selected_makes = st.sidebar.multiselect("Vehicle Make", makes, default=['All'])
    
    # Vehicle Class filter
    classes = ['All'] + sorted(df['Vehicle_Class'].unique().tolist())
    selected_classes = st.sidebar.multiselect("Vehicle Class", classes, default=['All'])
    
    # PM2.5 concentration range
    pm25_range = st.sidebar.slider(
        "PM2.5 Concentration Range (μg/m³)",
        min_value=float(df['PM25_Concentration'].min()),
        max_value=float(df['PM25_Concentration'].max()),
        value=(float(df['PM25_Concentration'].min()), float(df['PM25_Concentration'].max())),
        help="Filter by PM2.5 concentration levels"
    )
    
    # CO2 emissions range
    co2_range = st.sidebar.slider(
        "CO₂ Emissions Range (g/km)",
        min_value=float(df['Avg_CO2_Emissions'].min()),
        max_value=float(df['Avg_CO2_Emissions'].max()),
        value=(float(df['Avg_CO2_Emissions'].min()), float(df['Avg_CO2_Emissions'].max())),
        help="Filter by CO2 emission levels"
    )
    
    return {
        'year_range': year_range,
        'region': selected_region,
        'economic_region': selected_econ_region,
        'makes': selected_makes,
        'classes': selected_classes,
        'pm25_range': pm25_range,
        'co2_range': co2_range
    }

def apply_filters(df, filters):
    """Apply selected filters to the dataframe"""
    filtered_df = df.copy()
    
    # Year filter
    filtered_df = filtered_df[
        (filtered_df['Year'] >= filters['year_range'][0]) &
        (filtered_df['Year'] <= filters['year_range'][1])
    ]
    
    # Region filter
    if filters['region'] != 'All':
        filtered_df = filtered_df[filtered_df['Region'] == filters['region']]
    
    # Economic region filter
    if filters['economic_region'] != 'All':
        filtered_df = filtered_df[filtered_df['Economic_Region'] == filters['economic_region']]
    
    # Vehicle make filter
    if 'All' not in filters['makes'] and filters['makes']:
        filtered_df = filtered_df[filtered_df['Vehicle_Make'].isin(filters['makes'])]
    
    # Vehicle class filter
    if 'All' not in filters['classes'] and filters['classes']:
        filtered_df = filtered_df[filtered_df['Vehicle_Class'].isin(filters['classes'])]
    
    # PM2.5 filter
    filtered_df = filtered_df[
        (filtered_df['PM25_Concentration'] >= filters['pm25_range'][0]) &
        (filtered_df['PM25_Concentration'] <= filters['pm25_range'][1])
    ]
    
    # CO2 filter
    filtered_df = filtered_df[
        (filtered_df['Avg_CO2_Emissions'] >= filters['co2_range'][0]) &
        (filtered_df['Avg_CO2_Emissions'] <= filters['co2_range'][1])
    ]
    
    return filtered_df

def create_overview_charts(df):
    """Create overview charts"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 PM2.5 Concentration by Region")
        region_pm25 = df.groupby('Region')['PM25_Concentration'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=region_pm25.values,
            y=region_pm25.index,
            orientation='h',
            title="Average PM2.5 Concentration by Region",
            labels={'x': 'PM2.5 Concentration (μg/m³)', 'y': 'Region'},
            color=region_pm25.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚗 CO₂ Emissions by Vehicle Class")
        class_co2 = df.groupby('Vehicle_Class')['Avg_CO2_Emissions'].mean().sort_values(ascending=False)
        
        fig = px.bar(
            x=class_co2.index,
            y=class_co2.values,
            title="Average CO₂ Emissions by Vehicle Class",
            labels={'x': 'Vehicle Class', 'y': 'CO₂ Emissions (g/km)'},
            color=class_co2.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def create_time_series_analysis(df):
    """Create time series analysis"""
    st.subheader("📈 Temporal Analysis")
    
    # Time series of pollution metrics
    yearly_data = df.groupby('Year').agg({
        'PM25_Concentration': 'mean',
        'Avg_CO2_Emissions': 'mean',
        'Pollution_Impact_Score': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PM2.5 Concentration Over Time', 'CO₂ Emissions Over Time', 
                       'Pollution Impact Score Over Time', 'Hotspot Rate Over Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # PM2.5 over time
    fig.add_trace(
        go.Scatter(x=yearly_data['Year'], y=yearly_data['PM25_Concentration'],
                  mode='lines+markers', name='PM2.5', line=dict(color='red')),
        row=1, col=1
    )
    
    # CO2 over time
    fig.add_trace(
        go.Scatter(x=yearly_data['Year'], y=yearly_data['Avg_CO2_Emissions'],
                  mode='lines+markers', name='CO₂', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Pollution score over time
    fig.add_trace(
        go.Scatter(x=yearly_data['Year'], y=yearly_data['Pollution_Impact_Score'],
                  mode='lines+markers', name='Impact Score', line=dict(color='green')),
        row=2, col=1
    )
    
    # Hotspot rate over time
    if 'Hotspot' in df.columns:
        hotspot_rate = df.groupby('Year')['Hotspot'].mean() * 100
        fig.add_trace(
            go.Scatter(x=hotspot_rate.index, y=hotspot_rate.values,
                      mode='lines+markers', name='Hotspot Rate', line=dict(color='orange')),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    st.subheader("🔥 Correlation Analysis")
    
    # Select numeric columns for correlation
    numeric_cols = ['PM25_Concentration', 'Avg_CO2_Emissions', 'Avg_Fuel_Consumption', 
                   'Avg_Engine_Size', 'Estimated_Market_Share', 'Pollution_Impact_Score']
    
    if 'Hotspot' in df.columns:
        numeric_cols.append('Hotspot')
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Key Variables",
        color_continuous_scale='RdBu_r'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def create_geographic_analysis(df):
    """Create geographic analysis"""
    st.subheader("🗺️ Geographic Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Country-wise pollution analysis
        country_stats = df.groupby('Country').agg({
            'PM25_Concentration': 'mean',
            'Avg_CO2_Emissions': 'mean',
            'Pollution_Impact_Score': 'mean'
        }).reset_index()
        
        fig = px.scatter(
            country_stats,
            x='Avg_CO2_Emissions',
            y='PM25_Concentration',
            size='Pollution_Impact_Score',
            hover_name='Country',
            title="Country-wise: CO₂ vs PM2.5",
            labels={'Avg_CO2_Emissions': 'CO₂ Emissions (g/km)', 
                   'PM25_Concentration': 'PM2.5 (μg/m³)'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Economic region analysis
        econ_region_stats = df.groupby('Economic_Region').agg({
            'PM25_Concentration': 'mean',
            'Avg_CO2_Emissions': 'mean',
            'Pollution_Impact_Score': 'mean'
        }).sort_values('Pollution_Impact_Score', ascending=False)
        
        fig = px.bar(
            x=econ_region_stats.index,
            y=econ_region_stats['Pollution_Impact_Score'],
            title="Pollution Impact Score by Economic Region",
            labels={'x': 'Economic Region', 'y': 'Pollution Impact Score'},
            color=econ_region_stats['Pollution_Impact_Score'],
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def create_vehicle_analysis(df):
    """Create vehicle-specific analysis"""
    st.subheader("🚙 Vehicle Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Market share by vehicle make
        make_share = df.groupby('Vehicle_Make')['Estimated_Market_Share'].sum().sort_values(ascending=False).head(10)
        
        fig = px.pie(
            values=make_share.values,
            names=make_share.index,
            title="Top 10 Vehicle Makes by Market Share"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Engine size distribution
        fig = px.histogram(
            df,
            x='Avg_Engine_Size',
            nbins=30,
            title="Distribution of Engine Sizes",
            labels={'x': 'Engine Size (L)', 'y': 'Frequency'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_prediction_playground(artifacts, df):
    """Create ML prediction playground"""
    st.subheader("🔮 Prediction Playground")
    
    if not artifacts.get('model'):
        st.warning("ML model not found. Please ensure model artifacts are available in the models/ directory.")
        return
    
    st.write("Use the trained ML model to predict emission hotspots based on input parameters.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Input Parameters:**")
        
        # Input fields for prediction
        pm25 = st.number_input("PM2.5 Concentration (μg/m³)", 
                              min_value=0.0, max_value=100.0, value=15.0, step=0.1)
        co2 = st.number_input("CO₂ Emissions (g/km)", 
                             min_value=0.0, max_value=500.0, value=200.0, step=1.0)
        fuel_consumption = st.number_input("Fuel Consumption (L/100km)", 
                                         min_value=0.0, max_value=30.0, value=8.0, step=0.1)
        engine_size = st.number_input("Engine Size (L)", 
                                    min_value=0.0, max_value=8.0, value=2.0, step=0.1)
        market_share = st.number_input("Market Share", 
                                     min_value=0.0, max_value=1.0, value=0.1, step=0.01)
        
        # Categorical inputs
        if 'vehicle_make_encoder' in artifacts:
            vehicle_makes = list(artifacts['vehicle_make_encoder'].classes_)
            selected_make = st.selectbox("Vehicle Make", vehicle_makes)
        else:
            selected_make = st.text_input("Vehicle Make", value="TOYOTA")
        
        if 'vehicle_class_encoder' in artifacts:
            vehicle_classes = list(artifacts['vehicle_class_encoder'].classes_)
            selected_class = st.selectbox("Vehicle Class", vehicle_classes)
        else:
            selected_class = st.text_input("Vehicle Class", value="COMPACT")
        
        if 'economic_region_encoder' in artifacts:
            economic_regions = list(artifacts['economic_region_encoder'].classes_)
            selected_econ_region = st.selectbox("Economic Region", economic_regions)
        else:
            selected_econ_region = st.text_input("Economic Region", value="North_America")
    
    with col2:
        st.write("**Prediction Result:**")
        
        if st.button("🎯 Predict Hotspot", type="primary"):
            try:
                # Prepare input data
                input_data = pd.DataFrame({
                    'PM25_Concentration': [pm25],
                    'Avg_CO2_Emissions': [co2],
                    'Avg_Fuel_Consumption': [fuel_consumption],
                    'Avg_Engine_Size': [engine_size],
                    'Estimated_Market_Share': [market_share]
                })
                
                # Encode categorical variables
                if 'vehicle_make_encoder' in artifacts:
                    try:
                        make_encoded = artifacts['vehicle_make_encoder'].transform([selected_make])[0]
                        input_data['Vehicle_Make_Encoded'] = make_encoded
                    except:
                        input_data['Vehicle_Make_Encoded'] = 0
                
                if 'vehicle_class_encoder' in artifacts:
                    try:
                        class_encoded = artifacts['vehicle_class_encoder'].transform([selected_class])[0]
                        input_data['Vehicle_Class_Encoded'] = class_encoded
                    except:
                        input_data['Vehicle_Class_Encoded'] = 0
                
                if 'economic_region_encoder' in artifacts:
                    try:
                        region_encoded = artifacts['economic_region_encoder'].transform([selected_econ_region])[0]
                        input_data['Economic_Region_Encoded'] = region_encoded
                    except:
                        input_data['Economic_Region_Encoded'] = 0
                
                # Scale features if scaler is available
                if 'scaler' in artifacts:
                    feature_cols = []
                    scaler = artifacts['scaler']
                    
                    # Use scaler's feature names if available
                    if hasattr(scaler, 'feature_names_in_'):
                        feature_cols = scaler.feature_names_in_
                    else:
                        # Fallback to common features
                        feature_cols = ['PM25_Concentration', 'Avg_CO2_Emissions', 'Avg_Fuel_Consumption', 
                                      'Avg_Engine_Size', 'Estimated_Market_Share']
                    
                    # Select only available columns
                    available_cols = [col for col in feature_cols if col in input_data.columns]
                    input_scaled = scaler.transform(input_data[available_cols])
                else:
                    input_scaled = input_data.values
                
                # Make prediction
                prediction = artifacts['model'].predict(input_scaled)[0]
                probability = artifacts['model'].predict_proba(input_scaled)[0]
                
                # Display results
                if prediction == 1:
                    st.error("🔴 **EMISSION HOTSPOT DETECTED**")
                    st.write(f"Hotspot Probability: {probability[1]:.2%}")
                else:
                    st.success("🟢 **Normal Emission Level**")
                    st.write(f"Hotspot Probability: {probability[1]:.2%}")
                
                # Show probability breakdown
                st.write("**Probability Breakdown:**")
                st.write(f"- Normal: {probability[0]:.2%}")
                st.write(f"- Hotspot: {probability[1]:.2%}")
                
                # Add recommendation
                if probability[1] > 0.7:
                    st.warning("⚠️ High emission risk detected. Consider implementing emission reduction measures.")
                elif probability[1] > 0.3:
                    st.info("ℹ️ Moderate emission risk. Monitor closely.")
                else:
                    st.success("✅ Low emission risk. Good environmental performance.")
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                st.write("Please check that all model artifacts are properly saved and compatible.")

def create_data_explorer(df):
    """Create data exploration section"""
    st.subheader("📊 Data Explorer")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write(f"**Dataset Overview:** {len(df):,} records")
        
        # Display sample data
        st.write("**Sample Data:**")
        st.dataframe(df.head(100), use_container_width=True, height=400)
    
    with col2:
        st.write("**Quick Stats:**")
        
        # Dataset statistics
        stats_df = df.describe()
        st.dataframe(stats_df, use_container_width=True)
        
        # Download filtered data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_emissions_data.csv",
            mime="text/csv",
            help="Download the currently filtered dataset as CSV"
        )

def main():
    """Main application function"""
    # Header
    st.markdown('<h1 class="main-header">🌍 Emission Hotspot Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data and models
    df = load_data()
    if df is None:
        return
    
    artifacts = load_model_artifacts()
    
    # Sidebar filters
    filters = create_sidebar_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Show filter results
    st.sidebar.markdown("---")
    st.sidebar.write(f"**Records after filtering:** {len(filtered_df):,}")
    st.sidebar.write(f"**Original records:** {len(df):,}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Overview", "🌍 Geographic", "🚗 Vehicle Analysis", 
        "🔮 Predictions", "📊 Data Explorer", "ℹ️ About"
    ])
    
    with tab1:
        # KPI Cards
        create_kpi_cards(filtered_df)
        st.markdown("---")
        
        # Overview charts
        create_overview_charts(filtered_df)
        
        # Time series analysis
        create_time_series_analysis(filtered_df)
        
        # Correlation analysis
        create_correlation_heatmap(filtered_df)
    
    with tab2:
        create_geographic_analysis(filtered_df)
    
    with tab3:
        create_vehicle_analysis(filtered_df)
    
    with tab4:
        create_prediction_playground(artifacts, filtered_df)
    
    with tab5:
        create_data_explorer(filtered_df)
    
    with tab6:
        st.markdown("""
        ## About This Dashboard
        
        This dashboard provides comprehensive analysis of emission hotspots using merged pollution and vehicle data.
        
        ### Key Features:
        - **Interactive KPIs**: Real-time metrics with filtering
        - **Geographic Analysis**: Regional pollution patterns and trends
        - **Vehicle Analysis**: Make, class, and engine size correlations
        - **ML Predictions**: Trained model for hotspot classification
        - **Data Export**: Download filtered datasets
        
        ### Data Sources:
        - Air pollution data (PM2.5 concentrations)
        - Vehicle emissions data (CO₂, fuel consumption)
        - Market share and regional information
        
        ### Model Information:
        The prediction model uses:
        - Random Forest Classifier
        - Feature scaling and encoding
        - 70th percentile threshold for hotspot classification
        
        ### Usage Tips:
        1. Use sidebar filters to focus on specific regions or time periods
        2. Hover over charts for detailed information
        3. Use the prediction playground to test scenarios
        4. Download filtered data for further analysis
        
        ---
        *Built with Streamlit, Plotly, and scikit-learn*
        """)

if __name__ == "__main__":
    main() 