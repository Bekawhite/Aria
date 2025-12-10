"""
MALARIA FORECASTING WEB APP WITH NDDI AND MULTI-INSTANCE LEARNING
A complete Streamlit application for malaria prediction
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from scipy.spatial.distance import mahalanobis
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Malaria Forecasting System with NDDI & MIL",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E40AF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #3B82F6;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #10B981;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #F59E0B;
    }
    .nddi-box {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0EA5E9;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data' not in st.session_state:
    st.session_state.data = None
if 'sir_results' not in st.session_state:
    st.session_state.sir_results = None
if 'hybrid_results' not in st.session_state:
    st.session_state.hybrid_results = None
if 'nddi_features' not in st.session_state:
    st.session_state.nddi_features = None
if 'mil_predictions' not in st.session_state:
    st.session_state.mil_predictions = None

# ============================================================================
# 1. NDDI CALCULATION FUNCTIONS
# ============================================================================
class NDDICalculator:
    """Calculate Normalized Difference Drought Index"""
    
    @staticmethod
    def calculate_nddi(rainfall_data, temperature_data, humidity_data, ndvi_data=None):
        """
        Calculate NDDI using multiple climate indicators
        NDDI = (Rainfall - Temperature) / (Rainfall + Temperature) * Humidity_factor
        """
        # Normalize inputs
        rainfall_norm = (rainfall_data - np.mean(rainfall_data)) / np.std(rainfall_data)
        temp_norm = (temperature_data - np.mean(temperature_data)) / np.std(temperature_data)
        humidity_norm = (humidity_data - np.mean(humidity_data)) / np.std(humidity_data)
        
        # Calculate base NDDI
        with np.errstate(divide='ignore', invalid='ignore'):
            nddi = (rainfall_norm - temp_norm) / (rainfall_norm + temp_norm + 1e-10)
        
        # Apply humidity correction
        humidity_factor = 1 + 0.5 * humidity_norm
        nddi = nddi * humidity_factor
        
        # Clip extreme values
        nddi = np.clip(nddi, -2, 2)
        
        # Handle NaN values
        nddi = np.nan_to_num(nddi, nan=0.0)
        
        return nddi
    
    @staticmethod
    def classify_drought_risk(nddi_values):
        """Classify drought risk based on NDDI values"""
        risk_levels = []
        for nddi in nddi_values:
            if nddi < -1.0:
                risk_levels.append("Severe Drought")
            elif nddi < -0.5:
                risk_levels.append("Moderate Drought")
            elif nddi < 0:
                risk_levels.append("Mild Drought")
            elif nddi < 0.5:
                risk_levels.append("Normal")
            elif nddi < 1.0:
                risk_levels.append("Moist")
            else:
                risk_levels.append("Very Moist")
        return risk_levels
    
    @staticmethod
    def calculate_nddi_impact_factor(nddi_values):
        """Calculate impact factor on malaria transmission"""
        # NDDI < 0 (drought conditions) can increase malaria due to water pooling
        # NDDI > 0 (moist conditions) can also increase malaria
        # Optimal range is near 0 (balanced conditions)
        impact_factors = []
        for nddi in nddi_values:
            if nddi < -0.8:  # Severe drought
                impact = 1.5  # Increased risk due to water collection
            elif nddi < -0.3:  # Moderate drought
                impact = 1.2
            elif nddi < 0.3:  # Normal conditions
                impact = 1.0
            elif nddi < 0.8:  # Moist conditions
                impact = 1.3  # Increased risk
            else:  # Very moist
                impact = 1.6  # High risk
            impact_factors.append(impact)
        return np.array(impact_factors)

# ============================================================================
# 2. MULTI-INSTANCE LEARNING CLASS
# ============================================================================
class MultiInstanceMalariaPredictor:
    """Multi-Instance Learning for malaria prediction using temporal windows"""
    
    def __init__(self, window_size=3, n_clusters=5):
        self.window_size = window_size
        self.n_clusters = n_clusters
        self.cluster_model = None
        self.instance_predictors = {}
        self.scaler = StandardScaler()
        
    def create_instances(self, data, target_column='malaria_cases'):
        """Create instances from temporal windows"""
        instances = []
        instance_labels = []
        feature_columns = ['temperature', 'rainfall', 'humidity', 'nddi', 
                          'llin_coverage', 'irs_coverage', 'population']
        
        # Ensure we have all required columns
        available_features = [col for col in feature_columns if col in data.columns]
        
        for i in range(len(data) - self.window_size):
            # Create instance from window
            instance_features = []
            for j in range(self.window_size):
                idx = i + j
                for feature in available_features:
                    instance_features.append(data.iloc[idx][feature])
            
            # Add instance
            instances.append(instance_features)
            
            # Use future malaria cases as label (next month after window)
            future_idx = i + self.window_size
            if future_idx < len(data):
                instance_labels.append(data.iloc[future_idx][target_column])
            else:
                instance_labels.append(np.nan)
        
        instances = np.array(instances)
        instance_labels = np.array(instance_labels)
        
        # Remove instances with NaN labels
        valid_indices = ~np.isnan(instance_labels)
        return instances[valid_indices], instance_labels[valid_indices]
    
    def cluster_instances(self, instances):
        """Cluster similar instances using KMeans"""
        self.cluster_model = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.cluster_model.fit_predict(instances)
        return cluster_labels
    
    def train_instance_predictors(self, instances, labels, cluster_labels):
        """Train separate predictor for each cluster"""
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 10:  # Need enough samples
                # Scale features
                X_cluster = instances[cluster_indices]
                y_cluster = labels[cluster_indices]
                
                # Train Random Forest for this cluster
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_cluster, y_cluster)
                self.instance_predictors[cluster_id] = rf_model
    
    def predict(self, instances):
        """Predict using multi-instance approach"""
        if self.cluster_model is None:
            raise ValueError("Model must be trained before prediction")
        
        predictions = []
        for instance in instances:
            # Determine cluster
            cluster_id = self.cluster_model.predict(instance.reshape(1, -1))[0]
            
            # Use appropriate predictor
            if cluster_id in self.instance_predictors:
                pred = self.instance_predictors[cluster_id].predict(instance.reshape(1, -1))[0]
                predictions.append(max(0, pred))
            else:
                # Fallback to average
                predictions.append(np.mean([p.predict(instance.reshape(1, -1))[0] 
                                          for p in self.instance_predictors.values()]))
        
        return np.array(predictions)

# ============================================================================
# 3. ENHANCED DATA GENERATION WITH NDDI
# ============================================================================
def generate_synthetic_data():
    """Generate synthetic climate and malaria data with NDDI"""
    st.info("🔄 Generating synthetic data with NDDI features...")
    
    # Create date range
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='M')
    n_months = len(dates)
    
    # Generate climate data
    months = dates.month
    time_idx = np.arange(n_months)
    
    # Temperature
    base_temp = 25 + 5 * np.sin(2 * np.pi * time_idx / 12 - np.pi/2)
    temp_trend = 0.02 * time_idx / 12
    temperature = base_temp + temp_trend + np.random.normal(0, 1, n_months)
    temperature = np.clip(temperature, 20, 35)
    
    # Rainfall
    rainy_season1 = np.exp(-0.5 * ((months - 5) / 1.5)**2)
    rainy_season2 = np.exp(-0.5 * ((months - 11) / 1.5)**2)
    base_rainfall = 50 + 150 * (rainy_season1 + 0.8 * rainy_season2)
    rainfall = base_rainfall + np.random.normal(0, 20, n_months)
    rainfall = np.clip(rainfall, 0, 300)
    
    # Humidity
    humidity = 60 + 0.15 * rainfall + np.random.normal(0, 5, n_months)
    humidity = np.clip(humidity, 40, 95)
    
    # Generate synthetic NDVI (Normalized Difference Vegetation Index)
    ndvi = 0.4 + 0.3 * np.sin(2 * np.pi * months / 12) + np.random.normal(0, 0.05, n_months)
    ndvi = np.clip(ndvi, 0.1, 0.9)
    
    # Calculate NDDI
    nddi_calculator = NDDICalculator()
    nddi = nddi_calculator.calculate_nddi(rainfall, temperature, humidity, ndvi)
    drought_risk = nddi_calculator.classify_drought_risk(nddi)
    nddi_impact = nddi_calculator.calculate_nddi_impact_factor(nddi)
    
    # Interventions
    llin_coverage = np.zeros(n_months)
    for i in range(n_months):
        if dates[i].month == 1:
            llin_coverage[i] = 80
        else:
            llin_coverage[i] = llin_coverage[i-1] * 0.95
    
    irs_coverage = np.zeros(n_months)
    for i in range(n_months):
        if dates[i].month in [4, 10]:
            irs_coverage[i] = 60
        else:
            irs_coverage[i] = irs_coverage[i-1] * 0.7
    
    llin_coverage += np.random.normal(0, 2, n_months)
    irs_coverage += np.random.normal(0, 5, n_months)
    llin_coverage = np.clip(llin_coverage, 10, 85)
    irs_coverage = np.clip(irs_coverage, 0, 65)
    
    # Population
    monthly_growth = 0.022 / 12
    population = 1000000 * np.exp(monthly_growth * np.arange(n_months))
    population = np.round(population)
    
    # Malaria cases with NDDI impact
    temp_effect = np.clip((temperature - 25) / 5, -1, 1)
    rainfall_effect = np.clip(rainfall / 100, 0, 2)
    humidity_effect = np.clip((humidity - 70) / 20, -1, 1)
    nddi_effect = nddi_impact  # Use NDDI impact factor
    
    climate_transmission = 0.5 + 0.3*temp_effect + 0.4*rainfall_effect + 0.2*humidity_effect
    llin_effect = 1 - (llin_coverage / 100) * 0.6
    irs_effect = 1 - (irs_coverage / 100) * 0.4
    seasonality = 0.8 + 0.4 * np.sin(2 * np.pi * months / 12 - np.pi/2)
    
    # Apply NDDI effect
    transmission_rate = climate_transmission * llin_effect * irs_effect * seasonality * nddi_effect
    base_cases = transmission_rate * 5 * population / 1000
    cases = base_cases + np.random.poisson(10, n_months)
    cases = np.round(cases).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'year': dates.year,
        'month': dates.month,
        'temperature': np.round(temperature, 1),
        'rainfall': np.round(rainfall, 1),
        'humidity': np.round(humidity, 1),
        'ndvi': np.round(ndvi, 3),
        'nddi': np.round(nddi, 3),
        'drought_risk': drought_risk,
        'nddi_impact': np.round(nddi_impact, 2),
        'population': population,
        'llin_coverage': np.round(llin_coverage, 1),
        'irs_coverage': np.round(irs_coverage, 1),
        'malaria_cases': cases,
        'incidence_rate': np.round((cases / population) * 1000, 2)
    })
    
    # Calculate additional NDDI-based features
    df['nddi_trend'] = df['nddi'].rolling(window=3, min_periods=1).mean()
    df['drought_duration'] = (df['nddi'] < -0.5).astype(int)
    df['drought_duration'] = df['drought_duration'].groupby((df['drought_duration'] != df['drought_duration'].shift()).cumsum()).cumsum()
    
    st.success(f"✅ Generated {len(df)} months of synthetic data with NDDI!")
    return df

# ============================================================================
# 4. TRAIN MODELS FUNCTION
# ============================================================================
def train_all_models(data):
    """Train all models including MIL"""
    st.info("🤖 Training models (including Multi-Instance Learning)...")
    
    # Prepare data for MIL
    mil_predictor = MultiInstanceMalariaPredictor(window_size=3, n_clusters=5)
    
    # Create instances for MIL
    with st.spinner("Creating multi-instance representations..."):
        instances, instance_labels = mil_predictor.create_instances(data)
        
        if len(instances) > 0:
            # Scale instances
            scaled_instances = mil_predictor.scaler.fit_transform(instances)
            
            # Cluster instances
            cluster_labels = mil_predictor.cluster_instances(scaled_instances)
            
            # Train cluster-specific predictors
            mil_predictor.train_instance_predictors(scaled_instances, instance_labels, cluster_labels)
            
            # Make predictions
            mil_predictions = mil_predictor.predict(scaled_instances)
            
            # Align predictions with original data
            aligned_predictions = np.zeros(len(data))
            aligned_predictions[:] = np.nan
            aligned_predictions[3:3+len(mil_predictions)] = mil_predictions
            
            st.session_state.mil_predictions = aligned_predictions
            
            # Calculate performance metrics
            valid_indices = ~np.isnan(aligned_predictions)
            if np.sum(valid_indices) > 10:
                actual_cases = data['malaria_cases'].values[valid_indices]
                pred_cases = aligned_predictions[valid_indices]
                
                mil_rmse = np.sqrt(mean_squared_error(actual_cases, pred_cases))
                mil_r2 = r2_score(actual_cases, pred_cases)
                
                st.success(f"""
                ✅ Multi-Instance Learning trained successfully:
                - RMSE: {mil_rmse:.1f}
                - R² Score: {mil_r2:.3f}
                - Clusters used: {len(mil_predictor.instance_predictors)}
                """)
        else:
            st.warning("Insufficient data for Multi-Instance Learning")
            st.session_state.mil_predictions = None
    
    # Train traditional models for comparison
    X = data[['temperature', 'rainfall', 'humidity', 'nddi', 
              'llin_coverage', 'irs_coverage']].values
    y = data['malaria_cases'].values
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_predictions = rf_model.predict(X)
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X, y)
    gb_predictions = gb_model.predict(X)
    
    return {
        'rf_predictions': rf_predictions,
        'gb_predictions': gb_predictions,
        'rf_model': rf_model,
        'gb_model': gb_model
    }

# ============================================================================
# 5. MAIN APP LAYOUT
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
        st.title("🦟 Malaria Forecasting System")
        st.markdown("**with NDDI & Multi-Instance Learning**")
        st.markdown("---")
        
        # Data generation
        if st.button("📊 Generate Synthetic Data", use_container_width=True):
            with st.spinner("Generating data with NDDI features..."):
                st.session_state.data = generate_synthetic_data()
                st.session_state.data_generated = True
                st.session_state.models_trained = False
                st.rerun()
        
        if st.session_state.data_generated:
            st.success("Data available!")
            if st.button("🤖 Train All Models", use_container_width=True):
                with st.spinner("Training models including MIL..."):
                    st.session_state.model_results = train_all_models(st.session_state.data)
                    st.session_state.models_trained = True
                    st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Advanced Settings")
        
        # MIL Settings
        st.markdown("#### Multi-Instance Learning")
        window_size = st.slider("Temporal Window Size", 2, 6, 3)
        n_clusters = st.slider("Number of Clusters", 2, 8, 5)
        
        # NDDI Settings
        st.markdown("#### NDDI Analysis")
        show_nddi_analysis = st.checkbox("Show Detailed NDDI Analysis", True)
        
        # Forecast months
        forecast_months = st.slider("Forecast Months", 1, 24, 12)
        
        # Intervention sliders
        st.markdown("### 🛡️ Interventions")
        llin_increase = st.slider("LLIN Coverage Increase (%)", 0, 100, 0)
        irs_increase = st.slider("IRS Coverage Increase (%)", 0, 100, 0)
        
        st.markdown("---")
        st.markdown("""
        ### 📈 New Features Added:
        - **NDDI (Normalized Difference Drought Index)**
        - **Multi-Instance Learning (MIL)**
        - **Drought risk classification**
        - **Cluster-based predictions**
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">🦟 Malaria Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #0EA5E9;">with NDDI & Multi-Instance Learning</h3>', unsafe_allow_html=True)
    
    if not st.session_state.data_generated:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=200)
            st.markdown("""
            <div style='text-align: center;'>
                <h2>Welcome to the Enhanced Malaria Forecasting System</h2>
                <p>Now featuring <strong>NDDI (Normalized Difference Drought Index)</strong> and 
                <strong>Multi-Instance Learning</strong> for improved predictions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👈 Click 'Generate Synthetic Data' in the sidebar to get started!")
            
            # Quick stats
            st.markdown("### 📊 Enhanced Capabilities:")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Data Points", "120 months")
            with col2:
                st.metric("NDDI Features", "Added")
            with col3:
                st.metric("MIL Clusters", "5 adaptive")
            with col4:
                st.metric("Algorithms", "6 total")
        
        return
    
    # Display data preview
    st.markdown('<h2 class="sub-header">📋 Enhanced Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Months", len(st.session_state.data))
    with col2:
        st.metric("Avg Temperature", f"{st.session_state.data['temperature'].mean():.1f}°C")
    with col3:
        st.metric("Avg NDDI", f"{st.session_state.data['nddi'].mean():.3f}")
    with col4:
        st.metric("Drought Risk Months", 
                 f"{(st.session_state.data['nddi'] < -0.5).sum()}")
    
    # NDDI Overview Box
    st.markdown("""
    <div class="nddi-box">
        <h4>🌵 NDDI (Normalized Difference Drought Index) Insights</h4>
        <p><strong>NDDI Range:</strong> {:.3f} to {:.3f} (Negative = Drought, Positive = Moist)</p>
        <p><strong>Drought Risk Classification:</strong> {} severe, {} moderate, {} mild drought months detected</p>
        <p><strong>Impact on Malaria:</strong> NDDI values influence transmission rates through {} factor</p>
    </div>
    """.format(
        st.session_state.data['nddi'].min(),
        st.session_state.data['nddi'].max(),
        (st.session_state.data['nddi'] < -1.0).sum(),
        ((st.session_state.data['nddi'] >= -1.0) & (st.session_state.data['nddi'] < -0.5)).sum(),
        ((st.session_state.data['nddi'] >= -0.5) & (st.session_state.data['nddi'] < 0)).sum(),
        "nddi_impact"
    ), unsafe_allow_html=True)
    
    # Enhanced Data visualization tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Time Series", "🌵 NDDI Analysis", "🌡️ Climate Patterns", "📊 Statistics"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Time series with cases
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=st.session_state.data['malaria_cases'],
                mode='lines',
                name='Malaria Cases',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=st.session_state.data['temperature'],
                mode='lines',
                name='Temperature',
                yaxis='y2',
                line=dict(color='blue', width=1, dash='dash')
            ))
            fig.update_layout(
                title='Malaria Cases and Temperature',
                xaxis_title='Date',
                yaxis_title='Malaria Cases',
                yaxis2=dict(
                    title='Temperature (°C)',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # NDDI vs Cases
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=st.session_state.data['nddi'],
                mode='lines',
                name='NDDI',
                line=dict(color='brown', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=st.session_state.data['malaria_cases'],
                mode='lines',
                name='Malaria Cases',
                yaxis='y2',
                line=dict(color='red', width=1, dash='dash')
            ))
            fig.update_layout(
                title='NDDI Index and Malaria Cases',
                xaxis_title='Date',
                yaxis_title='NDDI Value',
                yaxis2=dict(
                    title='Malaria Cases',
                    overlaying='y',
                    side='right'
                ),
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # NDDI Analysis
        col1, col2 = st.columns(2)
        with col1:
            # NDDI distribution
            fig = px.histogram(
                st.session_state.data,
                x='nddi',
                color='drought_risk',
                title='NDDI Distribution by Drought Risk',
                nbins=30,
                barmode='overlay',
                opacity=0.7
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # NDDI-Cases correlation
            corr_nddi = st.session_state.data['nddi'].corr(st.session_state.data['malaria_cases'])
            st.metric("NDDI-Cases Correlation", f"{corr_nddi:.3f}", 
                     "Positive: Moist conditions increase cases" if corr_nddi > 0 else 
                     "Negative: Drought conditions affect cases")
        
        with col2:
            # NDDI impact on cases
            fig = px.scatter(
                st.session_state.data,
                x='nddi',
                y='malaria_cases',
                color='drought_risk',
                title='NDDI vs Malaria Cases',
                trendline='ols',
                size='rainfall',
                hover_data=['temperature', 'month']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Drought risk summary
            drought_summary = st.session_state.data['drought_risk'].value_counts()
            st.dataframe(drought_summary, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            # Rainfall vs Cases with NDDI
            fig = px.scatter(
                st.session_state.data,
                x='rainfall',
                y='malaria_cases',
                color='nddi',
                title='Rainfall vs Cases (colored by NDDI)',
                trendline='ols',
                color_continuous_scale='RdYlBu'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature vs Cases with NDDI
            fig = px.scatter(
                st.session_state.data,
                x='temperature',
                y='malaria_cases',
                color='nddi',
                title='Temperature vs Cases (colored by NDDI)',
                trendline='ols',
                color_continuous_scale='RdYlBu'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
        with col2:
            # Enhanced correlation matrix
            corr_data = st.session_state.data[['temperature', 'rainfall', 'humidity', 
                                              'nddi', 'nddi_impact',
                                              'llin_coverage', 'irs_coverage', 'malaria_cases']].corr()
            fig = px.imshow(corr_data, text_auto=True, aspect="auto", 
                          title="Enhanced Correlation Matrix with NDDI",
                          color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
    
    # Model training and results
    if st.session_state.models_trained:
        st.markdown('<h2 class="sub-header">🤖 Enhanced Model Results</h2>', unsafe_allow_html=True)
        
        # Multi-Instance Learning Section
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 Multi-Instance Learning (MIL) Approach</h4>
            <p><strong>Method:</strong> Temporal windows ({} months) → Clustering ({} clusters) → Cluster-specific predictors</p>
            <p><strong>Advantage:</strong> Adapts to different epidemiological patterns automatically</p>
        </div>
        """.format(3, 5), unsafe_allow_html=True)
        
        # Get predictions
        dates = st.session_state.data['date']
        cases = st.session_state.data['malaria_cases']
        
        if 'model_results' in st.session_state:
            rf_predictions = st.session_state.model_results['rf_predictions']
            gb_predictions = st.session_state.model_results['gb_predictions']
            mil_predictions = st.session_state.mil_predictions
            
            # Calculate metrics
            metrics = []
            models = ['Random Forest', 'Gradient Boosting', 'Multi-Instance Learning']
            predictions_list = [rf_predictions, gb_predictions, mil_predictions]
            
            for i, (model_name, preds) in enumerate(zip(models, predictions_list)):
                if preds is not None and len(preds) == len(cases):
                    valid_indices = ~np.isnan(preds) if i == 2 else slice(None)
                    actual = cases[valid_indices]
                    predicted = preds[valid_indices] if i == 2 else preds[valid_indices]
                    
                    if len(actual) > 0:
                        rmse = np.sqrt(mean_squared_error(actual, predicted))
                        r2 = r2_score(actual, predicted)
                        mae = mean_absolute_error(actual, predicted)
                        metrics.append({
                            'Model': model_name,
                            'RMSE': rmse,
                            'R²': r2,
                            'MAE': mae,
                            'Predictions': predicted
                        })
        
        # Display model performance
        if metrics:
            col1, col2, col3 = st.columns(3)
            for i, metric in enumerate(metrics[:3]):
                with [col1, col2, col3][i]:
                    st.metric(f"{metric['Model']} R²", f"{metric['R²']:.3f}")
                    st.metric(f"{metric['Model']} RMSE", f"{metric['RMSE']:.0f}")
            
            # Model comparison plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=cases,
                mode='lines',
                name='Actual Cases',
                line=dict(color='black', width=3)
            ))
            
            colors = ['blue', 'green', 'red']
            for i, metric in enumerate(metrics):
                if i < 3:  # Show first 3 models
                    preds = metric['Predictions']
                    if metric['Model'] == 'Multi-Instance Learning':
                        # For MIL, we need to handle NaN values
                        valid_dates = dates[~np.isnan(preds)]
                        valid_preds = preds[~np.isnan(preds)]
                        fig.add_trace(go.Scatter(
                            x=valid_dates,
                            y=valid_preds,
                            mode='lines',
                            name=f"{metric['Model']} (R²={metric['R²']:.3f})",
                            line=dict(color=colors[i], width=2, dash='dash')
                        ))
                    else:
                        fig.add_trace(go.Scatter(
                            x=dates,
                            y=preds,
                            mode='lines',
                            name=f"{metric['Model']} (R²={metric['R²']:.3f})",
                            line=dict(color=colors[i], width=2, dash='dash')
                        ))
            
            fig.update_layout(
                title='Enhanced Model Predictions vs Actual Cases',
                xaxis_title='Date',
                yaxis_title='Malaria Cases',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison table
            metrics_df = pd.DataFrame(metrics)[['Model', 'RMSE', 'R²', 'MAE']]
            st.dataframe(metrics_df.style.highlight_max(subset=['R²'], color='lightgreen')
                        .highlight_min(subset=['RMSE', 'MAE'], color='lightgreen'),
                        use_container_width=True)
        
        # Enhanced Forecast section with NDDI
        st.markdown('<h2 class="sub-header">🔮 Enhanced Future Forecast</h2>', unsafe_allow_html=True)
        
        # Generate forecast with NDDI considerations
        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                      periods=forecast_months, freq='M')
        forecast_months_numeric = forecast_dates.month
        
        # Base forecast with seasonality
        seasonal_pattern = 0.8 + 0.4 * np.sin(2 * np.pi * forecast_months_numeric / 12 - np.pi/2)
        base_forecast = cases.mean() * seasonal_pattern * np.random.normal(1, 0.2, forecast_months)
        base_forecast = np.maximum(base_forecast, 0)
        
        # Add NDDI-based forecast
        nddi_trend = st.session_state.data['nddi'].tail(6).mean()  # Use recent NDDI trend
        if nddi_trend < -0.5:  # Drought conditions
            nddi_adjustment = 1.3  # Increase due to water collection
        elif nddi_trend > 0.5:  # Very moist
            nddi_adjustment = 1.4  # Increase due to breeding sites
        else:
            nddi_adjustment = 1.0
        
        nddi_forecast = base_forecast * nddi_adjustment
        
        # Apply intervention effects
        intervention_factor = (1 - llin_increase/100 * 0.3) * (1 - irs_increase/100 * 0.2)
        intervention_forecast = nddi_forecast * intervention_factor
        
        # Display enhanced forecast
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=base_forecast,
                mode='lines+markers',
                name='Baseline Forecast',
                line=dict(color='gray', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=nddi_forecast,
                mode='lines+markers',
                name=f'With NDDI Adjustment',
                line=dict(color='orange', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=intervention_forecast,
                mode='lines+markers',
                name=f'With NDDI + Interventions',
                line=dict(color='green', width=3)
            ))
            fig.update_layout(
                title=f'Enhanced {forecast_months}-Month Malaria Forecast',
                xaxis_title='Date',
                yaxis_title='Predicted Cases',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # NDDI forecast interpretation
            st.markdown("""
            <div class="nddi-box">
                <h5>🌵 NDDI-Based Forecast Interpretation:</h5>
                <p><strong>Current NDDI Trend:</strong> {:.3f} ({})</p>
                <p><strong>Forecast Adjustment:</strong> {}% {} due to {} conditions</p>
                <p><strong>Key Insight:</strong> {}</p>
            </div>
            """.format(
                nddi_trend,
                "Drought" if nddi_trend < -0.5 else "Moist" if nddi_trend > 0.5 else "Normal",
                abs(nddi_adjustment-1)*100,
                "increase" if nddi_adjustment > 1 else "decrease",
                "drought" if nddi_trend < -0.5 else "moist" if nddi_trend > 0.5 else "normal",
                "Drought conditions may increase breeding sites in remaining water pools" if nddi_trend < -0.5 else
                "Moist conditions create ideal mosquito breeding environments" if nddi_trend > 0.5 else
                "Normal conditions with seasonal patterns dominant"
            ), unsafe_allow_html=True)
        
        with col2:
            # Forecast metrics
            st.markdown("### 📊 Enhanced Forecast Summary")
            
            cases_averted_nddi = np.sum(nddi_forecast) - np.sum(intervention_forecast)
            reduction_pct_nddi = (cases_averted_nddi / np.sum(nddi_forecast)) * 100
            
            st.metric("Baseline Total", f"{np.sum(base_forecast):,.0f}")
            st.metric("NDDI-Adjusted Total", f"{np.sum(nddi_forecast):,.0f}")
            st.metric("With Interventions", f"{np.sum(intervention_forecast):,.0f}")
            st.metric("Cases Averted", f"{cases_averted_nddi:,.0f}")
            st.metric("Reduction from NDDI+Interventions", f"{reduction_pct_nddi:.1f}%")
            
            # Intervention impact
            st.markdown("### 🛡️ Enhanced Intervention Impact")
            impact_text = ""
            if nddi_trend < -0.5:
                impact_text += "⚠️ **Drought Alert:** During drought, target water collection sites for larviciding.\n\n"
            elif nddi_trend > 0.5:
                impact_text += "💧 **Moist Conditions:** Increase IRS coverage in flood-prone areas.\n\n"
            
            if llin_increase > 0 or irs_increase > 0:
                st.success(f"""
                **Enhanced Intervention Strategy:**
                - LLIN coverage: +{llin_increase}%
                - IRS coverage: +{irs_increase}%
                - NDDI-based targeting: {nddi_trend:.2f} ({'Drought' if nddi_trend < -0.5 else 'Moist' if nddi_trend > 0.5 else 'Normal'})
                
                **Expected Impact:**
                - Prevent ~{cases_averted_nddi:,.0f} cases
                - Reduce transmission by {reduction_pct_nddi:.1f}%
                - NDDI adjustment: {nddi_adjustment:.1f}x multiplier
                
                {impact_text}
                """)
            else:
                st.info(f"""
                **Current NDDI Status:** {nddi_trend:.2f} ({'Drought' if nddi_trend < -0.5 else 'Moist' if nddi_trend > 0.5 else 'Normal'})
                {impact_text}
                Adjust intervention sliders in sidebar to see impact.
                """)
        
        # Enhanced What-if scenarios with NDDI
        st.markdown('<h2 class="sub-header">🔄 Enhanced What-If Scenarios</h2>', unsafe_allow_html=True)
        
        scenarios = st.multiselect(
            "Select enhanced scenarios to compare:",
            ["Severe Drought (NDDI = -1.5)", "Extreme Moist (NDDI = +1.5)", 
             "Climate Change (+3°C)", "Improved NDDI Monitoring",
             "Economic Downturn", "Healthcare System Strengthening"],
            default=["Severe Drought (NDDI = -1.5)", "Extreme Moist (NDDI = +1.5)"]
        )
        
        if scenarios:
            # Generate scenario forecasts
            scenario_data = []
            for scenario in scenarios:
                if "Severe Drought" in scenario:
                    factor = 1.5  # High risk during drought
                    nddi_value = -1.5
                elif "Extreme Moist" in scenario:
                    factor = 1.6  # Very high risk
                    nddi_value = 1.5
                elif "Climate Change" in scenario:
                    factor = 1.4  # Temperature increase
                    nddi_value = 0
                elif "Improved NDDI" in scenario:
                    factor = 0.8  # Better monitoring reduces cases
                    nddi_value = 0
                elif "Economic" in scenario:
                    factor = 1.3  # Reduced interventions
                    nddi_value = 0
                elif "Healthcare" in scenario:
                    factor = 0.7  # Better healthcare
                    nddi_value = 0
                else:
                    factor = 1.0
                    nddi_value = 0
                
                scenario_forecast = base_forecast * factor
                scenario_data.append((scenario, scenario_forecast, nddi_value))
            
            # Plot enhanced scenarios
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=base_forecast,
                mode='lines',
                name='Baseline',
                line=dict(color='black', width=3)
            ))
            
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
            for i, (scenario_name, forecast_vals, nddi_val) in enumerate(scenario_data):
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_vals,
                    mode='lines',
                    name=f"{scenario_name} (NDDI={nddi_val})",
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='Enhanced What-If Scenario Analysis with NDDI',
                xaxis_title='Date',
                yaxis_title='Predicted Cases',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Scenario summary table
            scenario_summary = []
            for scenario_name, forecast_vals, nddi_val in scenario_data:
                total_cases = np.sum(forecast_vals)
                vs_baseline = (total_cases / np.sum(base_forecast) - 1) * 100
                scenario_summary.append({
                    'Scenario': scenario_name,
                    'Total Cases': f"{total_cases:,.0f}",
                    'vs Baseline': f"{vs_baseline:+.1f}%",
                    'NDDI Value': nddi_val
                })
            
            st.dataframe(pd.DataFrame(scenario_summary), use_container_width=True)
        
        # Enhanced Recommendations with NDDI insights
        st.markdown('<h2 class="sub-header">💡 Enhanced Recommendations</h2>', unsafe_allow_html=True)
        
        # Analyze NDDI patterns
        recent_nddi = st.session_state.data['nddi'].tail(12)
        drought_months = (recent_nddi < -0.5).sum()
        moist_months = (recent_nddi > 0.5).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🌵 NDDI-Based Interventions</h4>
                <p><strong>During Drought (NDDI < -0.5):</strong></p>
                <p>• Target water collection sites</p>
                <p>• Increase larviciding in remaining pools</p>
                <p>• Monitor artificial containers</p>
                <p><strong>During Moist (NDDI > 0.5):</strong></p>
                <p>• Increase IRS in flood areas</p>
                <p>• Deploy extra LLINs</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🧠 Multi-Instance Strategy</h4>
                <p><strong>Pattern-Based Response:</strong></p>
                <p>• Cluster 1: Urban outbreaks → Focus on IRS</p>
                <p>• Cluster 2: Rural seasonal → Focus on LLINs</p>
                <p>• Cluster 3: Drought-related → Water management</p>
                <p>• Cluster 4: Flood-related → Emergency response</p>
                <p>• Cluster 5: Stable → Maintain surveillance</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>📊 Data-Driven Actions</h4>
                <p><strong>NDDI Monitoring:</strong></p>
                <p>• Alert threshold: NDDI < -0.8 or > 0.8</p>
                <p>• Response time: Within 2 weeks</p>
                <p>• Resource allocation: Based on cluster</p>
                <p><strong>MIL Updates:</strong></p>
                <p>• Retrain clusters every 6 months</p>
                <p>• Update window size seasonally</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical details expander
        with st.expander("🔧 Technical Details: NDDI & MIL Implementation"):
            st.markdown("""
            ### NDDI Implementation:
            ```
            NDDI = (Rainfall_norm - Temperature_norm) / 
                   (Rainfall_norm + Temperature_norm) * Humidity_factor
            ```
            **Key Features:**
            1. Normalized climate indicators
            2. Humidity correction factor
            3. Drought risk classification
            4. Impact factor calculation
            
            ### Multi-Instance Learning:
            **Process:**
            1. **Instance Creation:** 3-month temporal windows
            2. **Clustering:** K-means (k=5) groups similar patterns
            3. **Model Training:** Random Forest per cluster
            4. **Prediction:** Cluster-specific prediction
            
            **Advantages:**
            - Adapts to different outbreak patterns
            - Handles temporal dependencies
            - Provides pattern-specific interventions
            """)
            
            # Show NDDI calculation example
            sample_idx = 50
            sample_data = st.session_state.data.iloc[sample_idx]
            st.code(f"""
            Sample Calculation (Month {sample_idx}):
            Rainfall: {sample_data['rainfall']} mm
            Temperature: {sample_data['temperature']} °C
            Humidity: {sample_data['humidity']} %
            Calculated NDDI: {sample_data['nddi']:.3f}
            Drought Risk: {sample_data['drought_risk']}
            Impact Factor: {sample_data['nddi_impact']:.2f}x
            """)
    
    else:
        # Show data ready message with NDDI details
        st.markdown("""
        <div class="success-box">
            <h3>✅ Enhanced Data Generated Successfully!</h3>
            <p><strong>New Features Available:</strong></p>
            <ul>
                <li><strong>NDDI (Normalized Difference Drought Index)</strong> calculated for each month</li>
                <li><strong>Drought risk classification</strong> (Severe, Moderate, Mild, Normal, Moist, Very Moist)</li>
                <li><strong>NDDI impact factors</strong> on malaria transmission</li>
                <li><strong>Temporal features</strong> ready for Multi-Instance Learning</li>
            </ul>
            <p>Click <strong>"Train All Models"</strong> in the sidebar to build enhanced prediction models.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick insights from enhanced data
        st.markdown('<h3>📊 Enhanced Quick Insights</h3>', unsafe_allow_html=True)
        
        # Find correlations with NDDI
        corr_temp = st.session_state.data['temperature'].corr(st.session_state.data['malaria_cases'])
        corr_rain = st.session_state.data['rainfall'].corr(st.session_state.data['malaria_cases'])
        corr_nddi = st.session_state.data['nddi'].corr(st.session_state.data['malaria_cases'])
        corr_nddi_impact = st.session_state.data['nddi_impact'].corr(st.session_state.data['malaria_cases'])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            trend = "Positive" if corr_temp > 0 else "Negative"
            color = "🔴" if corr_temp > 0.3 else "🟡" if corr_temp > 0 else "🟢"
            st.metric(f"{color} Temperature", f"{corr_temp:.3f}")
        
        with col2:
            trend = "Positive" if corr_rain > 0 else "Negative"
            color = "🔴" if corr_rain > 0.3 else "🟡" if corr_rain > 0 else "🟢"
            st.metric(f"{color} Rainfall", f"{corr_rain:.3f}")
        
        with col3:
            trend = "Positive" if corr_nddi > 0 else "Negative"
            color = "🔴" if abs(corr_nddi) > 0.2 else "🟡"
            st.metric(f"{color} NDDI", f"{corr_nddi:.3f}")
        
        with col4:
            trend = "Positive" if corr_nddi_impact > 0 else "Negative"
            color = "🔴" if corr_nddi_impact > 0.3 else "🟡" if corr_nddi_impact > 0 else "🟢"
            st.metric(f"{color} NDDI Impact", f"{corr_nddi_impact:.3f}")
        
        # NDDI distribution summary - FIXED LINE
        st.markdown("#### 🌵 NDDI Distribution Summary")
        nddi_stats = st.session_state.data['drought_risk'].value_counts().reset_index()
        nddi_stats.columns = ['Drought Risk Level', 'Count']
        
        # Use a valid color sequence - FIX APPLIED HERE
        fig = px.bar(nddi_stats, x='Drought Risk Level', y='Count', 
                    title='Distribution of Drought Risk Levels',
                    color='Drought Risk Level',
                    color_discrete_sequence=px.colors.qualitative.Set3)  # Changed to valid color sequence
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak months with NDDI consideration
        monthly_avg = st.session_state.data.groupby('month').agg({
            'malaria_cases': 'mean',
            'nddi': 'mean'
        }).reset_index()
        
        peak_month = monthly_avg.loc[monthly_avg['malaria_cases'].idxmax(), 'month']
        peak_month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][peak_month-1]
        peak_nddi = monthly_avg.loc[monthly_avg['malaria_cases'].idxmax(), 'nddi']
        
        st.info(f"""
        **Seasonal Pattern Analysis:**
        - Peak transmission month: **{peak_month_name}** (average {monthly_avg['malaria_cases'].max():.0f} cases)
        - NDDI during peak: **{peak_nddi:.3f}** ({'Drought' if peak_nddi < -0.5 else 'Moist' if peak_nddi > 0.5 else 'Normal'})
        - Correlation pattern: { 'NDDI influences seasonal peaks' if abs(corr_nddi) > 0.2 else 'Seasonal factors dominant' }
        """)

# Run the app
if __name__ == "__main__":
    main()
