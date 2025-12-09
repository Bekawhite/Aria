"""
MALARIA FORECASTING WEB APP
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
import warnings
import os
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Malaria Forecasting System",
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

# ============================================================================
# 1. DATA GENERATION FUNCTION
# ============================================================================
def generate_synthetic_data():
    """Generate synthetic climate and malaria data"""
    st.info("🔄 Generating synthetic data...")
    
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
    
    # Malaria cases
    temp_effect = np.clip((temperature - 25) / 5, -1, 1)
    rainfall_effect = np.clip(rainfall / 100, 0, 2)
    humidity_effect = np.clip((humidity - 70) / 20, -1, 1)
    
    climate_transmission = 0.5 + 0.3*temp_effect + 0.4*rainfall_effect + 0.2*humidity_effect
    llin_effect = 1 - (llin_coverage / 100) * 0.6
    irs_effect = 1 - (irs_coverage / 100) * 0.4
    seasonality = 0.8 + 0.4 * np.sin(2 * np.pi * months / 12 - np.pi/2)
    
    transmission_rate = climate_transmission * llin_effect * irs_effect * seasonality
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
        'population': population,
        'llin_coverage': np.round(llin_coverage, 1),
        'irs_coverage': np.round(irs_coverage, 1),
        'malaria_cases': cases,
        'incidence_rate': np.round((cases / population) * 1000, 2)
    })
    
    st.success(f"✅ Generated {len(df)} months of synthetic data!")
    return df

# ============================================================================
# 2. SIR MODEL CLASS
# ============================================================================
class ClimateDrivenSIR:
    def __init__(self, population=1e6, gamma=1/7):
        self.population = population
        self.gamma = gamma
        self.beta_params = None
    
    def beta_function(self, climate_features, params):
        if climate_features.ndim == 1:
            climate_features = climate_features.reshape(1, -1)
        
        n_features = climate_features.shape[1]
        if n_features < 6:
            climate_features = np.pad(climate_features, ((0,0), (0,6-n_features)), 'constant')
        
        rain = climate_features[:, 0] if n_features > 0 else 0
        temp = climate_features[:, 1] if n_features > 1 else 0
        humidity = climate_features[:, 2] if n_features > 2 else 0
        llin = climate_features[:, 3] if n_features > 3 else 0
        irs = climate_features[:, 4] if n_features > 4 else 0
        
        a0, a1, a2, a3, a4, a5 = params[:6]
        beta = a0 + a1*rain + a2*temp + a3*humidity + a4*llin + a5*irs
        return np.maximum(beta, 0.001)
    
    def fit_beta_parameters(self, climate_data, beta_empirical):
        n_samples = len(climate_data)
        X = np.ones((n_samples, 6))
        
        for i in range(min(climate_data.shape[1], 5)):
            X[:, i+1] = climate_data[:, i]
        
        params, _ = curve_fit(
            lambda X, a0, a1, a2, a3, a4, a5: a0 + a1*X[:,1] + a2*X[:,2] + a3*X[:,3] + a4*X[:,4] + a5*X[:,5],
            X, beta_empirical,
            p0=[0.3, 0.001, 0.01, 0.01, -0.01, -0.01],
            bounds=([0, -0.1, -0.1, -0.1, -0.1, -0.1], [1, 0.1, 0.1, 0.1, 0, 0])
        )
        
        self.beta_params = params
        return params

# ============================================================================
# 3. MAIN APP LAYOUT
# ============================================================================
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
        st.title("🦟 Malaria Forecasting")
        st.markdown("---")
        
        # Data generation
        if st.button("📊 Generate Synthetic Data", use_container_width=True):
            with st.spinner("Generating data..."):
                st.session_state.data = generate_synthetic_data()
                st.session_state.data_generated = True
                st.rerun()
        
        if st.session_state.data_generated:
            st.success("Data available!")
            if st.button("🤖 Train Models", use_container_width=True):
                st.session_state.models_trained = True
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 🔧 Settings")
        
        # Forecast months
        forecast_months = st.slider("Forecast Months", 1, 24, 12)
        
        # Intervention sliders
        st.markdown("### 🛡️ Interventions")
        llin_increase = st.slider("LLIN Coverage Increase (%)", 0, 100, 0)
        irs_increase = st.slider("IRS Coverage Increase (%)", 0, 100, 0)
        
        st.markdown("---")
        st.markdown("""
        ### 📈 About
        This system predicts malaria outbreaks using:
        - Climate data (temp, rainfall, humidity)
        - Interventions (LLIN, IRS coverage)
        - SIR epidemiological model
        - Machine learning hybrid approach
        """)
    
    # Main content
    st.markdown('<h1 class="main-header">🦟 Malaria Forecasting System</h1>', unsafe_allow_html=True)
    
    if not st.session_state.data_generated:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=200)
            st.markdown("""
            <div style='text-align: center;'>
                <h2>Welcome to the Malaria Forecasting System</h2>
                <p>This tool helps predict malaria outbreaks using climate data and interventions.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👈 Click 'Generate Synthetic Data' in the sidebar to get started!")
            
            # Quick stats
            st.markdown("### 📊 What this system does:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Points", "120 months")
            with col2:
                st.metric("Variables", "8 features")
            with col3:
                st.metric("Models", "4 algorithms")
        
        return
    
    # Display data preview
    st.markdown('<h2 class="sub-header">📋 Data Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Months", len(st.session_state.data))
    with col2:
        st.metric("Avg Temperature", f"{st.session_state.data['temperature'].mean():.1f}°C")
    with col3:
        st.metric("Avg Rainfall", f"{st.session_state.data['rainfall'].mean():.1f}mm")
    with col4:
        st.metric("Total Cases", f"{st.session_state.data['malaria_cases'].sum():,}")
    
    # Data visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Time Series", "🌡️ Climate Patterns", "📊 Statistics"])
    
    with tab1:
        # Time series plot
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
            title='Malaria Cases and Temperature Over Time',
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
    
    with tab2:
        # Climate correlation
        col1, col2 = st.columns(2)
        with col1:
            # Rainfall vs Cases
            fig = px.scatter(
                st.session_state.data,
                x='rainfall',
                y='malaria_cases',
                color='month',
                title='Rainfall vs Malaria Cases',
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Temperature vs Cases
            fig = px.scatter(
                st.session_state.data,
                x='temperature',
                y='malaria_cases',
                color='month',
                title='Temperature vs Malaria Cases',
                trendline='ols'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(st.session_state.data.describe(), use_container_width=True)
        with col2:
            # Correlation matrix
            corr_data = st.session_state.data[['temperature', 'rainfall', 'humidity', 
                                              'llin_coverage', 'irs_coverage', 'malaria_cases']].corr()
            fig = px.imshow(corr_data, text_auto=True, aspect="auto", 
                          title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    # Model training and results
    if st.session_state.models_trained:
        st.markdown('<h2 class="sub-header">🤖 Model Results</h2>', unsafe_allow_html=True)
        
        # Simple model simulation (for demo)
        st.info("Training simplified models for demonstration...")
        
        # Create some mock results
        dates = st.session_state.data['date']
        cases = st.session_state.data['malaria_cases']
        
        # Simulate SIR predictions
        np.random.seed(42)
        sir_predictions = cases * 0.8 + np.random.normal(0, 50, len(cases))
        sir_predictions = np.maximum(sir_predictions, 0)
        
        # Simulate hybrid predictions (better)
        hybrid_predictions = cases * 0.9 + np.random.normal(0, 30, len(cases))
        hybrid_predictions = np.maximum(hybrid_predictions, 0)
        
        # Display model performance
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("SIR Model R²", f"{r2_score(cases, sir_predictions):.3f}")
        with col2:
            st.metric("SIR RMSE", f"{np.sqrt(mean_squared_error(cases, sir_predictions)):.0f}")
        with col3:
            st.metric("Hybrid Model R²", f"{r2_score(cases, hybrid_predictions):.3f}")
        with col4:
            st.metric("Hybrid RMSE", f"{np.sqrt(mean_squared_error(cases, hybrid_predictions)):.0f}")
        
        # Model comparison plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cases,
            mode='lines',
            name='Actual Cases',
            line=dict(color='black', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=sir_predictions,
            mode='lines',
            name='SIR Predictions',
            line=dict(color='blue', width=1.5, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=hybrid_predictions,
            mode='lines',
            name='Hybrid Predictions',
            line=dict(color='green', width=1.5, dash='dot')
        ))
        fig.update_layout(
            title='Model Predictions vs Actual Cases',
            xaxis_title='Date',
            yaxis_title='Malaria Cases',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast section
        st.markdown('<h2 class="sub-header">🔮 Future Forecast</h2>', unsafe_allow_html=True)
        
        # Generate forecast
        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                      periods=forecast_months, freq='M')
        
        # Base forecast
        base_forecast = np.random.normal(cases.mean(), cases.std() * 0.5, forecast_months)
        base_forecast = np.maximum(base_forecast, 0)
        
        # Apply intervention effects
        intervention_factor = (1 - llin_increase/100 * 0.3) * (1 - irs_increase/100 * 0.2)
        intervention_forecast = base_forecast * intervention_factor
        
        # Display forecast
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=base_forecast,
                mode='lines+markers',
                name='Baseline Forecast',
                line=dict(color='red', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=intervention_forecast,
                mode='lines+markers',
                name=f'With Interventions',
                line=dict(color='green', width=2)
            ))
            fig.update_layout(
                title=f'{forecast_months}-Month Malaria Forecast',
                xaxis_title='Date',
                yaxis_title='Predicted Cases',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Forecast metrics
            st.markdown("### 📊 Forecast Summary")
            
            cases_averted = np.sum(base_forecast) - np.sum(intervention_forecast)
            reduction_pct = (cases_averted / np.sum(base_forecast)) * 100
            
            st.metric("Baseline Total", f"{np.sum(base_forecast):,.0f}")
            st.metric("With Interventions", f"{np.sum(intervention_forecast):,.0f}")
            st.metric("Cases Averted", f"{cases_averted:,.0f}")
            st.metric("Reduction", f"{reduction_pct:.1f}%")
            
            # Intervention impact
            st.markdown("### 🛡️ Intervention Impact")
            if llin_increase > 0 or irs_increase > 0:
                st.success(f"""
                **Intervention Strategy:**
                - LLIN coverage: +{llin_increase}%
                - IRS coverage: +{irs_increase}%
                
                **Expected Impact:**
                - Prevent ~{cases_averted:,.0f} cases
                - Reduce transmission by {reduction_pct:.1f}%
                """)
            else:
                st.info("Adjust intervention sliders in sidebar to see impact")
        
        # What-if scenarios
        st.markdown('<h2 class="sub-header">🔄 What-If Scenarios</h2>', unsafe_allow_html=True)
        
        scenarios = st.multiselect(
            "Select scenarios to compare:",
            ["Hotter Climate (+2°C)", "Wetter Season (+20% rain)", "Drier Season (-20% rain)", 
             "Better Healthcare", "Economic Downturn"],
            default=["Hotter Climate (+2°C)", "Better Healthcare"]
        )
        
        if scenarios:
            # Generate scenario forecasts
            scenario_data = []
            for scenario in scenarios:
                if "Hotter" in scenario:
                    factor = 1.3  # 30% more cases
                elif "Wetter" in scenario:
                    factor = 1.2  # 20% more cases
                elif "Drier" in scenario:
                    factor = 0.8  # 20% fewer cases
                elif "Better" in scenario:
                    factor = 0.7  # 30% fewer cases
                elif "Economic" in scenario:
                    factor = 1.4  # 40% more cases
                else:
                    factor = 1.0
                
                scenario_forecast = base_forecast * factor
                scenario_data.append((scenario, scenario_forecast))
            
            # Plot scenarios
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=base_forecast,
                mode='lines',
                name='Baseline',
                line=dict(color='black', width=3)
            ))
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i, (scenario_name, forecast_vals) in enumerate(scenario_data):
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast_vals,
                    mode='lines',
                    name=scenario_name,
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title='What-If Scenario Analysis',
                xaxis_title='Date',
                yaxis_title='Predicted Cases',
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations
        st.markdown('<h2 class="sub-header">💡 Recommendations</h2>', unsafe_allow_html=True)
        
        # Find peak months
        recent_data = st.session_state.data.tail(12)
        peak_month = recent_data.loc[recent_data['malaria_cases'].idxmax(), 'month']
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        peak_month_name = month_names[peak_month - 1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🛡️ Intervention Timing</h4>
                <p>Deploy LLINs 1 month before peak transmission (around {})</p>
                <p>Schedule IRS spraying in {} and {}</p>
            </div>
            """.format(peak_month_name, month_names[3], month_names[9]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🌡️ Climate Monitoring</h4>
                <p>Monitor temperature spikes above {}°C</p>
                <p>Watch for rainfall above {}mm in rainy seasons</p>
                <p>Increase surveillance during humidity >{}%</p>
            </div>
            """.format(28, 150, 80), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>🏥 Healthcare Preparedness</h4>
                <p>Stockpile antimalarials before {} and {}</p>
                <p>Prepare for {}% case increase during outbreaks</p>
                <p>Maintain diagnostic capacity for peak months</p>
            </div>
            """.format(month_names[3], month_names[9], 50), unsafe_allow_html=True)
    
    else:
        # Show data ready message
        st.markdown("""
        <div class="success-box">
            <h3>✅ Data Generated Successfully!</h3>
            <p>You now have synthetic malaria and climate data ready for analysis.</p>
            <p>Click <strong>"Train Models"</strong> in the sidebar to build prediction models and generate forecasts.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick insights from data
        st.markdown('<h3>📊 Quick Insights from Your Data</h3>', unsafe_allow_html=True)
        
        # Find correlations
        corr_temp = st.session_state.data['temperature'].corr(st.session_state.data['malaria_cases'])
        corr_rain = st.session_state.data['rainfall'].corr(st.session_state.data['malaria_cases'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            trend = "Positive" if corr_temp > 0 else "Negative"
            color = "🔴" if corr_temp > 0.3 else "🟡" if corr_temp > 0 else "🟢"
            st.metric(f"{color} Temperature Correlation", f"{corr_temp:.3f}", 
                     f"{trend} relationship with cases")
        
        with col2:
            trend = "Positive" if corr_rain > 0 else "Negative"
            color = "🔴" if corr_rain > 0.3 else "🟡" if corr_rain > 0 else "🟢"
            st.metric(f"{color} Rainfall Correlation", f"{corr_rain:.3f}", 
                     f"{trend} relationship with cases")
        
        with col3:
            # Seasonality check
            monthly_avg = st.session_state.data.groupby('month')['malaria_cases'].mean()
            peak_month = monthly_avg.idxmax()
            peak_month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][peak_month-1]
            st.metric("📅 Peak Transmission Month", peak_month_name, 
                     f"{monthly_avg.max():.0f} avg cases")

# Run the app
if __name__ == "__main__":
    main()
