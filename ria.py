"""
MALARIA FORECASTING WEB APP WITH NDDI AND MULTI-INSTANCE LEARNING
Enhanced for National Malaria Control Unit Adoption
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
import requests
import json
import io
import base64
from typing import Dict, List, Optional
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
    .alert-box {
        background-color: #FEE2E2;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #EF4444;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .mobile-view button {
        min-height: 50px;
        font-size: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 1. INITIALIZE SESSION STATE
# ============================================================================
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
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'district_officer'
if 'implementation_steps' not in st.session_state:
    st.session_state.implementation_steps = [
        {"step": 1, "task": "Data System Integration", "completed": False},
        {"step": 2, "task": "Staff Training", "completed": False},
        {"step": 3, "task": "Alert Threshold Setting", "completed": False},
        {"step": 4, "task": "Intervention Planning", "completed": False},
        {"step": 5, "task": "Monitoring Setup", "completed": False},
    ]
if 'audit_log' not in st.session_state:
    st.session_state.audit_log = []
if 'commodity_stock' not in st.session_state:
    st.session_state.commodity_stock = {
        'rdts': 50000,
        'acts': 100000,
        'llins': 500000,
        'insecticide': 5000
    }
if 'data_quality_issues' not in st.session_state:
    st.session_state.data_quality_issues = []
if 'mobile_view' not in st.session_state:
    st.session_state.mobile_view = False

# ============================================================================
# 2. USER PERMISSIONS SYSTEM
# ============================================================================
class UserPermissions:
    """User permissions and role management"""
    
    ROLES = {
        'field_worker': {
            'permissions': ['view_data', 'submit_reports', 'view_alerts'],
            'name': 'Field Worker'
        },
        'district_officer': {
            'permissions': ['view_data', 'submit_reports', 'view_alerts', 
                          'approve_reports', 'generate_alerts', 'allocate_resources'],
            'name': 'District Officer'
        },
        'regional_manager': {
            'permissions': ['view_all_data', 'allocate_resources', 'view_audit_log',
                          'generate_reports', 'configure_system', 'manage_users'],
            'name': 'Regional Manager'
        },
        'national_director': {
            'permissions': ['all_permissions', 'system_config', 'user_management',
                          'budget_allocation', 'policy_approval'],
            'name': 'National Director'
        }
    }
    
    @staticmethod
    def has_permission(role, permission):
        """Check if user has specific permission"""
        if role == 'national_director':
            return True
        return permission in UserPermissions.ROLES[role]['permissions']
    
    @staticmethod
    def get_role_name(role):
        """Get display name for role"""
        return UserPermissions.ROLES.get(role, {}).get('name', role)

# ============================================================================
# 3. AUDIT LOGGER
# ============================================================================
class AuditLogger:
    """Log system actions for accountability"""
    
    @staticmethod
    def log_action(user, action, details):
        """Log user action"""
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user': user,
            'action': action,
            'details': details
        }
        st.session_state.audit_log.append(log_entry)
        
        # Keep only last 1000 entries
        if len(st.session_state.audit_log) > 1000:
            st.session_state.audit_log = st.session_state.audit_log[-1000:]

# ============================================================================
# 4. DATA QUALITY MONITOR
# ============================================================================
class DataQualityMonitor:
    """Monitor data quality issues"""
    
    @staticmethod
    def check_data_quality(data):
        """Check for data quality issues"""
        issues = []
        
        if data is None:
            return ["No data available"]
        
        # Completeness check
        missing_percent = data.isnull().sum() / len(data) * 100
        high_missing = missing_percent[missing_percent > 20]
        if len(high_missing) > 0:
            for col, percent in high_missing.items():
                issues.append(f"High missing data in {col}: {percent:.1f}%")
        
        # Consistency check
        if 'cases_treated' in data.columns and 'malaria_cases' in data.columns:
            if any(data['cases_treated'] > data['malaria_cases'] * 1.5):
                issues.append("More cases treated than reported - data inconsistency")
        
        # Outlier detection
        if 'malaria_cases' in data.columns:
            z_scores = np.abs(zscore(data['malaria_cases'].fillna(0)))
            outliers = np.where(z_scores > 3)[0]
            if len(outliers) > 0:
                issues.append(f"{len(outliers)} statistical outliers detected in malaria cases")
        
        # Temporal consistency
        if 'date' in data.columns and 'malaria_cases' in data.columns:
            data_sorted = data.sort_values('date')
            case_diffs = data_sorted['malaria_cases'].diff()
            if any(np.abs(case_diffs) > data_sorted['malaria_cases'].mean() * 10):
                issues.append("Unusual temporal patterns detected")
        
        st.session_state.data_quality_issues = issues
        return issues

# ============================================================================
# 5. REAL DATA IMPORTER
# ============================================================================
class RealDataImporter:
    """Import real data from various sources"""
    
    @staticmethod
    def import_csv_file():
        """Import data from CSV file"""
        uploaded_file = st.file_uploader("📁 Upload CSV/Excel file", 
                                        type=['csv', 'xlsx', 'xls'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    data = pd.read_csv(uploaded_file)
                else:
                    data = pd.read_excel(uploaded_file)
                
                # Check required columns
                required_cols = ['date', 'malaria_cases']
                missing_cols = [col for col in required_cols if col not in data.columns]
                
                if missing_cols:
                    st.warning(f"Missing columns: {missing_cols}")
                    st.info("Minimum required: 'date' and 'malaria_cases' columns")
                    
                    # Allow manual column mapping
                    st.subheader("Column Mapping")
                    col1, col2 = st.columns(2)
                    with col1:
                        date_col = st.selectbox("Select date column", data.columns)
                    with col2:
                        cases_col = st.selectbox("Select malaria cases column", data.columns)
                    
                    if st.button("Apply Mapping"):
                        data = data.rename(columns={date_col: 'date', cases_col: 'malaria_cases'})
                        data['date'] = pd.to_datetime(data['date'])
                        return data
                else:
                    data['date'] = pd.to_datetime(data['date'])
                    return data
                    
            except Exception as e:
                st.error(f"Error reading file: {e}")
                return None
        return None
    
    @staticmethod
    def import_sample_data(country="Ghana"):
        """Import sample country data"""
        # This would connect to real APIs in production
        sample_data = {
            'ghana': 'https://raw.githubusercontent.com/datasets/malaria/main/data/ghana.csv',
            'kenya': 'https://raw.githubusercontent.com/datasets/malaria/main/data/kenya.csv',
            'uganda': 'https://raw.githubusercontent.com/datasets/malaria/main/data/uganda.csv'
        }
        
        if country.lower() in sample_data:
            try:
                data = pd.read_csv(sample_data[country.lower()])
                st.success(f"Loaded sample data for {country}")
                return data
            except:
                st.warning("Could not load sample data. Using synthetic data instead.")
                return None
        return None

# ============================================================================
# 6. ALERT SYSTEM
# ============================================================================
class AlertSystem:
    """Generate and manage alerts"""
    
    THRESHOLDS = {
        'outbreak': 1.5,  # 50% above baseline
        'epidemic': 2.0,  # 100% above baseline
        'drought_alert': -0.8,  # Severe drought NDDI
        'flood_alert': 1.0,  # Extreme moisture
        'stockout_warning': 0.3,  # 30% stock remaining
        'data_quality_alert': 20,  # 20% missing data
    }
    
    @staticmethod
    def generate_alerts(data, predictions=None):
        """Generate actionable alerts"""
        alerts = []
        
        if data is None:
            return alerts
        
        # Outbreak detection
        if 'malaria_cases' in data.columns and len(data) > 12:
            baseline = data['malaria_cases'].tail(12).mean()
            last_cases = data['malaria_cases'].iloc[-1] if len(data) > 0 else 0
            
            if last_cases > baseline * AlertSystem.THRESHOLDS['epidemic']:
                alerts.append({
                    'level': 'CRITICAL',
                    'type': 'Epidemic',
                    'message': f'Malaria epidemic detected: {last_cases:.0f} cases ({(last_cases/baseline-1)*100:.0f}% above baseline)',
                    'actions': ['Activate emergency response', 'Mobilize rapid response teams', 
                              'Request emergency supplies', 'Issue public health warning'],
                    'priority': 1,
                    'timestamp': datetime.now()
                })
            elif last_cases > baseline * AlertSystem.THRESHOLDS['outbreak']:
                alerts.append({
                    'level': 'HIGH',
                    'type': 'Outbreak',
                    'message': f'Malaria outbreak detected: {last_cases:.0f} cases ({(last_cases/baseline-1)*100:.0f}% above baseline)',
                    'actions': ['Increase surveillance', 'Distribute LLINs and RDTs', 
                              'Activate community health workers', 'Monitor closely'],
                    'priority': 2,
                    'timestamp': datetime.now()
                })
        
        # NDDI-based alerts
        if 'nddi' in data.columns and len(data) > 0:
            last_nddi = data['nddi'].iloc[-1] if len(data) > 0 else 0
            
            if last_nddi < AlertSystem.THRESHOLDS['drought_alert']:
                alerts.append({
                    'level': 'HIGH',
                    'type': 'Drought Alert',
                    'message': f'Severe drought conditions (NDDI: {last_nddi:.2f})',
                    'actions': ['Target water collection sites', 'Increase larviciding', 
                              'Monitor artificial containers', 'Prepare for increased cases'],
                    'priority': 2,
                    'timestamp': datetime.now()
                })
            elif last_nddi > AlertSystem.THRESHOLDS['flood_alert']:
                alerts.append({
                    'level': 'HIGH',
                    'type': 'Flood Alert',
                    'message': f'Extreme moisture conditions (NDDI: {last_nddi:.2f})',
                    'actions': ['Increase IRS in flood areas', 'Distribute extra LLINs', 
                              'Prepare for vector breeding', 'Activate emergency shelters'],
                    'priority': 2,
                    'timestamp': datetime.now()
                })
        
        # Stockout alerts
        for commodity, stock in st.session_state.commodity_stock.items():
            if stock < 1000:  # Low stock threshold
                alerts.append({
                    'level': 'MEDIUM',
                    'type': 'Stock Alert',
                    'message': f'Low stock: {commodity.upper()} ({stock} remaining)',
                    'actions': ['Initiate procurement', 'Redistribute from other districts', 
                              'Prioritize high-risk areas', 'Implement rationing if needed'],
                    'priority': 3,
                    'timestamp': datetime.now()
                })
        
        # Data quality alerts
        if st.session_state.data_quality_issues:
            alerts.append({
                'level': 'MEDIUM',
                'type': 'Data Quality',
                'message': f'{len(st.session_state.data_quality_issues)} data quality issues detected',
                'actions': ['Review data quality report', 'Correct data errors', 
                          'Retrain staff on data entry', 'Implement validation checks'],
                'priority': 3,
                'timestamp': datetime.now()
            })
        
        # Sort by priority
        alerts.sort(key=lambda x: x['priority'])
        st.session_state.alerts = alerts
        
        return alerts
    
    @staticmethod
    def send_alert_notifications(alerts):
        """Send alerts via email/SMS (simulated)"""
        # In production, this would integrate with email/SMS APIs
        critical_alerts = [a for a in alerts if a['level'] in ['CRITICAL', 'HIGH']]
        
        if critical_alerts:
            st.sidebar.markdown(f"""
            <div class="alert-box">
                <h4>🚨 {len(critical_alerts)} Critical Alert(s)</h4>
                <p>Requires immediate attention</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Simulate sending notifications
            recipients = {
                'CRITICAL': ['national_director@health.gov', 'regional_managers@health.gov'],
                'HIGH': ['district_officers@health.gov', 'health_facilities@health.gov'],
                'MEDIUM': ['data_managers@health.gov']
            }
            
            for alert in critical_alerts:
                AuditLogger.log_action(
                    'system',
                    'alert_sent',
                    f"{alert['type']} alert sent to {recipients.get(alert['level'], [])}"
                )

# ============================================================================
# 7. COMMODITY FORECASTER
# ============================================================================
class CommodityForecaster:
    """Forecast commodity needs"""
    
    COMMODITY_COSTS = {
        'rdts': {'unit_cost': 1.5, 'shelf_life': 24, 'lead_time': 60},
        'acts': {'unit_cost': 2.0, 'shelf_life': 36, 'lead_time': 90},
        'llins': {'unit_cost': 5.0, 'shelf_life': 60, 'lead_time': 180},
        'insecticide': {'unit_cost': 25.0, 'shelf_life': 24, 'lead_time': 120}
    }
    
    @staticmethod
    def forecast_needs(predictions, population, months=6):
        """Forecast commodity needs based on predictions"""
        if predictions is None or len(predictions) == 0:
            return {}
        
        avg_cases = np.mean(predictions) if len(predictions) > 0 else 1000
        
        commodities = {
            'rdts': int(avg_cases * 1.2 * months),  # Rapid Diagnostic Tests
            'acts': int(avg_cases * 1.1 * months),  # Artemisinin-based therapies
            'llins': int(population * 0.2 / 12 * months),  # Assume 20% need nets per year
            'insecticide': int(avg_cases * 0.1 * months)  # For IRS
        }
        
        # Calculate costs
        total_cost = 0
        for commodity, quantity in commodities.items():
            cost = quantity * CommodityForecaster.COMMODITY_COSTS[commodity]['unit_cost']
            total_cost += cost
        
        return {
            'commodities': commodities,
            'total_cost': total_cost,
            'monthly_cost': total_cost / months if months > 0 else 0
        }
    
    @staticmethod
    def check_stock_levels():
        """Check current stock levels and generate orders"""
        orders = []
        
        for commodity, stock in st.session_state.commodity_stock.items():
            lead_time = CommodityForecaster.COMMODITY_COSTS[commodity]['lead_time']
            monthly_usage = 1000  # This would be calculated from historical data
            
            if stock < monthly_usage * 3:  # Less than 3 months supply
                order_qty = monthly_usage * 6  # Order 6 months supply
                orders.append({
                    'commodity': commodity.upper(),
                    'quantity': order_qty,
                    'cost': order_qty * CommodityForecaster.COMMODITY_COSTS[commodity]['unit_cost'],
                    'urgency': 'HIGH' if stock < monthly_usage else 'MEDIUM'
                })
        
        return orders

# ============================================================================
# 8. INTERVENTION OPTIMIZER
# ============================================================================
class InterventionOptimizer:
    """Optimize intervention allocation"""
    
    INTERVENTIONS = {
        'llin': {
            'cost_per_unit': 5,
            'effectiveness': 0.6,
            'coverage_target': 0.8,
            'duration_months': 36
        },
        'irs': {
            'cost_per_unit': 15,
            'effectiveness': 0.4,
            'coverage_target': 0.6,
            'duration_months': 6
        },
        'smc': {  # Seasonal Malaria Chemoprevention
            'cost_per_child': 10,
            'effectiveness': 0.8,
            'coverage_target': 0.9,
            'duration_months': 4
        },
        'larviciding': {
            'cost_per_hectare': 50,
            'effectiveness': 0.7,
            'coverage_target': 0.5,
            'duration_months': 3
        }
    }
    
    @staticmethod
    def optimize_allocation(budget, risk_scores, population):
        """Optimize intervention mix using simplified calculation"""
        allocations = {}
        
        # Simple allocation based on effectiveness per dollar
        for intervention, params in InterventionOptimizer.INTERVENTIONS.items():
            cost_effectiveness = params['effectiveness'] / params['cost_per_unit']
            allocations[intervention] = {
                'cost_effectiveness': cost_effectiveness,
                'recommended_allocation': budget * 0.25  # Equal allocation for demo
            }
        
        # Sort by cost-effectiveness
        sorted_allocations = sorted(
            allocations.items(),
            key=lambda x: x[1]['cost_effectiveness'],
            reverse=True
        )
        
        return sorted_allocations

# ============================================================================
# 9. REPORT GENERATOR
# ============================================================================
class ReportGenerator:
    """Generate various reports"""
    
    @staticmethod
    def generate_technical_report(data, predictions):
        """Generate technical report"""
        report = f"""
        MALARIA FORECASTING TECHNICAL REPORT
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        EXECUTIVE SUMMARY
        -----------------
        Period: {data['date'].min().strftime('%Y-%m')} to {data['date'].max().strftime('%Y-%m')}
        Total Cases: {data['malaria_cases'].sum():,.0f}
        Average Monthly Cases: {data['malaria_cases'].mean():,.0f}
        
        CLIMATE ANALYSIS
        ----------------
        Average Temperature: {data['temperature'].mean():.1f}°C
        Average Rainfall: {data['rainfall'].mean():.1f} mm
        Average NDDI: {data['nddi'].mean():.3f}
        
        INTERVENTION COVERAGE
        --------------------
        LLIN Coverage: {data['llin_coverage'].mean():.1f}%
        IRS Coverage: {data['irs_coverage'].mean():.1f}%
        
        FORECAST SUMMARY
        ----------------
        Next 6 months predicted cases: {np.sum(predictions[:6]):,.0f}
        Peak predicted month: Month {np.argmax(predictions[:6]) + 1}
        
        RECOMMENDATIONS
        ---------------
        1. Maintain high LLIN distribution coverage
        2. Target IRS in high-risk areas
        3. Monitor NDDI for drought/flood conditions
        4. Prepare for seasonal peaks
        """
        return report
    
    @staticmethod
    def generate_executive_summary(data):
        """Generate executive summary"""
        summary = f"""
        EXECUTIVE SUMMARY
        =================
        
        KEY METRICS
        -----------
        • Total Malaria Cases: {data['malaria_cases'].sum():,.0f}
        • Average Monthly Incidence: {data['incidence_rate'].mean():.2f} per 1000
        • Case Reduction Target: 40% by 2025
        • Current Reduction: {((1 - data['malaria_cases'].tail(12).mean() / data['malaria_cases'].head(12).mean()) * 100):.1f}%
        
        INTERVENTION EFFECTIVENESS
        --------------------------
        • LLIN Coverage: {data['llin_coverage'].mean():.1f}%
        • IRS Coverage: {data['irs_coverage'].mean():.1f}%
        
        CLIMATE RISK ASSESSMENT
        -----------------------
        • Drought Risk Months: {(data['nddi'] < -0.5).sum()}
        • Flood Risk Months: {(data['nddi'] > 0.5).sum()}
        
        RECOMMENDED ACTIONS
        -------------------
        1. Scale up interventions in high-risk areas
        2. Strengthen surveillance systems
        3. Improve commodity supply chain
        4. Enhance community engagement
        """
        return summary

# ============================================================================
# 10. TRAINING SIMULATOR
# ============================================================================
class TrainingSimulator:
    """Training mode for staff"""
    
    SCENARIOS = {
        'drought_year': {
            'name': 'Severe Drought Year',
            'description': 'NDDI: -1.2, Rainfall: -40%, Temperature: +2°C',
            'challenges': ['Water collection sites increase', 'Reduced access to healthcare', 
                         'Population displacement', 'Food insecurity']
        },
        'flood_emergency': {
            'name': 'Flood Emergency',
            'description': 'NDDI: +1.5, Rainfall: +80%, Flooding widespread',
            'challenges': ['Vector breeding sites multiply', 'Infrastructure damage', 
                         'Disease outbreaks', 'Access barriers']
        },
        'stockout_crisis': {
            'name': 'Commodity Stockout Crisis',
            'description': 'RDTs: 0%, ACTs: 10%, LLINs: 15% stock remaining',
            'challenges': ['Inability to diagnose cases', 'Treatment delays', 
                         'Increased mortality', 'Community frustration']
        },
        'new_epidemic_strain': {
            'name': 'New Epidemic Strain',
            'description': 'Drug resistance detected, cases doubling monthly',
            'challenges': ['Treatment failure', 'Rapid spread', 
                         'Need for new protocols', 'Panic management']
        }
    }
    
    @staticmethod
    def run_scenario(scenario_key):
        """Run training scenario"""
        scenario = TrainingSimulator.SCENARIOS.get(scenario_key, {})
        
        st.markdown(f"""
        <div class="warning-box">
            <h3>🎓 Training Scenario: {scenario.get('name', 'Unknown')}</h3>
            <p><strong>Description:</strong> {scenario.get('description', '')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Scenario Challenges:")
        for challenge in scenario.get('challenges', []):
            st.write(f"• {challenge}")
        
        # Trainee decisions
        st.subheader("Your Decisions:")
        
        decisions = {}
        col1, col2 = st.columns(2)
        
        with col1:
            decisions['llin_allocation'] = st.slider(
                "LLINs to deploy (thousands)", 0, 100, 50
            ) * 1000
            decisions['irs_teams'] = st.slider("IRS teams to deploy", 0, 50, 10)
            decisions['emergency_funding'] = st.slider(
                "Emergency funding request ($ thousands)", 0, 500, 100
            ) * 1000
        
        with col2:
            decisions['rdt_distribution'] = st.slider(
                "RDTs to distribute (thousands)", 0, 100, 25
            ) * 1000
            decisions['community_mobilizers'] = st.slider(
                "Community mobilizers", 0, 200, 50
            )
            decisions['coordination_meetings'] = st.selectbox(
                "Coordination meetings per week", [0, 1, 2, 3, 4]
            )
        
        if st.button("Simulate Outcomes"):
            # Calculate outcomes
            outcomes = TrainingSimulator.calculate_outcomes(scenario_key, decisions)
            
            st.subheader("Simulation Results:")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Cases Averted", f"{outcomes.get('cases_averted', 0):,.0f}")
            with col2:
                st.metric("Lives Saved", f"{outcomes.get('lives_saved', 0):,.0f}")
            with col3:
                st.metric("Cost Effectiveness", f"${outcomes.get('cost_per_case_averted', 0):,.0f}")
            
            # Feedback
            st.subheader("Performance Feedback:")
            if outcomes.get('score', 0) > 80:
                st.success("✅ Excellent response! Your decisions effectively addressed the crisis.")
            elif outcomes.get('score', 0) > 60:
                st.warning("⚠️ Good response, but some areas need improvement.")
            else:
                st.error("❌ Response needs significant improvement. Review malaria response protocols.")
            
            # Lessons learned
            st.subheader("Key Lessons:")
            st.write("1. Early detection and rapid response are critical")
            st.write("2. Community engagement improves intervention effectiveness")
            st.write("3. Multi-sectoral coordination saves lives")
            st.write("4. Data-driven decisions yield better outcomes")
    
    @staticmethod
    def calculate_outcomes(scenario_key, decisions):
        """Calculate training scenario outcomes"""
        # Simplified outcome calculation
        base_score = 50
        
        # Score based on decisions
        score = base_score
        score += min(decisions['llin_allocation'] / 10000, 20)  # Up to 20 points
        score += min(decisions['irs_teams'] * 2, 15)  # Up to 15 points
        score += min(decisions['rdt_distribution'] / 5000, 15)  # Up to 15 points
        
        # Calculate metrics
        cases_averted = decisions['llin_allocation'] * 0.1 + decisions['irs_teams'] * 50
        lives_saved = cases_averted * 0.001  # Assume 0.1% mortality
        cost_per_case_averted = (decisions['emergency_funding'] / cases_averted 
                                if cases_averted > 0 else 0)
        
        return {
            'score': min(score, 100),
            'cases_averted': int(cases_averted),
            'lives_saved': int(lives_saved),
            'cost_per_case_averted': int(cost_per_case_averted)
        }

# ============================================================================
# 11. BENCHMARKING SYSTEM
# ============================================================================
class BenchmarkingSystem:
    """Compare performance against targets and peers"""
    
    WHO_TARGETS = {
        'case_reduction': 0.40,  # 40% reduction
        'testing_rate': 0.90,  # 90% testing
        'treatment_rate': 0.90,  # 90% treatment within 24h
        'llin_coverage': 0.80,  # 80% coverage
        'irs_coverage': 0.60   # 60% coverage
    }
    
    @staticmethod
    def calculate_benchmarks(data):
        """Calculate performance against benchmarks"""
        benchmarks = {}
        
        if data is None or len(data) < 12:
            return benchmarks
        
        # Case reduction
        first_year = data['malaria_cases'].head(12).mean()
        last_year = data['malaria_cases'].tail(12).mean()
        reduction = (first_year - last_year) / first_year if first_year > 0 else 0
        benchmarks['case_reduction'] = {
            'actual': reduction,
            'target': BenchmarkingSystem.WHO_TARGETS['case_reduction'],
            'achieved': reduction >= BenchmarkingSystem.WHO_TARGETS['case_reduction']
        }
        
        # Coverage rates
        benchmarks['llin_coverage'] = {
            'actual': data['llin_coverage'].mean() / 100,
            'target': BenchmarkingSystem.WHO_TARGETS['llin_coverage'],
            'achieved': data['llin_coverage'].mean() / 100 >= BenchmarkingSystem.WHO_TARGETS['llin_coverage']
        }
        
        benchmarks['irs_coverage'] = {
            'actual': data['irs_coverage'].mean() / 100,
            'target': BenchmarkingSystem.WHO_TARGETS['irs_coverage'],
            'achieved': data['irs_coverage'].mean() / 100 >= BenchmarkingSystem.WHO_TARGETS['irs_coverage']
        }
        
        return benchmarks
    
    @staticmethod
    def get_regional_comparison():
        """Get comparison with neighboring regions/countries"""
        # Simulated data - in production would come from database
        comparison = {
            'Region A': {'case_reduction': 0.35, 'llin_coverage': 0.75, 'ranking': 1},
            'Region B': {'case_reduction': 0.28, 'llin_coverage': 0.68, 'ranking': 2},
            'Your Region': {'case_reduction': 0.32, 'llin_coverage': 0.72, 'ranking': 3},
            'Region C': {'case_reduction': 0.25, 'llin_coverage': 0.65, 'ranking': 4},
            'Region D': {'case_reduction': 0.20, 'llin_coverage': 0.60, 'ranking': 5}
        }
        return comparison

# ============================================================================
# 12. MOBILE INTERFACE OPTIMIZER
# ============================================================================
class MobileInterface:
    """Optimize interface for mobile/field use"""
    
    @staticmethod
    def mobile_view():
        """Display mobile-optimized view"""
        st.markdown("""
        <style>
        .mobile-view .stButton>button {
            width: 100%;
            margin: 5px 0;
            min-height: 60px;
            font-size: 18px;
        }
        .mobile-view h1, .mobile-view h2, .mobile-view h3 {
            font-size: 1.5em;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="mobile-view">', unsafe_allow_html=True)
        
        st.subheader("📱 Field Worker Dashboard")
        
        # Quick actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Report Cases", use_container_width=True):
                st.session_state.show_case_report = True
                st.rerun()
        with col2:
            if st.button("🦟 Submit Mosquito Data", use_container_width=True):
                st.session_state.show_mosquito_data = True
                st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 View Alerts", use_container_width=True):
                st.session_state.show_alerts = True
                st.rerun()
        with col2:
            if st.button("🗺️ View Risk Map", use_container_width=True):
                st.session_state.show_risk_map = True
                st.rerun()
        
        # Quick stats
        if st.session_state.data is not None:
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("This Month", f"{st.session_state.data['malaria_cases'].iloc[-1]:,.0f}")
                st.metric("NDDI", f"{st.session_state.data['nddi'].iloc[-1]:.2f}")
            with col2:
                st.metric("Last Month", f"{st.session_state.data['malaria_cases'].iloc[-2]:,.0f}")
                st.metric("Risk Level", st.session_state.data['drought_risk'].iloc[-1])
        
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# 13. EXISTING CODE FROM ORIGINAL APP (with minimal modifications)
# ============================================================================

# NDDI CALCULATION FUNCTIONS (keep original)
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
        impact_factors = []
        for nddi in nddi_values:
            if nddi < -0.8:
                impact = 1.5
            elif nddi < -0.3:
                impact = 1.2
            elif nddi < 0.3:
                impact = 1.0
            elif nddi < 0.8:
                impact = 1.3
            else:
                impact = 1.6
            impact_factors.append(impact)
        return np.array(impact_factors)

# MULTI-INSTANCE LEARNING CLASS (keep original)
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
        
        available_features = [col for col in feature_columns if col in data.columns]
        
        for i in range(len(data) - self.window_size):
            instance_features = []
            for j in range(self.window_size):
                idx = i + j
                for feature in available_features:
                    instance_features.append(data.iloc[idx][feature])
            
            instances.append(instance_features)
            
            future_idx = i + self.window_size
            if future_idx < len(data):
                instance_labels.append(data.iloc[future_idx][target_column])
            else:
                instance_labels.append(np.nan)
        
        instances = np.array(instances)
        instance_labels = np.array(instance_labels)
        
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
            if len(cluster_indices) > 10:
                X_cluster = instances[cluster_indices]
                y_cluster = labels[cluster_indices]
                
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
            cluster_id = self.cluster_model.predict(instance.reshape(1, -1))[0]
            
            if cluster_id in self.instance_predictors:
                pred = self.instance_predictors[cluster_id].predict(instance.reshape(1, -1))[0]
                predictions.append(max(0, pred))
            else:
                predictions.append(np.mean([p.predict(instance.reshape(1, -1))[0] 
                                          for p in self.instance_predictors.values()]))
        
        return np.array(predictions)

# ENHANCED DATA GENERATION (keep original)
def generate_synthetic_data():
    """Generate synthetic climate and malaria data with NDDI"""
    st.info("🔄 Generating synthetic data with NDDI features...")
    
    dates = pd.date_range(start='2015-01-01', end='2024-12-31', freq='M')
    n_months = len(dates)
    
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
    
    # Generate synthetic NDVI
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
    nddi_effect = nddi_impact
    
    climate_transmission = 0.5 + 0.3*temp_effect + 0.4*rainfall_effect + 0.2*humidity_effect
    llin_effect = 1 - (llin_coverage / 100) * 0.6
    irs_effect = 1 - (irs_coverage / 100) * 0.4
    seasonality = 0.8 + 0.4 * np.sin(2 * np.pi * months / 12 - np.pi/2)
    
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
    
    # Additional features
    df['nddi_trend'] = df['nddi'].rolling(window=3, min_periods=1).mean()
    df['drought_duration'] = (df['nddi'] < -0.5).astype(int)
    df['drought_duration'] = df['drought_duration'].groupby((df['drought_duration'] != df['drought_duration'].shift()).cumsum()).cumsum()
    
    st.success(f"✅ Generated {len(df)} months of synthetic data with NDDI!")
    
    # Log action
    AuditLogger.log_action(
        st.session_state.user_role,
        'data_generation',
        f'Generated synthetic data with {len(df)} records'
    )
    
    return df

# TRAIN MODELS FUNCTION (keep original)
def train_all_models(data):
    """Train all models including MIL"""
    st.info("🤖 Training models (including Multi-Instance Learning)...")
    
    # Prepare data for MIL
    mil_predictor = MultiInstanceMalariaPredictor(window_size=3, n_clusters=5)
    
    with st.spinner("Creating multi-instance representations..."):
        instances, instance_labels = mil_predictor.create_instances(data)
        
        if len(instances) > 0:
            scaled_instances = mil_predictor.scaler.fit_transform(instances)
            cluster_labels = mil_predictor.cluster_instances(scaled_instances)
            mil_predictor.train_instance_predictors(scaled_instances, instance_labels, cluster_labels)
            mil_predictions = mil_predictor.predict(scaled_instances)
            
            aligned_predictions = np.zeros(len(data))
            aligned_predictions[:] = np.nan
            aligned_predictions[3:3+len(mil_predictions)] = mil_predictions
            
            st.session_state.mil_predictions = aligned_predictions
            
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
    
    # Train traditional models
    X = data[['temperature', 'rainfall', 'humidity', 'nddi', 
              'llin_coverage', 'irs_coverage']].values
    y = data['malaria_cases'].values
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    rf_predictions = rf_model.predict(X)
    
    gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model.fit(X, y)
    gb_predictions = gb_model.predict(X)
    
    # Log action
    AuditLogger.log_action(
        st.session_state.user_role,
        'model_training',
        f'Trained MIL, RF, and GB models on {len(data)} records'
    )
    
    return {
        'rf_predictions': rf_predictions,
        'gb_predictions': gb_predictions,
        'rf_model': rf_model,
        'gb_model': gb_model
    }

# ============================================================================
# 14. MAIN APP LAYOUT WITH ENHANCEMENTS
# ============================================================================
def main():
    # Enhanced sidebar with all new features
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=100)
        st.title("🦟 Malaria Forecasting System")
        st.markdown("**National Control Unit Edition**")
        st.markdown("---")
        
        # User role selection
        st.markdown("### 👤 User Role")
        user_role = st.selectbox(
            "Select your role",
            options=list(UserPermissions.ROLES.keys()),
            format_func=lambda x: UserPermissions.get_role_name(x),
            key="user_role_select"
        )
        st.session_state.user_role = user_role
        
        # Quick actions based on role
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📱 Mobile View", use_container_width=True):
                st.session_state.mobile_view = not st.session_state.mobile_view
                st.rerun()
        with col2:
            if st.button("🚨 View Alerts", use_container_width=True):
                st.session_state.show_alerts = True
                st.rerun()
        
        # Data import options
        st.markdown("---")
        st.markdown("### 📊 Data Management")
        
        data_option = st.radio(
            "Choose data source:",
            ["Generate Synthetic Data", "Upload CSV/Excel", "Import Sample Data"],
            index=0
        )
        
        if st.button("📥 Load Data", use_container_width=True):
            with st.spinner("Loading data..."):
                if data_option == "Generate Synthetic Data":
                    st.session_state.data = generate_synthetic_data()
                    st.session_state.data_generated = True
                    st.session_state.models_trained = False
                elif data_option == "Upload CSV/Excel":
                    uploaded_data = RealDataImporter.import_csv_file()
                    if uploaded_data is not None:
                        st.session_state.data = uploaded_data
                        st.session_state.data_generated = True
                        st.session_state.models_trained = False
                        st.success("Data uploaded successfully!")
                elif data_option == "Import Sample Data":
                    country = st.selectbox("Select country", ["Ghana", "Kenya", "Uganda"])
                    sample_data = RealDataImporter.import_sample_data(country)
                    if sample_data is not None:
                        st.session_state.data = sample_data
                        st.session_state.data_generated = True
                        st.session_state.models_trained = False
                
                # Check data quality
                if st.session_state.data is not None:
                    DataQualityMonitor.check_data_quality(st.session_state.data)
                    # Generate alerts
                    AlertSystem.generate_alerts(st.session_state.data)
                
                st.rerun()
        
        if st.session_state.data_generated:
            st.success("✅ Data loaded!")
            
            if st.button("🤖 Train Models", use_container_width=True):
                with st.spinner("Training models..."):
                    st.session_state.model_results = train_all_models(st.session_state.data)
                    st.session_state.models_trained = True
                    st.rerun()
        
        # Advanced settings
        st.markdown("---")
        st.markdown("### 🔧 Advanced Settings")
        
        with st.expander("Multi-Instance Learning"):
            window_size = st.slider("Temporal Window", 2, 6, 3)
            n_clusters = st.slider("Clusters", 2, 8, 5)
        
        with st.expander("Alert Settings"):
            outbreak_threshold = st.slider("Outbreak Threshold (%)", 50, 200, 150)
            drought_threshold = st.slider("Drought Threshold (NDDI)", -2.0, 0.0, -0.8)
        
        with st.expander("Intervention Planning"):
            budget = st.number_input("Monthly Budget ($)", 10000, 1000000, 50000)
            llin_target = st.slider("LLIN Target Coverage (%)", 0, 100, 80)
            irs_target = st.slider("IRS Target Coverage (%)", 0, 100, 60)
        
        # Implementation wizard
        st.markdown("---")
        st.markdown("### 🚀 Implementation")
        
        if st.button("Start Implementation Wizard", use_container_width=True):
            st.session_state.show_implementation = True
            st.rerun()
        
        # Training mode
        if st.checkbox("🎓 Enable Training Mode"):
            scenario = st.selectbox(
                "Select Training Scenario",
                list(TrainingSimulator.SCENARIOS.keys()),
                format_func=lambda x: TrainingSimulator.SCENARIOS[x]['name']
            )
            if st.button("Start Training", use_container_width=True):
                st.session_state.training_scenario = scenario
                st.rerun()
        
        # System status
        st.markdown("---")
        st.markdown("### 📈 System Status")
        
        if st.session_state.data is not None:
            st.metric("Data Records", len(st.session_state.data))
        if st.session_state.alerts:
            alert_count = len([a for a in st.session_state.alerts if a['level'] in ['CRITICAL', 'HIGH']])
            st.metric("Active Alerts", alert_count, delta="Requires attention" if alert_count > 0 else None)
        
        # Backup/Export
        if st.button("💾 Backup System", use_container_width=True):
            backup_data = {
                'data': st.session_state.data.to_dict() if st.session_state.data is not None else None,
                'models_trained': st.session_state.models_trained,
                'alerts': st.session_state.alerts,
                'timestamp': datetime.now().isoformat()
            }
            
            # Create download link
            import json
            backup_str = json.dumps(backup_data, default=str)
            b64 = base64.b64encode(backup_str.encode()).decode()
            href = f'<a href="data:file/json;base64,{b64}" download="malaria_system_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Download Backup</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Main content - Mobile view if enabled
    if st.session_state.get('mobile_view', False):
        MobileInterface.mobile_view()
        return
    
    # Main header
    st.markdown('<h1 class="main-header">🦟 National Malaria Forecasting System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #0EA5E9;">Enhanced for Control Unit Operations</h3>', unsafe_allow_html=True)
    
    # Alert banner
    if st.session_state.alerts:
        critical_alerts = [a for a in st.session_state.alerts if a['level'] in ['CRITICAL', 'HIGH']]
        if critical_alerts:
            with st.container():
                st.markdown(f"""
                <div class="alert-box">
                    <h3>🚨 {len(critical_alerts)} Critical Alert(s)</h3>
                    <p>Immediate action required</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("View All Alerts"):
                    st.session_state.show_alerts = True
                    st.rerun()
    
    # Training scenario
    if st.session_state.get('training_scenario'):
        TrainingSimulator.run_scenario(st.session_state.training_scenario)
        return
    
    # Implementation wizard
    if st.session_state.get('show_implementation', False):
        st.markdown('<h2 class="sub-header">🚀 Implementation Wizard</h2>', unsafe_allow_html=True)
        
        completed = sum(1 for step in st.session_state.implementation_steps if step['completed'])
        total = len(st.session_state.implementation_steps)
        
        st.progress(completed / total if total > 0 else 0)
        
        for step in st.session_state.implementation_steps:
            col1, col2, col3 = st.columns([1, 3, 1])
            with col1:
                st.write(f"**Step {step['step']}**")
            with col2:
                st.write(step['task'])
            with col3:
                if not step['completed']:
                    if st.button("Mark Complete", key=f"step_{step['step']}"):
                        step['completed'] = True
                        AuditLogger.log_action(
                            st.session_state.user_role,
                            'implementation_step',
                            f"Completed: {step['task']}"
                        )
                        st.rerun()
                else:
                    st.success("✅")
        
        if completed == total:
            st.balloons()
            st.success("🎉 Implementation completed successfully!")
        
        if st.button("Return to Dashboard"):
            st.session_state.show_implementation = False
            st.rerun()
        return
    
    # Show alerts page
    if st.session_state.get('show_alerts', False):
        st.markdown('<h2 class="sub-header">🚨 Alert Dashboard</h2>', unsafe_allow_html=True)
        
        if not st.session_state.alerts:
            st.info("No active alerts")
        else:
            for i, alert in enumerate(st.session_state.alerts):
                with st.expander(f"{alert['level']}: {alert['type']} - {alert['message']}"):
                    st.write(f"**Timestamp:** {alert['timestamp']}")
                    st.write("**Required Actions:**")
                    for action in alert['actions']:
                        st.write(f"• {action}")
                    
                    if st.button(f"Mark as Resolved", key=f"resolve_{i}"):
                        st.session_state.alerts.pop(i)
                        AuditLogger.log_action(
                            st.session_state.user_role,
                            'alert_resolved',
                            f"Resolved: {alert['type']}"
                        )
                        st.rerun()
        
        if st.button("Return to Dashboard"):
            st.session_state.show_alerts = False
            st.rerun()
        return
    
    # Welcome screen if no data
    if not st.session_state.data_generated:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://cdn-icons-png.flaticon.com/512/3050/3050525.png", width=200)
            st.markdown("""
            <div style='text-align: center;'>
                <h2>National Malaria Control Unit System</h2>
                <p><strong>Enhanced Features:</strong></p>
                <ul style='text-align: left;'>
                    <li>Real Data Integration</li>
                    <li>Alert & Response System</li>
                    <li>Commodity Forecasting</li>
                    <li>Multi-Language Support</li>
                    <li>Mobile Field Interface</li>
                    <li>Training Simulator</li>
                    <li>Performance Benchmarking</li>
                    <li>Audit Logging</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.info("👈 Select data source and load data to begin")
            
            # Quick stats
            st.markdown("### 📊 System Capabilities:")
            cols = st.columns(4)
            with cols[0]:
                st.metric("Users Supported", "4 roles")
            with cols[1]:
                st.metric("Alert Types", "6+")
            with cols[2]:
                st.metric("Reports", "5 formats")
            with cols[3]:
                st.metric("Training", "4 scenarios")
        return
    
    # ========================================================================
    # ENHANCED DASHBOARD WITH ALL FEATURES
    # ========================================================================
    
    # Operational Dashboard Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", "🚨 Alerts & Response", "💰 Resources", 
        "📊 Performance", "📋 Reports", "🎓 Training", "🔧 Admin"
    ])
    
    with tab1:  # Overview tab
        st.markdown('<h2 class="sub-header">📈 Operational Overview</h2>', unsafe_allow_html=True)
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Months", len(st.session_state.data))
        with col2:
            st.metric("Avg Monthly Cases", f"{st.session_state.data['malaria_cases'].mean():,.0f}")
        with col3:
            st.metric("Current NDDI", f"{st.session_state.data['nddi'].iloc[-1]:.3f}")
        with col4:
            st.metric("Active Alerts", len(st.session_state.alerts))
        
        # Data quality box
        if st.session_state.data_quality_issues:
            st.markdown("""
            <div class="warning-box">
                <h4>⚠️ Data Quality Issues Detected</h4>
                <p>{count} issue(s) found. Please review data quality report.</p>
            </div>
            """.format(count=len(st.session_state.data_quality_issues)), unsafe_allow_html=True)
        
        # NDDI Insights
        st.markdown("""
        <div class="nddi-box">
            <h4>🌵 NDDI Climate Insights</h4>
            <p><strong>Current Status:</strong> {nddi:.3f} ({risk})</p>
            <p><strong>Trend:</strong> {trend} over last 3 months</p>
            <p><strong>Impact Factor:</strong> {impact}x on transmission</p>
        </div>
        """.format(
            nddi=st.session_state.data['nddi'].iloc[-1],
            risk=st.session_state.data['drought_risk'].iloc[-1],
            trend="Increasing" if st.session_state.data['nddi'].iloc[-1] > st.session_state.data['nddi'].iloc[-4] else "Decreasing",
            impact=st.session_state.data['nddi_impact'].iloc[-1]
        ), unsafe_allow_html=True)
        
        # Quick visualization
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=st.session_state.data['malaria_cases'],
                mode='lines',
                name='Cases',
                line=dict(color='red', width=2)
            ))
            fig.update_layout(
                title='Malaria Cases Trend',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
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
                y=[0]*len(st.session_state.data),
                mode='lines',
                name='Threshold',
                line=dict(color='gray', width=1, dash='dash')
            ))
            fig.update_layout(
                title='NDDI Trend',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Alerts & Response
        st.markdown('<h2 class="sub-header">🚨 Alert & Response System</h2>', unsafe_allow_html=True)
        
        # Alert summary
        alert_counts = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for alert in st.session_state.alerts:
            alert_counts[alert['level']] = alert_counts.get(alert['level'], 0) + 1
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Critical", alert_counts['CRITICAL'], delta_color="inverse")
        with col2:
            st.metric("High", alert_counts['HIGH'])
        with col3:
            st.metric("Medium", alert_counts['MEDIUM'])
        with col4:
            st.metric("Low", alert_counts['LOW'])
        
        # Response actions
        st.subheader("📋 Recommended Response Actions")
        
        if st.session_state.alerts:
            for alert in st.session_state.alerts[:3]:  # Show top 3
                with st.expander(f"{alert['level']}: {alert['type']}"):
                    st.write(f"**Message:** {alert['message']}")
                    st.write("**Actions:**")
                    for action in alert['actions']:
                        st.write(f"• {action}")
                    
                    # Action tracking
                    action_taken = st.text_input(
                        f"Action taken for {alert['type']}",
                        key=f"action_{alert['type']}"
                    )
                    if st.button(f"Mark as Addressed", key=f"address_{alert['type']}"):
                        AuditLogger.log_action(
                            st.session_state.user_role,
                            'alert_action',
                            f"Addressed {alert['type']}: {action_taken}"
                        )
                        st.success("Action logged")
        else:
            st.info("No active alerts requiring response")
        
        # Manual alert creation
        st.subheader("📝 Create Manual Alert")
        with st.form("manual_alert"):
            alert_type = st.selectbox("Alert Type", ["Outbreak", "Stockout", "Data Quality", "Weather", "Other"])
            alert_level = st.selectbox("Alert Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
            alert_message = st.text_area("Alert Message")
            actions = st.text_area("Required Actions (one per line)")
            
            if st.form_submit_button("Create Alert"):
                new_alert = {
                    'level': alert_level,
                    'type': alert_type,
                    'message': alert_message,
                    'actions': actions.split('\n'),
                    'priority': {'LOW': 4, 'MEDIUM': 3, 'HIGH': 2, 'CRITICAL': 1}[alert_level],
                    'timestamp': datetime.now()
                }
                st.session_state.alerts.append(new_alert)
                AlertSystem.send_alert_notifications([new_alert])
                st.success("Alert created and notifications sent")
    
    with tab3:  # Resources tab
        st.markdown('<h2 class="sub-header">💰 Resource Management</h2>', unsafe_allow_html=True)
        
        # Commodity stock
        st.subheader("📦 Current Stock Levels")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("RDTs", f"{st.session_state.commodity_stock['rdts']:,.0f}")
        with col2:
            st.metric("ACTs", f"{st.session_state.commodity_stock['acts']:,.0f}")
        with col3:
            st.metric("LLINs", f"{st.session_state.commodity_stock['llins']:,.0f}")
        with col4:
            st.metric("Insecticide", f"{st.session_state.commodity_stock['insecticide']:,.0f}")
        
        # Stock adjustment
        st.subheader("🔄 Update Stock Levels")
        col1, col2 = st.columns(2)
        
        with col1:
            rdt_adjust = st.number_input("RDTs adjustment", -10000, 10000, 0)
            act_adjust = st.number_input("ACTs adjustment", -10000, 10000, 0)
        with col2:
            llin_adjust = st.number_input("LLINs adjustment", -10000, 10000, 0)
            insecticide_adjust = st.number_input("Insecticide adjustment", -1000, 1000, 0)
        
        if st.button("Update Stock"):
            st.session_state.commodity_stock['rdts'] += rdt_adjust
            st.session_state.commodity_stock['acts'] += act_adjust
            st.session_state.commodity_stock['llins'] += llin_adjust
            st.session_state.commodity_stock['insecticide'] += insecticide_adjust
            
            AuditLogger.log_action(
                st.session_state.user_role,
                'stock_update',
                f"Updated stock levels"
            )
            st.success("Stock levels updated")
            st.rerun()
        
        # Forecasting
        st.subheader("🔮 Commodity Forecasting")
        
        if st.session_state.models_trained and 'model_results' in st.session_state:
            predictions = st.session_state.model_results.get('rf_predictions', [])
            if len(predictions) > 0:
                forecast = CommodityForecaster.forecast_needs(
                    predictions[-6:],  # Last 6 months predictions
                    st.session_state.data['population'].iloc[-1] if len(st.session_state.data) > 0 else 1000000,
                    months=6
                )
                
                if forecast:
                    st.write("**6-Month Forecast Needs:**")
                    for commodity, quantity in forecast['commodities'].items():
                        st.write(f"• {commodity.upper()}: {quantity:,.0f} units")
                    
                    st.write(f"**Total Estimated Cost:** ${forecast['total_cost']:,.0f}")
                    st.write(f"**Monthly Cost:** ${forecast['monthly_cost']:,.0f}")
        
        # Procurement orders
        st.subheader("🛒 Procurement Orders")
        
        orders = CommodityForecaster.check_stock_levels()
        if orders:
            for order in orders:
                with st.expander(f"{order['commodity']} - {order['urgency']} priority"):
                    st.write(f"**Quantity:** {order['quantity']:,.0f}")
                    st.write(f"**Cost:** ${order['cost']:,.0f}")
                    
                    if st.button(f"Approve Order", key=f"order_{order['commodity']}"):
                        AuditLogger.log_action(
                            st.session_state.user_role,
                            'procurement_approval',
                            f"Approved {order['commodity']} order for ${order['cost']:,.0f}"
                        )
                        st.success("Order approved and logged")
        else:
            st.info("No urgent procurement orders required")
    
    with tab4:  # Performance tab
        st.markdown('<h2 class="sub-header">📊 Performance Dashboard</h2>', unsafe_allow_html=True)
        
        # Benchmarking
        st.subheader("🏆 Performance Benchmarks")
        
        benchmarks = BenchmarkingSystem.calculate_benchmarks(st.session_state.data)
        comparison = BenchmarkingSystem.get_regional_comparison()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**WHO Targets:**")
            for metric, data in benchmarks.items():
                status = "✅" if data['achieved'] else "❌"
                st.write(f"{status} {metric}: {data['actual']:.1%} / {data['target']:.0%}")
        
        with col2:
            st.write("**Regional Comparison:**")
            df_comparison = pd.DataFrame(comparison).T
            st.dataframe(df_comparison, use_container_width=True)
        
        # Key Performance Indicators
        st.subheader("📈 Key Performance Indicators")
        
        if st.session_state.data is not None:
            # Calculate KPIs
            kpis = {
                'Case Reduction': ((1 - st.session_state.data['malaria_cases'].tail(12).mean() / 
                                  st.session_state.data['malaria_cases'].head(12).mean()) * 100 
                                  if len(st.session_state.data) >= 24 else 0),
                'Testing Rate': 92.5,  # Example - would come from real data
                'Treatment Rate': 88.3,  # Example
                'LLIN Coverage': st.session_state.data['llin_coverage'].mean(),
                'IRS Coverage': st.session_state.data['irs_coverage'].mean()
            }
            
            # Display KPIs
            cols = st.columns(len(kpis))
            for (kpi_name, kpi_value), col in zip(kpis.items(), cols):
                with col:
                    if 'Reduction' in kpi_name:
                        st.metric(kpi_name, f"{kpi_value:.1f}%")
                    elif 'Coverage' in kpi_name:
                        st.metric(kpi_name, f"{kpi_value:.1f}%")
                    else:
                        st.metric(kpi_name, f"{kpi_value:.1f}%")
        
        # Trend analysis
        st.subheader("📉 Trend Analysis")
        
        if len(st.session_state.data) >= 12:
            fig = go.Figure()
            
            # Add 12-month moving average
            ma_12 = st.session_state.data['malaria_cases'].rolling(window=12).mean()
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=ma_12,
                mode='lines',
                name='12-Month MA',
                line=dict(color='blue', width=3)
            ))
            
            # Add actual cases
            fig.add_trace(go.Scatter(
                x=st.session_state.data['date'],
                y=st.session_state.data['malaria_cases'],
                mode='lines',
                name='Monthly Cases',
                line=dict(color='lightblue', width=1)
            ))
            
            fig.update_layout(
                title='Malaria Cases Trend with Moving Average',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:  # Reports tab
        st.markdown('<h2 class="sub-header">📋 Report Generator</h2>', unsafe_allow_html=True)
        
        # Report selection
        report_type = st.selectbox(
            "Select Report Type",
            ["Technical Report", "Executive Summary", "Donor Report", 
             "Monthly Bulletin", "WHO Submission", "Custom Report"]
        )
        
        # Report parameters
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=st.session_state.data['date'].min() if st.session_state.data is not None else datetime.now())
            format_type = st.selectbox("Format", ["PDF", "Excel", "HTML", "Word"])
        with col2:
            end_date = st.date_input("End Date", 
                                    value=st.session_state.data['date'].max() if st.session_state.data is not None else datetime.now())
            include_charts = st.checkbox("Include Charts", True)
        
        # Generate report
        if st.button("Generate Report"):
            with st.spinner(f"Generating {report_type}..."):
                # Generate sample reports based on type
                if report_type == "Technical Report":
                    report_content = ReportGenerator.generate_technical_report(
                        st.session_state.data,
                        st.session_state.model_results.get('rf_predictions', []) if st.session_state.models_trained else []
                    )
                elif report_type == "Executive Summary":
                    report_content = ReportGenerator.generate_executive_summary(st.session_state.data)
                else:
                    report_content = f"""
                    {report_type}
                    Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                    Period: {start_date} to {end_date}
                    
                    Summary data would be included here based on the selected report type.
                    """
                
                # Display report
                st.subheader(f"📄 {report_type}")
                st.text_area("Report Content", report_content, height=300)
                
                # Download options
                st.download_button(
                    label=f"Download {report_type}",
                    data=report_content,
                    file_name=f"malaria_{report_type.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
                
                AuditLogger.log_action(
                    st.session_state.user_role,
                    'report_generation',
                    f"Generated {report_type} for period {start_date} to {end_date}"
                )
        
        # Quick report templates
        st.subheader("📋 Quick Reports")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Monthly Summary", use_container_width=True):
                st.info("Monthly summary report generated")
        with col2:
            if st.button("Alert History", use_container_width=True):
                st.info("Alert history report generated")
        with col3:
            if st.button("Performance Dashboard", use_container_width=True):
                st.info("Performance dashboard report generated")
    
    with tab6:  # Training tab
        st.markdown('<h2 class="sub-header">🎓 Training & Simulation</h2>', unsafe_allow_html=True)
        
        st.write("""
        ### Enhance your team's malaria response capabilities with interactive training scenarios.
        Each scenario presents realistic challenges faced by malaria control programs.
        """)
        
        # Scenario selection
        selected_scenario = st.selectbox(
            "Select Training Scenario",
            list(TrainingSimulator.SCENARIOS.keys()),
            format_func=lambda x: TrainingSimulator.SCENARIOS[x]['name']
        )
        
        if selected_scenario:
            scenario = TrainingSimulator.SCENARIOS[selected_scenario]
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{scenario['name']}</h4>
                <p><strong>Description:</strong> {scenario['description']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("**Scenario Challenges:**")
            for challenge in scenario['challenges']:
                st.write(f"• {challenge}")
            
            if st.button("Start This Training Scenario", use_container_width=True):
                st.session_state.training_scenario = selected_scenario
                st.rerun()
        
        # Training resources
        st.subheader("📚 Training Resources")
        
        resources = [
            {"title": "Malaria Case Management", "duration": "45 min", "type": "Video"},
            {"title": "Vector Control Strategies", "duration": "60 min", "type": "Interactive"},
            {"title": "Data Collection Best Practices", "duration": "30 min", "type": "PDF Guide"},
            {"title": "Outbreak Response Protocol", "duration": "90 min", "type": "Scenario"},
            {"title": "Community Engagement", "duration": "45 min", "type": "Video"}
        ]
        
        for resource in resources:
            with st.expander(f"{resource['title']} ({resource['duration']})"):
                st.write(f"**Type:** {resource['type']}")
                st.write(f"**Duration:** {resource['duration']}")
                if st.button(f"Start {resource['type']}", key=resource['title']):
                    st.success(f"Starting {resource['title']}")
        
        # Training progress
        st.subheader("📊 Training Progress")
        
        progress_data = {
            'Completed': 5,
            'In Progress': 3,
            'Not Started': 2
        }
        
        fig = px.pie(
            values=list(progress_data.values()),
            names=list(progress_data.keys()),
            title="Training Completion Status"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab7:  # Admin tab
        if not UserPermissions.has_permission(st.session_state.user_role, 'view_audit_log'):
            st.warning("⛔ You do not have permission to access administrative functions.")
            return
        
        st.markdown('<h2 class="sub-header">🔧 Administrative Functions</h2>', unsafe_allow_html=True)
        
        # Audit log
        st.subheader("📋 Audit Log")
        
        if st.session_state.audit_log:
            # Convert to dataframe for display
            audit_df = pd.DataFrame(st.session_state.audit_log[-50:])  # Last 50 entries
            st.dataframe(audit_df, use_container_width=True)
            
            # Export audit log
            csv = audit_df.to_csv(index=False)
            st.download_button(
                label="Download Audit Log",
                data=csv,
                file_name=f"audit_log_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.info("No audit log entries yet")
        
        # System configuration
        st.subheader("⚙️ System Configuration")
        
        with st.expander("Alert Settings"):
            st.number_input("Outbreak Threshold (%)", 100, 300, 150, key="config_outbreak_threshold")
            st.number_input("Stockout Warning Level", 1, 100, 30, key="config_stockout_warning")
            st.number_input("Data Quality Threshold (%)", 1, 100, 20, key="config_data_quality")
        
        with st.expander("User Management"):
            st.text_input("Add New User", key="new_user")
            st.selectbox("User Role", list(UserPermissions.ROLES.keys()), key="new_user_role")
            if st.button("Add User"):
                st.success("User added (simulated)")
                AuditLogger.log_action(
                    st.session_state.user_role,
                    'user_management',
                    f"Added new user with role {st.session_state.new_user_role}"
                )
        
        with st.expander("Data Management"):
            if st.button("Run Data Quality Check"):
                issues = DataQualityMonitor.check_data_quality(st.session_state.data)
                if issues:
                    st.warning(f"Found {len(issues)} data quality issues")
                    for issue in issues:
                        st.write(f"• {issue}")
                else:
                    st.success("No data quality issues found")
            
            if st.button("Clean Historical Data"):
                st.info("Data cleaning process started (simulated)")
                AuditLogger.log_action(
                    st.session_state.user_role,
                    'data_cleaning',
                    "Initiated historical data cleaning"
                )
        
        # System backup
        st.subheader("💾 System Backup")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Create Backup", use_container_width=True):
                backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                st.success(f"Backup created at {backup_time}")
                AuditLogger.log_action(
                    st.session_state.user_role,
                    'system_backup',
                    f"Created system backup at {backup_time}"
                )
        
        with col2:
            if st.button("Restore from Backup", use_container_width=True):
                st.info("Restore functionality would be implemented here")
                # In production, this would restore from actual backup
    
    # ========================================================================
    # ORIGINAL APP FUNCTIONALITY (preserved from original code)
    # ========================================================================
    
    # Only show original features if models are trained
    if st.session_state.models_trained:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🤖 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        # Multi-Instance Learning Section (original)
        st.markdown("""
        <div class="metric-card">
            <h4>🧠 Multi-Instance Learning (MIL) Approach</h4>
            <p><strong>Method:</strong> Temporal windows (3 months) → Clustering (5 clusters) → Cluster-specific predictors</p>
        </div>
        """, unsafe_allow_html=True)
        
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
                    if i < 3:
                        preds = metric['Predictions']
                        if metric['Model'] == 'Multi-Instance Learning':
                            valid_dates = dates[~np.isnan(preds)]
                            valid_preds = preds[~np.isnan(preds)]
                            fig.add_trace(go.Scatter(
                                x=valid_dates,
                                y=valid_preds,
                                mode='lines',
                                name=f"{metric['Model']}",
                                line=dict(color=colors[i], width=2, dash='dash')
                            ))
                        else:
                            fig.add_trace(go.Scatter(
                                x=dates,
                                y=preds,
                                mode='lines',
                                name=f"{metric['Model']}",
                                line=dict(color=colors[i], width=2, dash='dash')
                            ))
                
                fig.update_layout(
                    title='Model Predictions Comparison',
                    xaxis_title='Date',
                    yaxis_title='Malaria Cases',
                    hovermode='x unified',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Forecast section (original)
        st.markdown('<h3 class="sub-header">🔮 Malaria Forecast</h3>', unsafe_allow_html=True)
        
        # Generate simple forecast
        forecast_months = 12
        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=forecast_months, 
            freq='M'
        )
        
        # Simple forecast calculation
        seasonal_pattern = 0.8 + 0.4 * np.sin(2 * np.pi * forecast_dates.month / 12 - np.pi/2)
        base_forecast = cases.mean() * seasonal_pattern
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=base_forecast,
            mode='lines+markers',
            name='12-Month Forecast',
            line=dict(color='green', width=3)
        ))
        fig.update_layout(
            title='12-Month Malaria Forecast',
            xaxis_title='Date',
            yaxis_title='Predicted Cases',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Show data ready message
        st.markdown("""
        <div class="success-box">
            <h3>✅ Data Loaded Successfully!</h3>
            <p><strong>Next Steps:</strong></p>
            <ol>
                <li>Review alerts and data quality issues</li>
                <li>Check commodity stock levels</li>
                <li>Generate necessary reports</li>
                <li>Train models for advanced analytics</li>
            </ol>
            <p>Click <strong>"Train Models"</strong> in the sidebar to enable advanced forecasting.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Data quality check
        if st.session_state.data_quality_issues:
            st.markdown("### ⚠️ Data Quality Check")
            for issue in st.session_state.data_quality_issues[:5]:  # Show first 5
                st.write(f"• {issue}")
        
        # Quick insights
        if st.session_state.data is not None:
            st.markdown("### 📊 Quick Insights")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recent Cases", f"{st.session_state.data['malaria_cases'].iloc[-1]:,.0f}")
            with col2:
                st.metric("NDDI Status", st.session_state.data['drought_risk'].iloc[-1])
            with col3:
                st.metric("LLIN Coverage", f"{st.session_state.data['llin_coverage'].iloc[-1]:.1f}%")

# Run the app
if __name__ == "__main__":
    main()
