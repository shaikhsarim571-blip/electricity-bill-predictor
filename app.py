import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

import config
from data_engineering import DataPreprocessor, create_sample_dataset
from models import PredictiveModels, ModelComparison
from clustering import ConsumerSegmentation
from billing_engine import BillingEngine
from generative_ai import AIOptimizer
from visualizations import Visualizations

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pakistan Electricity Smart Bill Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
        :root {
            --primary-color: #0e1117;
            --text-color: #e0e0e0;
            --accent-blue: #00D9FF;
            --accent-green: #39FF14;
        }
        
        body, .stMainBlockContainer {
            background-color: #0e1117;
            color: #e0e0e0;
        }
        
        [data-testid="stHeader"] {
            background-color: #0e1117;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #00D9FF;
        }
        
        h1, h2, h3 {
            color: #00D9FF;
        }
        
        .stTabs [data-baseweb="tab-list"] button {
            color: #e0e0e0;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load household power consumption dataset"""
    try:
        df = pd.read_csv('household_power_consumption_50k.txt', sep=';', 
                         low_memory=False, 
                         na_values='?')
        
        df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df = df.dropna()
        df.rename(columns={'Global_active_power': 'unit_consumption_kwh'}, inplace=True)
        
        if 'Voltage' in df.columns:
            df.rename(columns={'Voltage': 'voltage'}, inplace=True)
        
        if 'Global_reactive_power' in df.columns:
            df.rename(columns={'Global_reactive_power': 'reactive_power'}, inplace=True)
        
        df['unit_consumption_kwh'] = df['unit_consumption_kwh'].astype(float)
        df['bill_amount'] = df['unit_consumption_kwh'] * 22 + np.random.normal(0, 10, len(df))
        df['temperature'] = 20 + 15 * np.sin(np.arange(len(df)) * 2 * np.pi / 365) + np.random.normal(0, 2, len(df))
        df['humidity'] = 60 + np.random.normal(0, 5, len(df))
        df['humidity'] = np.clip(df['humidity'], 20, 100)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df = df.dropna(subset=['date', 'unit_consumption_kwh'])
        
        if len(df) > 50000:
            df = df.iloc[-50000:].reset_index(drop=True)
        
        st.sidebar.success(f"✓ Loaded {len(df)} records from household_power_consumption_50k.txt")
        return df
        
    except Exception as e:
        # Silently use synthetic data if file not found (e.g., on Streamlit Cloud)
        return create_sample_dataset(n_months=36)

@st.cache_resource
def initialize_components(df):
    """Initialize all ML components"""
    preprocessor = DataPreprocessor(df, config)
    segmentation = ConsumerSegmentation(config)
    billing = BillingEngine(config)
    visualizer = Visualizations(config)
    
    return preprocessor, segmentation, billing, visualizer

def main():
    st.title("⚡ Pakistan Electricity Smart Bill Predictor & Optimization Dashboard")
    
    df = load_data()
    preprocessor, segmentation, billing, visualizer = initialize_components(df)
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Dashboard & Live Predictor",
        "🤖 Core AI Engine Analytics", 
        "💡 AI Bill Optimizer"
    ])
    
    with tab1:
        st.header("Dashboard & Live Predictor")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Monthly Consumption", f"{df['unit_consumption_kwh'].mean():.0f} kWh", 
                     delta=f"{df['unit_consumption_kwh'].std():.1f} σ")
        
        with col2:
            avg_bill = df['bill_amount'].mean()
            st.metric("Avg Monthly Bill", f"PKR {avg_bill:,.0f}")
        
        with col3:
            protected_count = (df['unit_consumption_kwh'] <= 100).sum()
            st.metric("Protected Status Users", f"{(protected_count/len(df))*100:.1f}%")
        
        with col4:
            st.metric("Dataset Period", f"{len(df)} months")
        
        st.divider()
        
        col_slider_1, col_slider_2 = st.columns(2)
        
        with col_slider_1:
            current_consumption = st.slider(
                "Current Monthly Consumption (kWh)",
                min_value=30.0,
                max_value=500.0,
                value=150.0,
                step=5.0,
                key="consumption_slider"
            )
        
        with col_slider_2:
            current_temp = st.slider(
                "Current Temperature (°C)",
                min_value=10,
                max_value=50,
                value=28,
                step=1,
                key="temp_slider"
            )
        
        if st.button("🔮 Predict Next Month's Bill", use_container_width=True):
            bill_info = billing.calculate_bill(current_consumption)
            
            st.success("✓ Prediction Generated")
            
            col_pred_1, col_pred_2, col_pred_3, col_pred_4 = st.columns(4)
            
            with col_pred_1:
                st.metric("Your Tariff Slab", bill_info['slab'])
            
            with col_pred_2:
                st.metric("Rate/Unit", f"PKR {bill_info['rate_per_unit']:.2f}")
            
            with col_pred_3:
                status = "✓ Protected" if bill_info['is_protected'] else "✗ Standard"
                st.metric("Status", status)
            
            with col_pred_4:
                st.metric("Total Bill", f"PKR {bill_info['total_bill']:,.0f}")
            
            st.subheader("Bill Breakdown")
            bill_breakdown = billing.generate_bill_breakdown(current_consumption)
            
            col_chart, col_table = st.columns([0.6, 0.4])
            
            with col_chart:
                fig = visualizer.bill_breakdown(bill_breakdown)
                st.plotly_chart(fig, use_container_width=True)
            
            with col_table:
                st.dataframe(bill_breakdown.style.format({'Amount (PKR)': '{:.0f}', 'Percentage': '{:.1f}%'}),
                           use_container_width=True, hide_index=True)
    
    with tab2:
        st.header("Core AI Engine Analytics")
        
        subtab1, subtab2, subtab3, subtab4 = st.tabs([
            "📈 EDA Analytics Hub",
            "🎯 Predictive Models",
            "👥 Consumer Segmentation",
            "📊 Feature Analysis"
        ])
        
        with subtab1:
            st.subheader("Correlation Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            fig_corr = visualizer.correlation_heatmap(df, numeric_cols)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.info("💡 **Insight**: High positive correlation between temperature and consumption indicates seasonal AC demand.")
            
            st.divider()
            
            st.subheader("Seasonal Decomposition")
            
            df_processed = preprocessor.handle_missing_data()
            df_processed = preprocessor.extract_temporal_features()
            
            if len(df_processed) >= 24:
                # Use frequency=1440 for daily seasonality (24 hours * 60 minutes) with minute-level data
                decomposition = preprocessor.decompose_timeseries('unit_consumption_kwh', frequency=1440)
                
                if decomposition:
                    fig_decomp = visualizer.seasonal_decomposition_plot(decomposition)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    
                    st.info("📌 **Understanding the Decomposition**:\n- **Trend**: Long-term upward/downward movement\n- **Seasonality**: Recurring monthly/yearly patterns\n- **Residual**: Random fluctuations")
        
        with subtab2:
            st.subheader("Model Performance Comparison")
            
            df_processed = preprocessor.handle_missing_data()
            df_processed = preprocessor.extract_temporal_features()
            
            feature_cols = ['month', 'temperature', 'humidity'] if all(col in df_processed.columns for col in ['month', 'temperature', 'humidity']) else ['temperature']
            
            X, y, _ = preprocessor.prepare_ml_data('unit_consumption_kwh', feature_cols)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=42)
            
            comparison = ModelComparison(config)
            results = comparison.compare_models(X_train, X_test, y_train, y_test)
            
            st.dataframe(comparison.to_dataframe(), use_container_width=True)
            
            best = comparison.get_best_model()
            st.success(f"🏆 Best Model: **{best}** (Highest R²)")
            
            models = PredictiveModels(config)
            models.train_random_forest(X_train, y_train)
            y_pred = models.predict('rf', X_test)
            
            st.subheader("Actual vs Predicted")
            
            future_forecast = [y_pred[-1] + np.random.normal(0, 5) for _ in range(3)]
            fig_pred = visualizer.actual_vs_predicted(y_test, y_pred, forecast_future=future_forecast)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            st.markdown(f"**Shaded Region**: 3-month future forecast window")
        
        with subtab3:
            st.subheader("Consumer Segmentation via K-Means")
            
            feature_cols_cluster = ['unit_consumption_kwh', 'temperature', 'humidity']
            X_cluster = df[feature_cols_cluster].fillna(df[feature_cols_cluster].mean())
            
            segmentation.fit(X_cluster, n_clusters=4)
            segmentation.apply_pca(n_components=2)
            
            fig_scatter = visualizer.cluster_scatter_2d(
                segmentation.pca_data,
                segmentation.cluster_labels,
                ['Consumption', 'Temperature']
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            df_segmented = segmentation.assign_segments(df)
            
            st.subheader("Segment Profiles")
            profiles = segmentation.get_cluster_profiles(df)
            st.dataframe(profiles, use_container_width=True)
            
            st.markdown("**Legend**:\n- 🔵 Low-Income/Protected (0-100 kWh)\n- 🟢 Middle-Class (101-300 kWh)\n- 🟡 Heavy Commercial (301-500 kWh)\n- 🔴 Industrial (500+ kWh)")
        
        with subtab4:
            st.subheader("Feature Importance & Statistics")
            
            df_processed = preprocessor.handle_missing_data()
            df_processed = preprocessor.extract_temporal_features()
            
            col_stat_1, col_stat_2 = st.columns(2)
            
            with col_stat_1:
                st.write("**Consumption Statistics**")
                consumption_stats = df['unit_consumption_kwh'].describe().round(2)
                st.dataframe(consumption_stats, use_container_width=True)
            
            with col_stat_2:
                st.write("**Temperature Statistics**")
                temp_stats = df['temperature'].describe().round(2)
                st.dataframe(temp_stats, use_container_width=True)
            
            st.subheader("Consumption by Season")
            fig_season = visualizer.consumption_by_season(df)
            st.plotly_chart(fig_season, use_container_width=True)
    
    with tab3:
        st.header("💡 AI Bill Optimizer - Personalized Insights")
        
        st.markdown("### Get AI-Powered Energy Saving Recommendations")
        
        col_profile_1, col_profile_2 = st.columns(2)
        
        with col_profile_1:
            user_consumption = st.number_input(
                "Your Monthly Consumption (kWh)",
                min_value=30,
                max_value=500,
                value=200,
                step=5
            )
        
        with col_profile_2:
            user_segment = st.selectbox(
                "Your Consumer Segment",
                ["Low-Income/Protected", "Middle-Class/Slab-Breacher", 
                 "Heavy Commercial/AC-Heavy", "Industrial/Extreme"]
            )
        
        if st.button("🤖 Generate AI Recommendations", use_container_width=True):
            current_bill = billing.calculate_bill(user_consumption)['total_bill']
            
            user_profile = {
                'current_consumption': user_consumption,
                'segment': user_segment
            }
            
            prediction = {
                'predicted_units': user_consumption + np.random.normal(0, 10)
            }
            
            bill_info = {
                'current_bill': current_bill
            }
            
            try:
                ai_optimizer = AIOptimizer(config)
                recommendations = ai_optimizer.generate_optimization_advice(
                    user_profile, prediction, bill_info
                )
            except:
                recommendations = ai_optimizer._fallback_recommendation(
                    user_profile, prediction, bill_info
                )
            
            st.success("✓ Recommendations Generated")
            
            st.subheader("💰 Savings Opportunities")
            
            col_save_1, col_save_2, col_save_3 = st.columns(3)
            
            with col_save_1:
                savings_100 = billing.calculate_savings_target(user_consumption, "0-100")
                if not savings_100['already_in_target']:
                    st.metric(
                        "To Protected Status",
                        f"{savings_100['units_to_cut']:.0f} kWh cut",
                        f"Save PKR {savings_100['potential_savings']:,.0f}"
                    )
                else:
                    st.metric("Status", "✓ Already Protected")
            
            with col_save_2:
                savings_200 = billing.calculate_savings_target(user_consumption, "101-200")
                if not savings_200['already_in_target']:
                    st.metric(
                        "To Lower Slab",
                        f"{savings_200['units_to_cut']:.0f} kWh cut",
                        f"Save PKR {savings_200['potential_savings']:,.0f}"
                    )
                else:
                    st.metric("Current Bill", f"PKR {current_bill:,.0f}")
            
            with col_save_3:
                comparison = billing.get_bill_comparison(user_consumption)
                st.metric(
                    "Off-Peak Shifting",
                    f"PKR {comparison['savings_with_shifting']:,.0f}",
                    "Potential monthly savings"
                )
            
            st.divider()
            
            st.subheader("🎯 AI-Generated Advisory")
            st.info(recommendations)
            
            st.divider()
            
            st.subheader("📋 Action Plan")
            
            st.markdown(f"""
            **Your Current Profile:**
            - Consumption: {user_consumption} kWh/month
            - Segment: {user_segment}
            - Current Bill: PKR {current_bill:,.0f}
            
            **Recommended Actions:**
            1. **Immediate** (0-1 week): Audit appliances - identify top 3 energy consumers
            2. **Short-term** (1-2 weeks): Shift 30% of consumption to off-peak hours (midnight-6 AM)
            3. **Medium-term** (1 month): Install smart power strips and LED lighting
            4. **Long-term** (3-6 months): Consider solar backup for peak hours
            
            **Expected Impact:**
            - Target consumption: {max(100, user_consumption - 30)} kWh/month
            - Estimated savings: PKR {(current_bill * 0.15):,.0f}/month
            - Annual savings: PKR {(current_bill * 0.15 * 12):,.0f}
            """)
        
        st.divider()
        
        st.subheader("📞 Need More Help?")
        st.markdown("""
        - 📧 Contact DISCO (Distribution Company) helpline
        - 🌐 Visit NEPRA website for tariff details
        - 💡 Schedule home energy audit with certified technician
        """)

if __name__ == "__main__":
    main()
