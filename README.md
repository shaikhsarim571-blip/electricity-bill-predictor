# Pakistan Electricity Smart Bill Predictor & Optimization Dashboard

A comprehensive AI/ML-powered dashboard for predicting electricity consumption, optimizing bills, and providing personalized energy-saving recommendations.

## 📋 Project Structure

```
ai project 2/
├── config.py                 # Configuration, constants, and business rules
├── data_engineering.py       # Data preprocessing, EDA, feature engineering
├── models.py                 # Random Forest, XGBoost, and Prophet models
├── clustering.py             # K-Means segmentation and PCA visualization
├── billing_engine.py         # NEPRA tariff calculation and bill generation
├── generative_ai.py          # LLM integration (Groq/OpenAI) for recommendations
├── visualizations.py         # Plotly/Seaborn charts and interactive plots
├── app.py                    # Main Streamlit dashboard application
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🎯 Core AI/ML Modules

### Module A: Data Engineering & Feature Exploration
**File**: `data_engineering.py`

**Responsibilities**:
- Missing data imputation using seasonal median (instead of dropping rows)
- IQR-based outlier detection and removal
- Temporal feature extraction (month, quarter, season, peak/off-peak indicators)
- Feature normalization (MinMax or Z-score scaling)
- Seasonal decomposition (trend, seasonality, residuals)

**Key Functions**:
- `DataPreprocessor.handle_missing_data()` - Seasonal median filling
- `DataPreprocessor.detect_outliers()` - IQR-based detection
- `DataPreprocessor.decompose_timeseries()` - Seasonal decomposition
- `create_sample_dataset()` - Generate synthetic Pakistan electricity data

### Module B: Predictive Modeling
**File**: `models.py`

**Concept**: Supervised learning regression + Time-series forecasting

**Models Implemented**:
1. **Random Forest Regressor** - Ensemble method for robust predictions
2. **XGBoost** - Gradient boosting for improved accuracy
3. **Facebook Prophet** - Time-series forecasting with seasonality

**Evaluation Metrics**:
- **R² Score**: Proportion of variance explained (range: 0-1, higher is better)
- **RMSE**: Root Mean Squared Error (units of kWh)
- **MAE**: Mean Absolute Error (units of kWh)
- **MAPE**: Mean Absolute Percentage Error (%)

**Formula Reference**:
```
R² = 1 - (SS_res / SS_tot)
RMSE = √(Σ(y_actual - y_pred)² / n)
MAE = Σ|y_actual - y_pred| / n
MAPE = (100/n) * Σ|y_actual - y_pred| / y_actual
```

### Module C: Consumer Segmentation
**File**: `clustering.py`

**Concept**: Unsupervised K-Means clustering with PCA visualization

**Process**:
1. Standardize features (temperature, consumption, humidity)
2. Apply K-Means clustering (k=4) on standardized data
3. Reduce dimensionality using PCA for 2D/3D visualization
4. Assign segment names based on consumption levels

**Segments**:
- Cluster 0: Low-Income/Protected (0-100 kWh)
- Cluster 1: Middle-Class/Slab-Breacher (101-300 kWh)
- Cluster 2: Heavy Commercial/AC-Heavy (301-500 kWh)
- Cluster 3: Industrial/Extreme (500+ kWh)

**Quality Metrics**:
- **Silhouette Score**: Measures cluster cohesion (-1 to 1, higher is better)
- **Davies-Bouldin Index**: Average cluster similarity (lower is better)

### Module D: Agentic Prescriptive Insights
**File**: `generative_ai.py`

**Concept**: RAG-based LLM optimization with NEPRA tariff rules

**Features**:
- Integration with Groq (free), OpenAI, or HuggingFace
- Rule-guided prompt engineering with exact tariff slabs
- Fallback recommendations if LLM unavailable
- JSON extraction from model responses

**Output**: Personalized energy-saving strategies with expected PKR savings

## 🎨 UI/UX Dashboard (Streamlit)

**Theme**: Dark mode with electric blue (#00D9FF) and mint green (#39FF14) accents

**Three Main Tabs**:

1. **Dashboard & Live Predictor**
   - KPI cards (avg consumption, protected status %, dataset size)
   - Interactive sliders for what-if simulation
   - Real-time bill calculation
   - Pie chart bill breakdown

2. **Core AI Engine Analytics**
   - EDA Analytics Hub: Correlation heatmap, seasonal decomposition
   - Predictive Models: Model comparison, actual vs predicted with forecast window
   - Consumer Segmentation: 2D/3D scatter plots with cluster profiles
   - Feature Analysis: Statistical distributions by season

3. **AI Bill Optimizer**
   - Personalized energy-saving recommendations
   - Savings opportunities by tariff slab
   - Off-peak shifting benefits
   - AI-generated action plan

## 🚀 Quick Start

### Installation

```bash
cd "ai project 2"
pip install -r requirements.txt
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

### Using Your Own Dataset

Replace in `app.py`:
```python
@st.cache_resource
def load_data():
    df = pd.read_csv('your_dataset.csv')
    return df
```

**Required columns**:
- `date` (datetime)
- `unit_consumption_kwh` (numeric)
- `temperature` (numeric, optional)
- `humidity` (numeric, optional)
- `bill_amount` (numeric, optional)

## 📊 Key Formulas & Metrics

### Tariff Calculation (NEPRA)
```
Total Bill = Base Charge + (Units × Rate/Unit) + Taxes
Base Charge = PKR 300
Tax Rate = 17%

Slab Rates:
- 0-100 kWh: PKR 18.50/unit (Protected)
- 101-200 kWh: PKR 20.50/unit
- 201-300 kWh: PKR 24.75/unit
- 301-500 kWh: PKR 28.90/unit
- 500+ kWh: PKR 32.50/unit
```

### Feature Importance (Random Forest/XGBoost)
```
Importance = Σ(information_gain) across all trees
Normalized to 0-1 range
Shows which features drive predictions most
```

### Seasonal Decomposition (Additive Model)
```
Y(t) = Trend(t) + Seasonality(t) + Residual(t)
- Trend: Long-term component
- Seasonality: Repeating pattern (12-month cycle)
- Residual: Random noise
```

### PCA Variance Explained
```
Explained Variance = Σ(λᵢ) / Σ(all eigenvalues)
Where λ = eigenvalue
Shows how much information each component captures
```

## 🔧 Configuration

Edit `config.py` to customize:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `OPTIMAL_CLUSTERS` | 4 | Number of consumer segments |
| `TEST_SIZE` | 0.2 | Train-test split ratio |
| `FORECAST_HORIZON` | 3 | Months to forecast ahead |
| `LLM_PROVIDER` | "groq" | "groq", "openai", or "huggingface" |
| `TEMPERATURE` | 0.7 | LLM creativity (0-1) |
| `MAX_TOKENS` | 500 | LLM response length |

## 🌐 API Keys (Optional)

For LLM recommendations:

```bash
# For Groq (free, recommended)
export GROQ_API_KEY="your_groq_api_key"

# For OpenAI
export OPENAI_API_KEY="your_openai_api_key"
```

Get free Groq API: https://console.groq.com

## 📈 Sample Output

The dashboard generates:
- ✓ Interactive correlation heatmap
- ✓ Seasonal decomposition plots
- ✓ Model performance comparison charts
- ✓ Actual vs predicted consumption curves
- ✓ Consumer segmentation scatter plots
- ✓ Bill breakdown pie charts
- ✓ AI-powered recommendations

## 🛠️ Technologies Used

- **Data Processing**: Pandas, NumPy, SciPy
- **ML Models**: Scikit-learn, XGBoost, Facebook Prophet
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Dashboard**: Streamlit
- **LLM**: Groq API (Free tier)
- **Clustering**: K-Means, PCA

## 📝 Notes

1. **Data Quality**: IQR-based outlier removal preserves seasonal patterns
2. **Imputation Strategy**: Uses seasonal median instead of global mean
3. **Model Selection**: Automatic best model selection based on R²
4. **Forecasting**: Prophet handles seasonality better for electricity data
5. **Privacy**: All recommendations based on consumption profiles, no personal data

## 🎓 Educational Value

This project demonstrates:
- End-to-end ML pipeline
- Time-series forecasting
- Unsupervised clustering
- LLM integration
- Interactive dashboard design
- Real-world business problem solving

## 📄 License

Open source - feel free to modify and extend

---

**Built with ❤️ for Pakistan's energy sustainability**
