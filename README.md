# 🏦 Corporate Credit Volume Forecast - ARIMA Decision Support System

A comprehensive financial forecasting application that uses ARIMA time series modeling to predict corporate credit volumes. This Decision Support System (DSS) provides managers with actionable insights for liquidity planning and risk management.

## 📋 Features

- **Time Series Forecasting**: ARIMA(1,1,1) model for 6-month credit volume predictions
- **Interactive Dashboard**: Streamlit-based web interface with real-time visualizations
- **Model Diagnostics**: Comprehensive validation including ADF tests, ACF/PACF analysis, and residual diagnostics
- **Confidence Intervals**: 95% prediction intervals for risk assessment
- **Historical Validation**: In-sample fit analysis comparing actual vs. fitted values
- **Database Integration**: SQLite database for efficient data storage and retrieval

## 🏗️ Architecture

```
├── app.py                    # Streamlit web application
├── analiz_baslangic.py       # Data processing and ARIMA model training
├── bankacilik_verileri.db    # SQLite database with financial data
├── requirements.txt          # Python dependencies
└── manuel_veri.csv          # Input data file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip package manager

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Place your CSV file as `manuel_veri.csv` in the project root
   - Expected columns: `Tarih`, `Kredi_Hacmi`, `Faiz_Orani`, `Tuketici_Guveni`, `USD_TRY`, `TUFE`
   - Date format: `DD.MM.YYYY`

4. **Run the analysis**
   ```bash
   python analiz_baslangic.py
   ```
   This will:
   - Load and clean the data
   - Perform stationarity tests (ADF)
   - Generate ACF/PACF plots
   - Train the ARIMA model
   - Save diagnostics to the database
   - Generate forecast visualizations

5. **Launch the web application**
   ```bash
   streamlit run app.py
   ```
   The dashboard will open at `http://localhost:8501`

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker-compose up -d
```

The application will be available at `http://localhost:8501`

### Using Docker directly

```bash
# Build the image
docker build -t credit-forecast-app .

# Run the container
docker run -p 8501:8501 credit-forecast-app
```

## 📊 Data Format

Your CSV file should have the following structure:

| Tarih | Kredi_Hacmi | Faiz_Orani | Tuketici_Guveni | USD_TRY | TUFE |
|-------|-------------|------------|-----------------|---------|------|
| 01.01.2020 | 1234567890 | 12.5 | 95.2 | 6.85 | 125.3 |
| 01.02.2020 | 1245678901 | 12.3 | 96.1 | 6.92 | 126.1 |

**Important Notes:**
- Dates must be in `DD.MM.YYYY` format
- Credit volume should be in Turkish Lira (without decimal separators)
- The script automatically handles various CSV delimiters (`,` or `;`)

## 🔬 Model Details

### ARIMA(1, 1, 1) Parameters

- **p (AR Component)**: 1 - Determined by PACF analysis
- **d (Differencing Order)**: 1 - Determined by Augmented Dickey-Fuller (ADF) test
- **q (MA Component)**: 1 - Determined by ACF analysis

### Model Validation

- **Ljung-Box Q Test**: p-value = 0.34 (confirms model adequacy)
- **Residual Analysis**: White noise confirmation
- **In-Sample Fit**: Historical validation against actual data

## 📈 Dashboard Sections

1. **Managerial Forecast and Liquidity Decisions**
   - Current credit volume metrics
   - Expected growth projections
   - Average interest rate indicators

2. **Credit Volume Forecast Visualization**
   - Historical data with 6-month forecast
   - 95% confidence intervals
   - Risk margin calculations

3. **Detailed Liquidity and Risk Margin Table**
   - Month-by-month forecasts
   - Upper and lower bounds
   - Risk margin analysis

4. **Model Diagnostics**
   - Actual vs. Fitted values comparison
   - Residuals analysis (white noise check)
   - Model performance metrics

## 🛠️ Development

### Project Structure

```
.
├── app.py                    # Main Streamlit application
├── analiz_baslangic.py       # Data processing and modeling
├── requirements.txt          # Python dependencies
├── Dockerfile               # Docker container configuration
├── docker-compose.yml       # Docker Compose configuration
├── .dockerignore            # Docker ignore patterns
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

### Key Functions

- `veriyi_yukle()`: Loads cleaned data from SQLite database
- `tahmin_tablosunu_yukle()`: Retrieves forecast results
- `load_diagnostics_data()`: Loads model diagnostics
- `plot_residuals()`: Generates residual analysis plots

## 🔧 Configuration

Key configuration variables in `app.py`:

```python
DB_ADI = "bankacilik_verileri.db"
TABLO_ADI = "makro_finans_tablosu"
TAHMIN_GRAFIK_ADI = "credit_volume_forecast.png"
SCALING_FACTOR = 1_000_000_000  # Converts to Trillion TRY
```

## 📝 Output Files

- `bankacilik_verileri.db`: SQLite database with processed data
- `credit_volume_forecast.png`: Forecast visualization
- `acf_pacf_analysis.png`: ACF/PACF diagnostic plots

## 🐛 Troubleshooting

### Database not found
- Ensure `analiz_baslangic.py` has been run successfully
- Check that `bankacilik_verileri.db` exists in the project root

### CSV loading errors
- Verify CSV file format matches expected structure
- Check date format is `DD.MM.YYYY`
- Ensure numeric columns don't have text values

### Model training issues
- Ensure sufficient data points (minimum 15 observations)
- Check for missing values in the dataset
- Verify time series is properly sorted by date

## 📄 License

This project is provided as-is for educational and business purposes.

## 👥 Contributing

Contributions are welcome! Please ensure:
- Code follows Python PEP 8 style guidelines
- All tests pass before submitting
- Documentation is updated for new features

## 📧 Support

For issues or questions, please open an issue in the repository.

---

**Built with**: Python, Streamlit, Statsmodels, Pandas, Matplotlib, SQLite
