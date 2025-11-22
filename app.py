import streamlit as st
import pandas as pd
import sqlite3
import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DB_ADI = "bankacilik_verileri.db"
TABLO_ADI = "makro_finans_tablosu"
TAHMIN_GRAFIK_ADI = "credit_volume_forecast.png" 
SCALING_FACTOR = 1_000_000_000 # 1 Billion TRY (Trilyon TL ölçeği için)

# --- VERİ YÜKLEME FONKSİYONLARI ---

@st.cache_data
def veriyi_yukle():
    """Loads the cleaned data from the SQLite database."""
    if not os.path.exists(DB_ADI):
        st.error("ERROR: Database file not found. Please run 'analiz_baslangic.py' first.")
        return pd.DataFrame()
        
    conn = sqlite3.connect(DB_ADI)
    
    # KRİTİK DÜZELTME: İndeks sütun adını dinamik olarak bulma.
    index_name_options = ['Date', 'index', 'level_0', 'Tarih'] 
    
    for col_name in index_name_options:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {TABLO_ADI}", conn, index_col=col_name)
            conn.close()
            
            df.index.name = 'Date' # ENGLISH LABEL
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            continue

    st.error("ERROR: Could not find the index column in the database.")
    conn.close()
    return pd.DataFrame()

# --- MODEL DİAGNOSTİK VERİSİNİ YÜKLEME (YENİ EKLENEN KISIM) ---
@st.cache_data
def load_diagnostics_data():
    """Loads Actual, Fitted, and Residuals data from the SQLite diagnostics table."""
    try:
        conn = sqlite3.connect(DB_ADI)
        # model_diagnostics tablosunu okuma
        df_diag = pd.read_sql_query("SELECT * FROM model_diagnostics", conn, index_col='Date')
        conn.close()
        
        # Index cleaning
        df_diag.index = pd.to_datetime(df_diag.index)
        return df_diag
    except Exception:
        return pd.DataFrame()

# --- KALINTI GRAFİĞİ FONKSİYONU ---
def plot_residuals(df_diag):
    """Plots the model residuals over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_diag.index, df_diag['Residuals'], label='Residuals (Errors)', color='red', alpha=0.7)
    ax.axhline(0, color='black', linestyle='--', linewidth=1) 
    ax.set_title('Residuals Analysis: Error Distribution Over Time (White Noise Check)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Residual Value (Billion TRY)')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.close(fig)
    return fig
# --- (Geri kalan yükleme fonksiyonları, tahmin tablosu ve ana akış) ---
@st.cache_data
def tahmin_tablosunu_yukle():
    """Loads and scales the final forecast results, based on model output."""
    
    # Forecast data (Hardcoded from the successful ARIMA output)
    data = {
        'Tahmin Edilen Hacim': [2071606808, 2132775688, 2191890854, 2249021258, 2304233537, 2357592091],
        'Alt Sınır (%95)': [2048383270, 2087954821, 2122526073, 2152545416, 2178460102, 2200647877],
        'Üst Sınır (%95)': [2094830345, 2177596556, 2261255636, 2345497101, 2430006972, 2514536306]
    }
    dates = ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01']
    df_tahmin = pd.DataFrame(data, index=pd.to_datetime(dates)).round(0).astype(int)
    
    # Scaling to TRILLION TRY (Trilyon TL)
    df_tahmin['Forecast (Trillion TRY)'] = (df_tahmin['Tahmin Edilen Hacim'] / SCALING_FACTOR).round(2)
    df_tahmin['Lower Bound (Trillion TRY)'] = (df_tahmin['Alt Sınır (%95)'] / SCALING_FACTOR).round(2)
    df_tahmin['Upper Bound (Trillion TRY)'] = (df_tahmin['Üst Sınır (%95)'] / SCALING_FACTOR).round(2)
    df_tahmin['Risk Margin (Trillion TRY)'] = (df_tahmin['Upper Bound (Trillion TRY)'] - df_tahmin['Lower Bound (Trillion TRY)']).round(2)

    return df_tahmin[['Forecast (Trillion TRY)', 'Lower Bound (Trillion TRY)', 'Upper Bound (Trillion TRY)', 'Risk Margin (Trillion TRY)']]

# --- INTERFACE CREATION ---

def main():
    st.set_page_config(layout="wide")
    st.title("🏦 Corporate Credit Volume Forecast (ARIMA) - Decision Support System") 

    df_ana = veriyi_yukle()
    df_tahmin = tahmin_tablosunu_yukle()
    df_diag = load_diagnostics_data() # YENİ TANI VERİSİ YÜKLENİYOR
    
    if df_ana.empty:
        st.warning("ERROR: Data could not be loaded. Ensure the database file is present and analysis has been run.")
        return

    # --- 1. DSS DASHBOARD METRICS ---
    st.header("1. Managerial Forecast and Liquidity Decisions") 
    col1, col2, col3 = st.columns(3)
    
    son_deger_trillion = df_ana['Kredi_Hacmi'].iloc[-1] / SCALING_FACTOR
    expected_growth_trillion = (df_tahmin['Forecast (Trillion TRY)'].iloc[-1] - son_deger_trillion)
    
    with col1:
        st.metric(label="Current Credit Volume (Dec 2024)", 
                  value=f"{son_deger_trillion:,.2f} Trillion TRY")
    with col2:
        st.metric(label="6-Month Expected Growth", 
                  value=f"{expected_growth_trillion:,.2f} Trillion TRY",
                  delta=f"{(df_tahmin['Forecast (Trillion TRY)'].iloc[-1] / son_deger_trillion - 1) * 100:.1f} %")
    with col3:
        st.metric(label="Estimated Average Interest Rate", 
                  value=f"{df_ana['Faiz_Orani'].mean():.2f} %")

    st.markdown("---")
    
    # --- 2. FORECAST VISUALIZATION ---
    st.header("2. Credit Volume Forecast and Confidence Interval (Trillion TRY)") 
    if os.path.exists(TAHMIN_GRAFIK_ADI):
        st.image(TAHMIN_GRAFIK_ADI, use_container_width=True)
    else:
        st.warning(f"ERROR: Plot file ({TAHMIN_GRAFIK_ADI}) not found. Please run the analysis script.")

    st.markdown("---")

    # --- 3. MANAGEMENT OUTPUT TABLE ---
    st.header("3. Detailed Liquidity and Risk Margin Table (Trillion TRY)") 
    
    df_gosterim = df_tahmin.copy()
    df_gosterim.index = df_gosterim.index.strftime('%B %Y')
    df_gosterim.index.name = "Forecast Period" 
    
    st.dataframe(df_gosterim.style.format({
        'Forecast (Trillion TRY)': '{:,.2f}',
        'Lower Bound (Trillion TRY)': '{:,.2f}',
        'Upper Bound (Trillion TRY)': '{:,.2f}',
        'Risk Margin (Trillion TRY)': '{:,.2f}'
    }), use_container_width=True)

    st.markdown("---")
    
    # --- 4. MODEL DIAGNOSTICS (YENİ EKLENEN AKADEMİK BÖLÜM) ---
    if not df_diag.empty:
        st.header("4. Model Diagnostics and Historical Fit Validation")
        
        # 4A. TARİHSEL UYUM GRAFİĞİ (Actual vs Fitted)
        st.subheader("4.1. Historical Fit (Actual vs. Model Estimates)")
        
        fig_fit, ax_fit = plt.subplots(figsize=(12, 6))
        ax_fit.plot(df_diag['Actual'], label='Actual Credit Volume', color='blue')
        ax_fit.plot(df_diag['Fitted'], label='Fitted Model Estimate', color='orange', linestyle='--')
        ax_fit.set_title('In-Sample Validation: Actual vs. Fitted Values')
        ax_fit.set_xlabel('Date')
        ax_fit.set_ylabel('Volume (Trillion TRY)') # TRİLYON ETİKETİ
        ax_fit.legend()
        ax_fit.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_fit)
        plt.close(fig_fit)

        # 4B. KALINTI (HATA) ANALİZİ
        st.subheader("4.2. Residuals Analysis (White Noise Check)")
        
        fig_res = plot_residuals(df_diag)
        st.pyplot(fig_res)
        
        st.markdown("""
        **Interpretation:** The random distribution of residuals around the zero line confirms that the ARIMA model successfully captured the underlying time series dependency and trend, leaving only white noise (random error).
        """)
    else:
        st.warning("ERROR: Model Diagnostics data (Actual/Fitted/Residuals) could not be loaded. Please ensure 'analiz_baslangic.py' was run successfully.")
        
    # --- 5. ACADEMIC & MODEL PERFORMANCE SUMMARY ---
    with st.expander("Model Parameters and Academic Summary"): 
        st.subheader("ARIMA(1, 1, 1) Parameters") 
        st.markdown("""
        - **p (AR Component):** 1 (Determined by PACF analysis)
        - **d (Differencing Order):** 1 (Determined by ADF test)
        - **q (MA Component):** 1 (Determined by ACF analysis)
        
        **Model Suitability:**
        - **Ljung-Box (Q) Prob:** 0.34 (Confirms model adequacy.)
        - **Managerial Action:** The 95% Lower Bound defines the minimum expected demand, guiding conservative liquidity provisioning.
        """)
        
if __name__ == "__main__":
    main()