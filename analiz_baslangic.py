import pandas as pd
import sqlite3
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os 
from io import StringIO 
import re 
import matplotlib
# CRITICAL FIX: Set backend to 'Agg' to prevent Tcl/Tk error during plotting
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima.model import ARIMA 

# --- CONFIGURATION ---
DOSYA_ADI = "manuel_veri.csv" 
DB_ADI = "bankacilik_verileri.db"
TABLO_ADI = "makro_finans_tablosu"

# KRİTİK DÜZELTME: GRAFİK ÇIKTISI TRİLYON TL ETİKETİYLE DEĞİŞTİRİLDİ
TAHMIN_GRAFIK_ADI = "credit_volume_forecast.png" 
SCALING_FACTOR = 1_000_000_000 # 1 Billion TRY (Trilyon TL ölçeği için)

# Sütun isimlerini manuel olarak belirle
MANUEL_SUTUNLAR = ['Tarih', 'Kredi_Hacmi', 'Faiz_Orani', 'Tuketici_Guveni', 'USD_TRY', 'TUFE']

# --- VERİ TEMİZLİK FONKSİYONLARI ---

def on_isleme_yap(raw_data):
    """Pre-processes raw data by fixing line breaks, quotes, and inconsistent separators."""
    clean_data = re.sub(r';\s*[\r\n]+', ';', raw_data)
    clean_data = re.sub(r'"(\d+);\s*(\d+\.?\d*)"', r'\1\2', clean_data)
    clean_data = re.sub(r'"(\d+),(\d+),(\d+\.?\d*)"', r'\1\2\3', clean_data) 
    clean_data = clean_data.replace('"', '')
    clean_data = '\n'.join([line.strip() for line in clean_data.split('\n')])
    return clean_data

def veriyi_yukle_manuel_garantili_cozum():
    """Tries to load CSV with multiple delimiters and applies final cleaning."""
    if not os.path.exists(DOSYA_ADI):
        print(f"ERROR: '{DOSYA_ADI}' file not found.")
        return pd.DataFrame()
    
    try:
        with open(DOSYA_ADI, 'r', encoding='latin-1') as f: raw_data = f.read()

        clean_data = on_isleme_yap(raw_data)
        veri_df = pd.DataFrame()
        
        def safe_read(sep_char):
            return pd.read_csv(StringIO(clean_data), sep=sep_char, engine='python', 
                                 skiprows=1, header=None, names=MANUEL_SUTUNLAR, skipinitialspace=True)

        # Delimiter attempts
        try:
            temp_df = safe_read(';')
            if 'Tarih' in temp_df.columns and not temp_df['Tarih'].isnull().all(): veri_df = temp_df
        except Exception: pass
            
        if veri_df.empty:
            try:
                temp_df = safe_read(',')
                if 'Tarih' in temp_df.columns and not temp_df['Tarih'].isnull().all(): veri_df = temp_df
            except Exception: pass
        
        if veri_df.empty:
             print("ERROR: Both delimiter attempts failed or 'Tarih' column is empty.")
             return pd.DataFrame()

        # Final Cleaning (Index, Date, Numeric Conversion)
        tarih_sutunu_adi = 'Tarih'
        veri_df.set_index(tarih_sutunu_adi, inplace=True)
        veri_df.index.name = 'Date' 
        
        veri_df = veri_df.sort_index()
        veri_df.index = pd.to_datetime(veri_df.index, format='%d.%m.%Y', errors='coerce') 
        veri_df = veri_df[veri_df.index.notna()] 

        for col in veri_df.columns:
            temiz_seri = veri_df[col].astype(str)
            if col == 'Kredi_Hacmi': 
                temiz_seri = temiz_seri.str.replace('.', '', regex=False)
            temiz_seri = temiz_seri.str.replace(',', '.', regex=False)
            veri_df[col] = pd.to_numeric(temiz_seri, errors='coerce')
            
        veri_df = veri_df.dropna()
        veri_df = veri_df.astype(float)
        
        print("Data cleaning and preparation complete.")
        return veri_df
    
    except Exception: return pd.DataFrame()

def veriyi_kaydet(df):
    """Saves the DataFrame to the SQLite database."""
    conn = sqlite3.connect(DB_ADI)
    df.to_sql(TABLO_ADI, conn, if_exists='replace', index=True) 
    conn.close()
    print(f"Data successfully saved to {DB_ADI} database.")

# --- ANALİZ VE MODELLEME FONKSİYONLARI ---

def adf_testi_yap(seri, isim):
    """Performs the Augmented Dickey-Fuller (ADF) test."""
    seri = seri.dropna()
    if len(seri) < 15:
        print(f"WARNING: {isim} series is too short for ADF test ({len(seri)} observations).")
        return
        
    print(f"\n--- ADF Test Result for {isim} Series ---")
    sonuc = adfuller(seri)
    p_degeri = sonuc[1]
    
    print(f"P-value: {p_degeri:.4f}")
    
    if p_degeri <= 0.05:
        print("Conclusion: Series is STATIONARY. 🎉")
    else:
        print("Conclusion: Series is NON-STATIONARY (Differencing is required). ⚠️")

def acf_pacf_ciz(seri, baslik):
    """Plots ACF and PACF graphs and saves them to a file."""
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle(baslik, fontsize=16)

    plot_acf(seri, lags=20, ax=axes[0], title='Autocorrelation Function (ACF)') 
    axes[0].grid(True, linestyle='--', alpha=0.7)

    plot_pacf(seri, lags=20, ax=axes[1], title='Partial Autocorrelation Function (PACF)', method='ywm') 
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig('acf_pacf_analysis.png')
    plt.close(fig)
    print("ACF/PACF graph saved to 'acf_pacf_analysis.png'.")

def model_egit_ve_degerlendir(veri_df):
    """Trains the ARIMA(1, 1, 1) model and returns its fit object."""
    
    ts = veri_df['Kredi_Hacmi']
    ts = ts.sort_index()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.asfreq('MS') 
    
    model = ARIMA(ts, order=(1, 1, 1), 
                  enforce_stationarity=False, 
                  enforce_invertibility=False)
                  
    model_fit = model.fit()
    return model_fit

def kayit_diagnostik_verisi(model_fit, veri_df):
    """Modelin geçmiş tahminlerini (Fitted) ve hatalarını (Residuals) hesaplar ve veritabanına kaydeder."""
    start_date = veri_df.index[0]
    end_date = veri_df.index[-1]
    
    fitted_values = model_fit.predict(start=start_date, end=end_date, dynamic=False)
    residuals = model_fit.resid

    diagnostic_df = pd.DataFrame({
        'Actual': veri_df['Kredi_Hacmi'] / SCALING_FACTOR,
        'Fitted': fitted_values / SCALING_FACTOR,
        'Residuals': residuals / SCALING_FACTOR
    }).dropna()
    
    try:
        conn = sqlite3.connect(DB_ADI)
        diagnostic_df.to_sql('model_diagnostics', conn, if_exists='replace', index=True)
        conn.close()
        print("Model diagnostics (Actual, Fitted, Residuals) successfully saved to 'model_diagnostics' table.")
    except Exception as e:
        print(f"ERROR saving diagnostics: {e}")

def tahmin_yap_ve_ciz(model_fit, veri_df, adim_sayisi=6):
    """Performs the 6-month forecast, generates the output table, and saves the forecast plot."""
    
    forecast_results = model_fit.get_forecast(steps=adim_sayisi)
    
    forecast_mean = forecast_results.predicted_mean
    confidence_intervals = forecast_results.conf_int()
    
    # Tahmin tablosunu oluşturma (Konsol için)
    tahmin_df = pd.DataFrame({
        'Forecasted Volume': forecast_mean,
        'Lower Bound (95%)': confidence_intervals['lower Kredi_Hacmi'],
        'Upper Bound (95%)': confidence_intervals['upper Kredi_Hacmi']
    })
    
    # Grafik Çizimi (Trilyon TL ölçeğinde)
    plt.figure(figsize=(14, 7))
    historic_data_trillion = veri_df['Kredi_Hacmi'] / SCALING_FACTOR
    
    plt.plot(historic_data_trillion, label='Actual Credit Volume', color='blue')
    plt.plot(forecast_mean / SCALING_FACTOR, label='Forecasted Credit Volume', color='red', linestyle='--')
    
    plt.fill_between(confidence_intervals.index, 
                     confidence_intervals['lower Kredi_Hacmi'] / SCALING_FACTOR, 
                     confidence_intervals['upper Kredi_Hacmi'] / SCALING_FACTOR, 
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    
    # KRİTİK DÜZELTME: Grafik etiketleri Trilyon TL'ye ayarlandı.
    plt.title(f'Credit Volume Forecast and {adim_sayisi}-Month Projection (Trillion TRY)') 
    plt.xlabel('Date')
    plt.ylabel('Credit Volume (Trillion TRY)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('credit_volume_forecast.png') # ENGLISH FILENAME
    plt.close()
    print("Forecast graph saved to 'credit_volume_forecast.png'.")
    
    return tahmin_df

# --- ANA AKIŞ ---
print("--- Data Loading and Preprocessing Started ---")
veri_df = veriyi_yukle_manuel_garantili_cozum() 

if not veri_df.empty:
    
    veriyi_kaydet(veri_df)
    
    print("\n--- FIRST 5 ROWS (Successfully Loaded) ---")
    print(veri_df.head())
    
    if 'Kredi_Hacmi' in veri_df.columns:
        print("\n--- Analysis Commencing ---")
        
        # 1. ADF Testi
        adf_testi_yap(veri_df['Kredi_Hacmi'], 'Credit Volume (Raw)')
        fark_seri = veri_df['Kredi_Hacmi'].diff().dropna()
        adf_testi_yap(fark_seri, 'Credit Volume (Differenced)')

        # 2. ACF/PACF Çizim
        acf_pacf_ciz(fark_seri, "ACF and PACF Analysis for Differenced Credit Volume Series")
        
        # 3. Model Eğitimi
        arima_model = model_egit_ve_degerlendir(veri_df)
        
        # 4. Model Tanı Verisi Kaydı
        kayit_diagnostik_verisi(arima_model, veri_df)
        
        # 5. Tahmin ve Grafikleme
        tahmin_df = tahmin_yap_ve_ciz(arima_model, veri_df, adim_sayisi=6)
        
    else:
         print("ERROR: 'Kredi_Hacmi' column not found.")