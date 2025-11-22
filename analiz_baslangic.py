import pandas as pd
import sqlite3
import numpy as np
from statsmodels.tsa.stattools import adfuller
import os 
from io import StringIO 
import re 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf 
from statsmodels.tsa.arima.model import ARIMA 

# --- CONFIGURATION ---
DOSYA_ADI = "manuel_veri.csv" 
DB_ADI = "bankacilik_verileri.db"
TABLO_ADI = "makro_finans_tablosu"
MANUEL_SUTUNLAR = ['Tarih', 'Kredi_Hacmi', 'Faiz_Orani', 'Tuketici_Guveni', 'USD_TRY', 'TUFE']
SCALING_FACTOR = 1_000_000_000 # 1 Billion TRY

# --- VERİ TEMİZLİK FONKSİYONLARI ---
# (Önceki başarılı kod blokları)

def on_isleme_yap(raw_data):
    clean_data = re.sub(r';\s*[\r\n]+', ';', raw_data)
    clean_data = re.sub(r'"(\d+);\s*(\d+\.?\d*)"', r'\1\2', clean_data)
    clean_data = re.sub(r'"(\d+),(\d+),(\d+\.?\d*)"', r'\1\2\3', clean_data) 
    clean_data = clean_data.replace('"', '')
    clean_data = '\n'.join([line.strip() for line in clean_data.split('\n')])
    return clean_data

def veriyi_yukle_manuel_garantili_cozum():
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
            
        veri_df = veri_df.dropna().astype(float)
        return veri_df
    
    except Exception: return pd.DataFrame()

def veriyi_kaydet(df):
    """Saves the DataFrame to the SQLite database."""
    conn = sqlite3.connect(DB_ADI)
    df.to_sql(TABLO_ADI, conn, if_exists='replace', index=True) 
    conn.close()
    print(f"Data successfully saved to {DB_ADI} database.")

# --- YENİ EKLENEN TANI VERİSİ KAYIT FONKSİYONU ---
def kayit_diagnostik_verisi(model_fit, veri_df):
    """Modelin geçmiş tahminlerini (Fitted) ve hatalarını (Residuals) hesaplar ve veritabanına kaydeder."""
    
    # Modelin eğitildiği serinin başlangıç ve bitiş tarihini kullan
    start_date = veri_df.index[0]
    end_date = veri_df.index[-1]
    
    # In-sample (tarihsel) tahminleri hesapla
    fitted_values = model_fit.predict(start=start_date, end=end_date, dynamic=False)
    residuals = model_fit.resid

    # Verileri birleştir ve Milyar TL'ye ölçekle
    diagnostic_df = pd.DataFrame({
        'Actual': veri_df['Kredi_Hacmi'] / SCALING_FACTOR,
        'Fitted': fitted_values / SCALING_FACTOR,
        'Residuals': residuals / SCALING_FACTOR # Hatalar da ölçeklenir
    }).dropna()
    
    # Yeni tabloya kaydet
    try:
        conn = sqlite3.connect(DB_ADI)
        diagnostic_df.to_sql('model_diagnostics', conn, if_exists='replace', index=True)
        conn.close()
        print("Model diagnostics (Actual, Fitted, Residuals) successfully saved to 'model_diagnostics' table.")
    except Exception as e:
        print(f"ERROR saving diagnostics: {e}")

# --- ANALİZ VE MODELLEME FONKSİYONLARI ---
# (ADF ve ACF/PACF fonksiyonları aynı kalır)

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
    """ARIMA(1, 1, 1) modelini eğitir ve özetini döndürür."""
    ts = veri_df['Kredi_Hacmi']
    ts = ts.sort_index()
    ts.index = pd.to_datetime(ts.index)
    ts = ts.asfreq('MS') 
    
    model = ARIMA(ts, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False)
    model_fit = model.fit()
    
    print("\n" + "="*50)
    print("ARIMA(1, 1, 1) MODEL SUMMARY")
    print("="*50)
    print(model_fit.summary())
    
    return model_fit

def tahmin_yap_ve_ciz(model_fit, veri_df, adim_sayisi=6):
    """Performs the 6-month forecast, generates the output table, and saves the forecast plot."""
    # Forecast ve plotting kodları (önceki başarılı adımlardan)
    forecast_results = model_fit.get_forecast(steps=adim_sayisi)
    forecast_mean = forecast_results.predicted_mean
    confidence_intervals = forecast_results.conf_int()
    
    tahmin_df = pd.DataFrame({
        'Forecasted Volume': forecast_mean, 
        'Lower Bound (95%)': confidence_intervals['lower Kredi_Hacmi'], 
        'Upper Bound (95%)': confidence_intervals['upper Kredi_Hacmi'] 
    })
    
    tahmin_df = tahmin_df.round(0).astype(int)
    tahmin_df_billion = (tahmin_df / SCALING_FACTOR).round(2)
    tahmin_df_billion.index.name = "Forecast Date"

    # Konsol Çıktısı
    print("\n" + "="*70)
    print(f"ARIMA(1, 1, 1) - {adim_sayisi}-MONTH CREDIT VOLUME FORECAST (Billion TRY)")
    print("="*70)
    print(tahmin_df_billion.to_string())

    # Grafik Çizimi (KDS Çıktı Bileşeni)
    plt.figure(figsize=(14, 7))
    historic_data_billion = veri_df['Kredi_Hacmi'] / SCALING_FACTOR
    
    plt.plot(historic_data_billion, label='Actual Credit Volume', color='blue')
    plt.plot(tahmin_df_billion['Forecasted Volume'], label='Forecasted Credit Volume', color='red', linestyle='--')
    
    plt.fill_between(confidence_intervals.index, 
                     tahmin_df_billion['Lower Bound (95%)'], 
                     tahmin_df_billion['Upper Bound (95%)'], 
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    
    plt.title(f'Credit Volume Forecast and {adim_sayisi}-Month Projection (Billion TRY)') 
    plt.xlabel('Date')
    plt.ylabel('Credit Volume (Billion TRY)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('credit_volume_forecast.png')
    plt.close()
    print("Forecast graph saved to 'credit_volume_forecast.png'.")
    
    return tahmin_df_billion

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
        
        # 4. YENİ: Model Tanı Verisi Kaydı
        kayit_diagnostik_verisi(arima_model, veri_df)
        
        # 5. Tahmin ve Grafikleme
        tahmin_df = tahmin_yap_ve_ciz(arima_model, veri_df, adim_sayisi=6)