import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def calculate_rsi(data, period=14):
    """
    RSI hesaplama fonksiyonu
    """
    # Kapanış fiyatı değişimlerini hesaplama
    delta = data.diff()
    
    # Pozitif ve negatif değişimleri ayırma
    gain = delta.copy()
    loss = delta.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    loss = abs(loss)
    
    # İlk ortalama değerleri hesaplama
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Aşağıdaki hesaplamalar için NaN değerleri temizleme
    avg_gain = avg_gain.fillna(0)
    avg_loss = avg_loss.fillna(0)
    
    # RS hesaplama (Relative Strength)
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Sıfıra bölmeyi önleme
    
    # RSI hesaplama
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_bist30_tickers():
    """
    BIST 30 hisselerinin sembollerini döndürür.
    Not: BIST endeksi değişebildiğinden, bu liste güncel olmayabilir.
    """
    # BIST 30 hisseleri (Türkiye'de .IS uzantısı ile kullanılır)
    bist30_tickers = [
        'AKBNK.IS', 'ARCLK.IS', 'ASELS.IS', 'BIMAS.IS', 'EKGYO.IS',
        'EREGL.IS', 'FROTO.IS', 'GARAN.IS', 'HALKB.IS', 'ISCTR.IS',
        'KCHOL.IS', 'KOZAA.IS', 'KOZAL.IS', 'KRDMD.IS', 'PETKM.IS',
        'PGSUS.IS', 'SAHOL.IS', 'SISE.IS', 'TCELL.IS', 'THYAO.IS',
        'TOASO.IS', 'TUPRS.IS', 'VAKBN.IS', 'YKBNK.IS', 'TAVHL.IS',
        'TKFEN.IS', 'TTKOM.IS', 'ULKER.IS', 'SODA.IS', 'VESTL.IS'
    ]
    return bist30_tickers

def analyze_bist30_rsi(period=14, days=180):
    """
    BIST 30 hisselerinin RSI değerlerini hesaplar ve analiz eder
    """
    # Tarih aralığını belirle
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # BIST 30 hisselerinin sembollerini al
    bist30_tickers = get_bist30_tickers()
    
    # Sonuçları saklamak için DataFrame oluştur
    results = pd.DataFrame(columns=['Ticker', 'Current RSI', 'Last Price', '5-day RSI Trend'])
    
    for ticker in bist30_tickers:
        try:
            # Yahoo Finance'dan verileri al
            stock_data = yf.download(ticker, start=start_date, end=end_date)
            
            if len(stock_data) == 0:
                print(f"{ticker} için veri bulunamadı.")
                continue
                
            # RSI hesapla
            stock_data['RSI'] = calculate_rsi(stock_data['Close'], period=period)
            
            # Son RSI değeri
            current_rsi = stock_data['RSI'].iloc[-1]
            
            # Son 5 günlük RSI trendi (pozitif, negatif veya yatay)
            rsi_5days = stock_data['RSI'].iloc[-5:].values
            if rsi_5days[-1] > rsi_5days[0] * 1.05:  # %5'ten fazla artış
                trend = "Yükseliş"
            elif rsi_5days[-1] < rsi_5days[0] * 0.95:  # %5'ten fazla düşüş
                trend = "Düşüş"
            else:
                trend = "Yatay"
            
            # Sonuçları ekle
            results = pd.concat([results, pd.DataFrame({
                'Ticker': [ticker],
                'Current RSI': [round(current_rsi, 2)],
                'Last Price': [round(stock_data['Close'].iloc[-1], 2)],
                '5-day RSI Trend': [trend]
            })], ignore_index=True)
            
            print(f"{ticker} - RSI: {round(current_rsi, 2)}")
            
        except Exception as e:
            print(f"{ticker} için hata oluştu: {e}")
    
    return results, stock_data

def plot_rsi_signals(ticker, days=90):
    """
    Belirli bir hisse için RSI sinyallerini gösteren grafik çizer
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Verileri al
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    
    # RSI hesapla
    stock_data['RSI'] = calculate_rsi(stock_data['Close'])
    
    # Aşırı alım ve satım sinyallerini belirle
    stock_data['Overbought'] = stock_data['RSI'] > 70
    stock_data['Oversold'] = stock_data['RSI'] < 30
    
    # Grafikleri çiz
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Fiyat grafiği
    ax1.plot(stock_data.index, stock_data['Close'], label='Kapanış Fiyatı')
    
    # Aşırı alım ve satım noktalarını işaretle
    overbought_points = stock_data[stock_data['Overbought']]
    oversold_points = stock_data[stock_data['Oversold']]
    
    ax1.scatter(overbought_points.index, overbought_points['Close'], color='red', marker='^', 
                label='Aşırı Alım (RSI > 70)')
    ax1.scatter(oversold_points.index, oversold_points['Close'], color='green', marker='v', 
                label='Aşırı Satım (RSI < 30)')
    
    ax1.set_title(f'{ticker} Hisse Senedi Fiyatı ve RSI Sinyalleri')
    ax1.set_ylabel('Fiyat (TL)')
    ax1.legend()
    ax1.grid(True)
    
    # RSI grafiği
    ax2.plot(stock_data.index, stock_data['RSI'], color='purple', label='RSI')
    ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
    ax2.fill_between(stock_data.index, 70, 30, color='gray', alpha=0.2)
    
    ax2.set_ylabel('RSI')
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('Tarih')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# Ana çalıştırma kodu
if __name__ == "__main__":
    # Tüm BIST 30 hisseleri için RSI analizi yap
    results, _ = analyze_bist30_rsi(period=14, days=180)
    
    # Sonuçları göster
    print("\nBIST 30 RSI Analizi Sonuçları:")
    print("="*60)
    
    # RSI değerine göre sırala (düşükten yükseğe)
    sorted_results = results.sort_values(by='Current RSI')
    print(sorted_results.to_string(index=False))
    
    # Aşırı alım-satım bölgelerindeki hisseleri göster
    oversold = sorted_results[sorted_results['Current RSI'] < 30].reset_index(drop=True)
    overbought = sorted_results[sorted_results['Current RSI'] > 70].reset_index(drop=True)
    
    if not oversold.empty:
        print("\nAşırı Satım Bölgesindeki Hisseler (RSI < 30):")
        print(oversold.to_string(index=False))
    else:
        print("\nAşırı Satım Bölgesinde hisse bulunmuyor.")
    
    if not overbought.empty:
        print("\nAşırı Alım Bölgesindeki Hisseler (RSI > 70):")
        print(overbought.to_string(index=False))
    else:
        print("\nAşırı Alım Bölgesinde hisse bulunmuyor.")
    
    # Örnek bir hisse için RSI grafiği çiz
    if not results.empty:
        example_ticker = results.iloc[0]['Ticker']
        print(f"\n{example_ticker} için örnek RSI grafiği çiziliyor...")
        plot_rsi_signals(example_ticker)