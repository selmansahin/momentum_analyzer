import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime, timedelta
import argparse
import warnings
from scipy.signal import argrelextrema
from bist_symbols import get_bist30_symbols, get_bist100_symbols

warnings.filterwarnings('ignore')

# Grafik ayarları
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def find_local_minima(data, window=5):
    """
    Bir veri serisindeki yerel minimum noktalarını bulur
    
    Parametreler:
    data (numpy.array): Minimum noktaları bulunacak veri serisi
    window (int): Minimum noktaları bulmak için kullanılacak pencere boyutu
    
    Dönüş:
    numpy.array: Minimum noktaların indeksleri
    """
    # Scipy'nin argrelextrema fonksiyonu ile yerel minimaları bul
    # order parametresi, bir noktanın minimum olarak kabul edilmesi için
    # her iki tarafında kaç nokta olması gerektiğini belirtir
    minima_indices = argrelextrema(data, np.less, order=window)[0]
    return minima_indices

def detect_divergence(price_data, indicator_data, window=10, lookback_periods=3):
    """
    Fiyat ve gösterge arasındaki pozitif uyumsuzlukları tespit eder
    
    Parametreler:
    price_data (numpy.array): Fiyat verisi
    indicator_data (numpy.array): Gösterge verisi (RSI, MACD, vb.)
    window (int): Minimum noktaları bulmak için kullanılacak pencere boyutu
    lookback_periods (int): Kaç dönem geriye bakılacağı
    
    Dönüş:
    list: Tespit edilen uyumsuzlukların listesi. Her uyumsuzluk için:
          [indeks, fiyat_değişimi, gösterge_değişimi, uyumsuzluk_gücü]
    """
    # Yerel minimaları bul
    price_minima = find_local_minima(price_data, window)
    indicator_minima = find_local_minima(indicator_data, window)
    
    # Eğer yeterli minimum nokta yoksa, boş liste döndür
    if len(price_minima) < 2 or len(indicator_minima) < 2:
        return []
    
    # Son lookback_periods kadar minimum noktayı al
    if len(price_minima) > lookback_periods:
        price_minima = price_minima[-lookback_periods:]
    if len(indicator_minima) > lookback_periods:
        indicator_minima = indicator_minima[-lookback_periods:]
    
    divergences = []
    
    # Fiyat minimumları arasında dolaş
    for i in range(1, len(price_minima)):
        price_idx1 = price_minima[i-1]
        price_idx2 = price_minima[i]
        
        # Fiyat daha düşük bir dip yapmış mı kontrol et
        if price_data[price_idx2] < price_data[price_idx1]:
            # Bu fiyat diplerinin yakınındaki gösterge diplerini bul
            # Gösterge diplerinin fiyat diplerinden önce veya sonra olabileceğini düşünerek
            # bir pencere içinde arama yapıyoruz
            search_window = window * 2
            
            # İlk fiyat dipine yakın gösterge dibi
            indicator_idx1 = None
            min_dist1 = search_window
            for idx in indicator_minima:
                dist = abs(idx - price_idx1)
                if dist < min_dist1:
                    min_dist1 = dist
                    indicator_idx1 = idx
            
            # İkinci fiyat dipine yakın gösterge dibi
            indicator_idx2 = None
            min_dist2 = search_window
            for idx in indicator_minima:
                dist = abs(idx - price_idx2)
                if dist < min_dist2:
                    min_dist2 = dist
                    indicator_idx2 = idx
            
            # Eğer her iki fiyat dibi için de gösterge dibi bulunmuşsa
            if indicator_idx1 is not None and indicator_idx2 is not None:
                # Gösterge daha yüksek bir dip yapmış mı kontrol et (pozitif uyumsuzluk)
                # Fiyat daha düşük dip yaparken, gösterge daha yüksek dip yapmalı
                # Gösterge değişimi pozitif olmalı
                indicator_change = (indicator_data[indicator_idx2] - indicator_data[indicator_idx1]) / abs(indicator_data[indicator_idx1]) * 100 if indicator_data[indicator_idx1] != 0 else 0
                if indicator_data[indicator_idx2] > indicator_data[indicator_idx1] and indicator_change > 0:
                    # Fiyat değişimi (negatif olmalı)
                    price_change = (price_data[price_idx2] - price_data[price_idx1]) / price_data[price_idx1] * 100
                    
                    # Uyumsuzluk gücü: gösterge değişimi - fiyat değişimi
                    # Fiyat ne kadar düşük, gösterge ne kadar yüksek dip yaparsa, uyumsuzluk o kadar güçlü
                    # Pozitif uyumsuzlukta fiyat değişimi genellikle negatif olacağından, mutlak değerini alarak
                    # uyumsuzluk gücünü pozitif bir değer olarak hesaplıyoruz
                    divergence_strength = indicator_change + abs(price_change)
                    
                    # Uyumsuzluk bilgilerini ekle
                    divergences.append({
                        'price_idx1': price_idx1,
                        'price_idx2': price_idx2,
                        'indicator_idx1': indicator_idx1,
                        'indicator_idx2': indicator_idx2,
                        'price_change': float(price_change),
                        'indicator_change': float(indicator_change),
                        'divergence_strength': float(divergence_strength)
                    })
    
    # Uyumsuzlukları güçlerine göre sırala
    divergences.sort(key=lambda x: x['divergence_strength'], reverse=True)
    
    return divergences

def calculate_rsi(prices, window=14):
    """
    RSI (Relative Strength Index) hesaplar
    
    Parametreler:
    prices (numpy.array): Fiyat verisi
    window (int): RSI hesaplama penceresi
    
    Dönüş:
    numpy.array: RSI değerleri
    """
    # Fiyat değişimlerini hesapla
    delta = np.diff(prices)
    delta = np.insert(delta, 0, 0)  # İlk eleman için 0 ekle
    
    # Pozitif ve negatif değişimleri ayır
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Ortalama kazanç ve kayıpları hesapla
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # İlk pencere için basit ortalama
    if len(gain) >= window:
        avg_gain[window-1] = np.mean(gain[0:window])
        avg_loss[window-1] = np.mean(loss[0:window])
        
        # Sonraki değerler için üstel ortalama
        for i in range(window, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i]) / window
    
    # RS ve RSI hesaplama
    rs = np.zeros_like(prices)
    rsi = np.zeros_like(prices)
    
    for i in range(window, len(prices)):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
        rsi[i] = 100 - (100 / (1 + rs[i]))
    
    return rsi

def download_stock_data(symbol, period='1d', lookback_days=180):
    """
    Yahoo Finance'dan hisse senedi verilerini indirir
    
    Parametreler:
    symbol (str): Hisse senedi sembolü
    period (str): Veri periyodu ('1h': saatlik, '1d': günlük, vb.)
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    
    Dönüş:
    pandas.DataFrame: İndirilen veri
    """
    try:
        # Veri çekme
        end_date = datetime.now()
        
        # 15 dakikalık veri için Yahoo Finance sınırlaması: son 60 gün
        if period == '15m' and lookback_days > 60:
            actual_lookback = 60
        else:
            actual_lookback = lookback_days
            
        start_date = end_date - timedelta(days=actual_lookback)
        
        # Veriyi çek
        data = yf.download(symbol, start=start_date, end=end_date, interval=period, progress=False)
        
        # Minimum veri sayısı kontrolü
        if period == '15m':
            min_data_points = 400  # 15 dakikalık veriler için daha fazla veri noktası
        elif period == '1h':
            min_data_points = 100  # Saatlik veriler için
        else:
            min_data_points = 50   # Günlük ve diğer periyotlar için
        
        if data.empty or len(data) < min_data_points:
            return None
        
        return data
    except Exception as e:
        print(f"Veri indirme hatası: {str(e)}")
        return None

def download_bist_stocks(endeks="bist30"):
    """
    BIST hisselerinin sembollerini alır
    
    Parametreler:
    endeks (str): BIST endeksi ('bist30', 'bist100', vb.)
    
    Dönüş:
    list: Hisse sembollerinin listesi
    """
    try:
        if endeks.lower() == "bist30":
            symbols = get_bist30_symbols()
        elif endeks.lower() == "bist100":
            symbols = get_bist100_symbols()
        else:
            return None
        
        # Sembolleri dönüştür
        stocks = [{'symbol': symbol} for symbol in symbols]
        return stocks
    except Exception as e:
        print(f"BIST hisselerini alma hatası: {str(e)}")
        return None

def calculate_macd(prices, fast_period=12, slow_period=26, signal_period=9):
    """
    MACD (Moving Average Convergence Divergence) hesaplar
    
    Parametreler:
    prices (numpy.array): Fiyat verisi
    fast_period (int): Hızlı EMA penceresi
    slow_period (int): Yavaş EMA penceresi
    signal_period (int): Sinyal EMA penceresi
    
    Dönüş:
    tuple: (macd, signal, histogram)
    """
    # Hızlı ve yavaş EMA'ları hesapla
    ema_fast = np.zeros_like(prices)
    ema_slow = np.zeros_like(prices)
    
    # İlk değerleri basit ortalama olarak ayarla
    ema_fast[fast_period-1] = np.mean(prices[:fast_period])
    ema_slow[slow_period-1] = np.mean(prices[:slow_period])
    
    # EMA'ları hesapla
    alpha_fast = 2 / (fast_period + 1)
    alpha_slow = 2 / (slow_period + 1)
    
    for i in range(fast_period, len(prices)):
        ema_fast[i] = prices[i] * alpha_fast + ema_fast[i-1] * (1 - alpha_fast)
    
    for i in range(slow_period, len(prices)):
        ema_slow[i] = prices[i] * alpha_slow + ema_slow[i-1] * (1 - alpha_slow)
    
    # MACD çizgisini hesapla
    macd = ema_fast - ema_slow
    
    # Sinyal çizgisini hesapla (MACD'nin EMA'sı)
    signal = np.zeros_like(macd)
    signal[slow_period+signal_period-1] = np.mean(macd[slow_period-1:slow_period+signal_period-1])
    
    alpha_signal = 2 / (signal_period + 1)
    for i in range(slow_period+signal_period, len(macd)):
        signal[i] = macd[i] * alpha_signal + signal[i-1] * (1 - alpha_signal)
    
    # Histogramı hesapla
    histogram = macd - signal
    
    return macd, signal, histogram

def analyze_stock_divergence(symbol, period='1d', lookback_days=180, min_window=5, divergence_window=20, recent_days=None, verbose=False):
    """
    Bir hisse senedi için pozitif uyumsuzluk analizi yapar
    
    Parametreler:
    symbol (str): Analiz edilecek hisse senedi sembolü
    period (str): Veri periyodu ('1h': saatlik, '1d': günlük, vb.)
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    min_window (int): Minimum noktaları bulmak için kullanılacak pencere boyutu
    divergence_window (int): Uyumsuzluk tespitinde kullanılacak pencere boyutu
    recent_days (int): Son kaç gün içindeki uyumsuzlukları göster (None: tüm uyumsuzluklar)
    verbose (bool): Detaylı bilgi gösterilip gösterilmeyeceği
    
    Dönüş:
    dict: Analiz sonuçlarını içeren sözlük
    """
    if verbose:
        print(f"{symbol} için {period} periyodunda pozitif uyumsuzluk analizi yapılıyor...")
    
    # Veri indir
    data = download_stock_data(symbol, period=period, lookback_days=lookback_days)
    
    if data is None or len(data) < 30:
        if verbose:
            print(f"Yeterli veri bulunamadı: {symbol}")
        return None
    
    if verbose:
        print(f"Toplam {len(data)} {period} veri çekildi")
        print(f"En son veri tarihi: {data.index[-1]} (Yahoo Finance verileri genellikle 15 dakika gecikmeli)")
    
    # Teknik göstergeleri hesapla
    close_prices = data['Close'].values
    rsi_values = calculate_rsi(close_prices)
    macd_values, signal_values, histogram_values = calculate_macd(close_prices)
    
    # Yerel minimum noktalarını bul
    price_mins = find_local_minima(close_prices, window=min_window)
    rsi_mins = find_local_minima(rsi_values, window=min_window)
    macd_mins = find_local_minima(macd_values, window=min_window)
    
    # Uyumsuzlukları tespit et
    rsi_divergences = detect_divergence(close_prices, rsi_values, window=divergence_window, lookback_periods=3)
    macd_divergences = detect_divergence(close_prices, macd_values, window=divergence_window, lookback_periods=3)
    
    # Son recent_days gün içindeki uyumsuzlukları filtrele
    if recent_days is not None and len(data) > 0:
        last_date = data.index[-1]
        
        # RSI uyumsuzluklarını filtrele
        filtered_rsi_divergences = []
        for div in rsi_divergences:
            price_idx2 = div['price_idx2']
            div_date = data.index[price_idx2]
            days_diff = (last_date - div_date).days
            if days_diff <= recent_days:
                filtered_rsi_divergences.append(div)
        rsi_divergences = filtered_rsi_divergences
        
        # MACD uyumsuzluklarını filtrele
        filtered_macd_divergences = []
        for div in macd_divergences:
            price_idx2 = div['price_idx2']
            div_date = data.index[price_idx2]
            days_diff = (last_date - div_date).days
            if days_diff <= recent_days:
                filtered_macd_divergences.append(div)
        macd_divergences = filtered_macd_divergences
    
    # Uyumsuzluk sayılarını ve durumlarını hesapla
    rsi_divergence_count = len(rsi_divergences)
    macd_divergence_count = len(macd_divergences)
    total_divergence_count = rsi_divergence_count + macd_divergence_count
    has_rsi_divergence = rsi_divergence_count > 0
    has_macd_divergence = macd_divergence_count > 0
    
    # Son veri tarihini kaydet
    last_date = data.index[-1] if len(data) > 0 else None
    
    # Sonuçları dön
    return {
        'Symbol': symbol,
        'Period': period,
        'Data': data,
        'Close_Prices': close_prices,
        'RSI_Values': rsi_values,
        'MACD_Values': macd_values,
        'Signal_Values': signal_values,
        'Histogram_Values': histogram_values,
        'RSI_Divergences': rsi_divergences,
        'MACD_Divergences': macd_divergences,
        'Divergence_Count': total_divergence_count,
        'Has_RSI_Divergence': has_rsi_divergence,
        'Has_MACD_Divergence': has_macd_divergence,
        'Last_Date': last_date
    }

def scan_bist_for_divergences(period='1d', endeks="bist30", lookback_days=180, min_window=5, divergence_window=20, recent_days=None, verbose=False):
    """
    BIST hisselerini tarar ve pozitif uyumsuzluk gösterenleri bulur
    
    Parametreler:
    period (str): Veri periyodu ('1h': saatlik, '1d': günlük, vb.)
    endeks (str): BIST endeksi ('bist30', 'bist100', vb.)
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    min_window (int): Minimum noktaları bulmak için kullanılacak pencere boyutu
    divergence_window (int): Uyumsuzluk tespitinde kullanılacak pencere boyutu
    recent_days (int): Son kaç gün içindeki uyumsuzlukları göster (None: tüm uyumsuzluklar)
    verbose (bool): Detaylı bilgi gösterilip gösterilmeyeceği
    
    Dönüş:
    list: Uyumsuzluk gösteren hisselerin listesi
    """
    try:
        # BIST sembollerini al
        if endeks.lower() == "bist30":
            symbols = get_bist30_symbols()
        elif endeks.lower() == "bist100":
            symbols = get_bist100_symbols()
        else:
            if verbose:
                print(f"Geçersiz endeks: {endeks}. Desteklenen endeksler: bist30, bist100")
            return []
        
        if not symbols:
            if verbose:
                print(f"Semboller alınamadı: {endeks}")
            return []
        
        if verbose:
            print(f"Toplam {len(symbols)} hisse {period} periyodunda pozitif uyumsuzluk için taranacak...")
        
        # Sonuçları saklamak için liste
        candidates = []
        
        # Tüm sembolleri tara
        for symbol in symbols:
            if verbose:
                print(f"{symbol} taranıyor...", end="")
            
            # Hisse için uyumsuzluk analizi yap
            result = analyze_stock_divergence(symbol, period=period, lookback_days=lookback_days, 
                                            min_window=min_window, divergence_window=divergence_window,
                                            recent_days=recent_days, verbose=False)
            
            if result is None:
                if verbose:
                    print(" Analiz yapılamadı")
                continue
            
            # Uyumsuzluk sayısı
            rsi_count = len(result['RSI_Divergences'])
            macd_count = len(result['MACD_Divergences'])
            total_count = rsi_count + macd_count
            
            if verbose:
                if total_count > 0:
                    div_types = []
                    if rsi_count > 0:
                        div_types.append("RSI")
                    if macd_count > 0:
                        div_types.append("MACD")
                    div_str = ", ".join(div_types)
                    print(f" {total_count} uyumsuzluk bulundu: {div_str}")
                else:
                    print(" 0 uyumsuzluk bulundu: Yok")
            
            # Uyumsuzluk varsa listeye ekle
            if total_count > 0:
                # Uyumsuzluk gücünü hesapla
                max_strength = 0
                if rsi_count > 0:
                    max_strength = max([div['divergence_strength'] for div in result['RSI_Divergences']])
                if macd_count > 0:
                    macd_strength = max([div['divergence_strength'] for div in result['MACD_Divergences']])
                    max_strength = max(max_strength, macd_strength)
                
                candidates.append({
                    'Symbol': symbol,
                    'Count': total_count,
                    'RSI_Count': rsi_count,
                    'MACD_Count': macd_count,
                    'Last_Close': result['Last_Close'],
                    'RSI': result['RSI'],
                    'MACD': result['MACD'],
                    'Signal': result['Signal'],
                    'Strength': max_strength,
                    'Has_RSI_Divergence': rsi_count > 0,
                    'Has_MACD_Divergence': macd_count > 0,
                    'RSI_Divergences': result['RSI_Divergences'],
                    'MACD_Divergences': result['MACD_Divergences']
                })
        
        # Uyumsuzluk gücüne göre sırala (en güçlü uyumsuzluk en üste)
        sorted_candidates = sorted(candidates, key=lambda x: x['Strength'], reverse=True)
        
        # Sonuçları göster
        if verbose:
            print(f"\nTüm BIST Hisselerinin {period} Periyodunda Pozitif Uyumsuzluk Analizi:")
            print("-" * 80)
            
            for res in sorted_candidates:
                symbol = res['Symbol']
                count = res['Count']
                price = res['Last_Close']
                rsi = res['RSI']
                
                divergence_types = []
                if res['Has_RSI_Divergence']:
                    divergence_types.append("RSI")
                if res['Has_MACD_Divergence']:
                    divergence_types.append("MACD")
                    
                divergence_str = ", ".join(divergence_types) if divergence_types else "Yok"
                print(f"{symbol}: {count} uyumsuzluk - Fiyat: {price:.2f}, RSI: {rsi:.2f} - Uyumsuzluklar: {divergence_str}")
            
            # Sadece uyumsuzluk gösterenleri listele
            if sorted_candidates:
                print(f"\nPozitif Uyumsuzluk Gösteren Hisseler ({period} periyodu):")
                print("-" * 80)
                
                for res in sorted_candidates:
                    symbol = res['Symbol']
                    count = res['Count']
                    price = res['Last_Close']
                    rsi = res['RSI']
                    
                    divergence_types = []
                    if res['Has_RSI_Divergence']:
                        divergence_types.append("RSI")
                    if res['Has_MACD_Divergence']:
                        divergence_types.append("MACD")
                        
                    divergence_str = ", ".join(divergence_types) if divergence_types else "Yok"
                    print(f"{symbol}: {count} uyumsuzluk - Fiyat: {price:.2f}, RSI: {rsi:.2f} - Uyumsuzluklar: {divergence_str}")
        
        return sorted_candidates
    except Exception as e:
        if verbose:
            print(f"Tarama hatası: {str(e)}")
            import traceback
            traceback.print_exc()
        return []

def plot_divergence(results, save_path=None, show_plot=True):
    """
    Pozitif uyumsuzluk grafikleri çizer
    
    Parametreler:
    results (dict): analyze_stock_divergence fonksiyonundan dönen sonuçlar
    save_path (str): Grafiğin kaydedileceği dosya yolu (opsiyonel)
    show_plot (bool): Grafiğin gösterilip gösterilmeyeceği
    """
    if results is None:
        print("Grafik çizilemiyor: Sonuç verisi yok")
        return
    
    symbol = results['Symbol']
    period = results['Period']
    data = results['Data']
    close_prices = results['Close_Prices']
    rsi_values = results['RSI_Values']
    macd_values = results['MACD_Values']
    signal_values = results['Signal_Values']
    histogram_values = results['Histogram_Values']
    rsi_divergences = results['RSI_Divergences']
    macd_divergences = results['MACD_Divergences']
    
    # Tarih dizisi
    dates = data.index
    
    try:
        # Tarih indeksi yerine sayısal indeks kullan
        x_indices = list(range(len(dates)))
        
        # Tek bir figur oluştur
        plt.figure(figsize=(12, 12))
        
        # 1. Fiyat grafiği
        plt.subplot(3, 1, 1)
        plt.title(f"{symbol} - Pozitif Uyumsuzluk: {dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")
        plt.plot(x_indices, close_prices, label='Fiyat', color='blue')
        plt.ylabel('Fiyat')
        plt.grid(True)
        
        # MACD uyumsuzluklarını fiyat grafiğinde işaretle
        for div in macd_divergences:
            price_idx1 = div['price_idx1']
            price_idx2 = div['price_idx2']
            
            # Dip noktalarını işaretle
            plt.plot(price_idx1, close_prices[price_idx1], 'ro', markersize=5)
            plt.plot(price_idx2, close_prices[price_idx2], 'ro', markersize=5)
            
            # Dip noktaları arasına çizgi çiz
            plt.plot([price_idx1, price_idx2], 
                    [close_prices[price_idx1], close_prices[price_idx2]], 
                    'r--', linewidth=1.5)
        
        # X eksenini gösterme
        plt.xticks([])
        
        # 2. RSI grafiği
        plt.subplot(3, 1, 2)
        plt.title("RSI Göstergesi")
        plt.plot(x_indices, rsi_values, label='RSI', color='purple')
        plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
        plt.axhline(y=50, color='b', linestyle='--', alpha=0.3)
        plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
        plt.ylim(0, 100)
        plt.ylabel('RSI')
        plt.grid(True)
        
        # RSI uyumsuzluklarını işaretle
        for div in rsi_divergences:
            indicator_idx1 = div['indicator_idx1']
            indicator_idx2 = div['indicator_idx2']
            
            # RSI'daki dip noktalarını işaretle
            plt.plot(indicator_idx1, rsi_values[indicator_idx1], 'ro', markersize=5)
            plt.plot(indicator_idx2, rsi_values[indicator_idx2], 'ro', markersize=5)
            
            # RSI'daki dip noktaları arasına çizgi çiz
            plt.plot([indicator_idx1, indicator_idx2], 
                    [rsi_values[indicator_idx1], rsi_values[indicator_idx2]], 
                    'g-', linewidth=1.5)
        
        # X eksenini gösterme
        plt.xticks([])
        
        # 3. MACD grafiği
        plt.subplot(3, 1, 3)
        plt.title("MACD Göstergesi")
        plt.plot(x_indices, macd_values, label='MACD', color='green')
        plt.plot(x_indices, signal_values, label='Sinyal', color='red')
        
        # Histogram çizimi - basit çizgi olarak göster
        for i in range(len(histogram_values)):
            if histogram_values[i] >= 0:
                plt.plot([i, i], [0, histogram_values[i]], 'g-', alpha=0.3)
            else:
                plt.plot([i, i], [0, histogram_values[i]], 'r-', alpha=0.3)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        plt.ylabel('MACD')
        plt.grid(True)
        plt.legend(loc='upper left')
        
        # MACD uyumsuzluklarını işaretle
        for div in macd_divergences:
            indicator_idx1 = div['indicator_idx1']
            indicator_idx2 = div['indicator_idx2']
            
            # MACD'deki dip noktalarını işaretle
            plt.plot(indicator_idx1, macd_values[indicator_idx1], 'bo', markersize=5)
            plt.plot(indicator_idx2, macd_values[indicator_idx2], 'bo', markersize=5)
            
            # MACD'deki dip noktaları arasına çizgi çiz
            plt.plot([indicator_idx1, indicator_idx2], 
                    [macd_values[indicator_idx1], macd_values[indicator_idx2]], 
                    'g-', linewidth=1.5)
        
        # X ekseni etiketlerini göster - sadece önemli tarihler
        step = len(x_indices) // 10  # 10 etiket göster
        if step < 1:
            step = 1
        xtick_positions = x_indices[::step]
        xtick_labels = [dates[i].strftime('%Y-%m-%d') for i in xtick_positions]
        plt.xticks(xtick_positions, xtick_labels, rotation=45)
        
        # Grafikleri sıkıştır
        plt.tight_layout()
        
        # Grafiği kaydet
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grafik kaydedildi: {save_path}")
        
        # Grafiği göster
        if show_plot:
            plt.show()
        else:
            plt.close('all')
            
    except Exception as e:
        print(f"Grafik çizim hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        # Hata durumunda tüm grafikleri kapat
        plt.close('all')

def scan_bist_for_divergences(period='1d', endeks="bist30", lookback_days=180, min_window=5, divergence_window=20, recent_days=None, verbose=False):
    """
    BIST hisselerini tarar ve pozitif uyumsuzluk gösterenleri bulur
    
    Parametreler:
    period (str): Veri periyodu ('15m': 15 dakikalık, '1h': saatlik, '1d': günlük, vb.)
    endeks (str): BIST hisseleri için endeks ('bist30' veya 'bist100')
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    min_window (int): Minimum noktaları bulmak için kullanılacak pencere boyutu
    divergence_window (int): Uyumsuzluk tespitinde kullanılacak pencere boyutu
    recent_days (int): Son kaç gün içindeki uyumsuzlukları göster (None: tüm uyumsuzluklar)
    verbose (bool): Detaylı çıktı gösterilip gösterilmeyeceği
    
    Dönüş:
    list: Pozitif uyumsuzluk gösteren hisselerin listesi
    """
    symbols = get_bist100_symbols() if endeks == "bist100" else get_bist30_symbols()

    divergence_candidates = []
    all_results = []
    latest_data_timestamp = None
    
    print(f"Toplam {len(symbols)} hisse {period} periyodunda pozitif uyumsuzluk için taranacak...")
    
    for symbol in symbols:
        print(f"{symbol} taranıyor...", end="")
        
        result = analyze_stock_divergence(symbol, period=period, lookback_days=lookback_days, 
                                         min_window=min_window, divergence_window=divergence_window, 
                                         recent_days=recent_days, verbose=False)
        
        if result is None:
            print(" Veri yok veya işlenemedi")
            continue
            
        # Uyumsuzluk sayısını al
        divergence_count = result['Divergence_Count']
        has_rsi_div = result['Has_RSI_Divergence']
        has_macd_div = result['Has_MACD_Divergence']
        
        # Hangi uyumsuzlukların tespit edildiğini göster
        divergence_types = []
        if has_rsi_div:
            divergence_types.append("RSI")
        if has_macd_div:
            divergence_types.append("MACD")
        
        divergence_str = ", ".join(divergence_types) if divergence_types else "Yok"
        print(f" {divergence_count} uyumsuzluk bulundu: {divergence_str}")
        
        # Tüm sonuçları kaydet
        all_results.append(result)

        # En son veri tarihini güncelle (eğer daha yeni ise)
        if 'Last_Date' in result and (latest_data_timestamp is None or result['Last_Date'] > latest_data_timestamp):
            latest_data_timestamp = result['Last_Date']
            
        if divergence_count > 0:
            divergence_candidates.append(result)

    # Basit bir şekilde sonuçları yazdır
    print(f"\nTüm BIST Hisselerinin {period} Periyodunda Pozitif Uyumsuzluk Analizi:")
    print("-" * 80)
    
    # Sonuçları uyumsuzluk sayısına göre sırala
    sorted_results = sorted(all_results, key=lambda x: x['Divergence_Count'], reverse=True)
    
    for res in sorted_results:
        symbol = res['Symbol']
        count = res['Divergence_Count']
        rsi = float(res['RSI_Values'][-1]) if len(res['RSI_Values']) > 0 else 0.0
        price = float(res['Close_Prices'][-1]) if len(res['Close_Prices']) > 0 else 0.0
        
        divergence_types = []
        if res['Has_RSI_Divergence']:
            divergence_types.append("RSI")
        if res['Has_MACD_Divergence']:
            divergence_types.append("MACD")
            
        divergence_str = ", ".join(divergence_types) if divergence_types else "Yok"
        print(f"{symbol}: {count} uyumsuzluk - Fiyat: {price:.2f}, RSI: {rsi:.2f} - Uyumsuzluklar: {divergence_str}")
    
    # En son veri tarihini göster (verbose modunda)
    if verbose:
        print(f"\nEn son veri tarihi: {latest_data_timestamp} (Yahoo Finance verileri genellikle 15 dakika gecikmeli)")
    
    # Uyumsuzluk adaylarını döndür
    if divergence_candidates:
        # Sonuçları uyumsuzluk sayısına göre sırala
        sorted_candidates = sorted(divergence_candidates, key=lambda x: x['Divergence_Count'], reverse=True)
        
        print(f"\nPozitif Uyumsuzluk Gösteren Hisseler ({period} periyodu):")
        print("-" * 80)
        
        for res in sorted_candidates:
            symbol = res['Symbol']
            count = res['Divergence_Count']
            rsi = float(res['RSI_Values'][-1]) if len(res['RSI_Values']) > 0 else 0.0
            price = float(res['Close_Prices'][-1]) if len(res['Close_Prices']) > 0 else 0.0
            
            divergence_types = []
            if res['Has_RSI_Divergence']:
                divergence_types.append("RSI")
            if res['Has_MACD_Divergence']:
                divergence_types.append("MACD")
                
            divergence_str = ", ".join(divergence_types) if divergence_types else "Yok"
            print(f"{symbol}: {count} uyumsuzluk - Fiyat: {price:.2f}, RSI: {rsi:.2f} - Uyumsuzluklar: {divergence_str}")
        
        return sorted_candidates
    else:
        print("\nPozitif uyumsuzluk gösteren hisse bulunamadı.")
        return []

def analyze_single_stock_divergence(symbol, period='1d', lookback_days=180, min_window=5, divergence_window=20, recent_days=None, plot=True, save_plot=False):
    """
    Tek bir hisse için detaylı pozitif uyumsuzluk analizi yapar ve sonuçları gösterir
    
    Parametreler:
    symbol (str): Analiz edilecek hisse senedi sembolü
    period (str): Veri periyodu ('1h': saatlik, '1d': günlük, vb.)
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    min_window (int): Minimum noktaları bulmak için kullanılacak pencere boyutu
    divergence_window (int): Uyumsuzluk tespitinde kullanılacak pencere boyutu
    recent_days (int): Son kaç gün içindeki uyumsuzlukları göster (None: tüm uyumsuzluklar)
    plot (bool): Grafik çizilip çizilmeyeceği
    save_plot (bool): Grafiğin kaydedilip kaydedilmeyeceği
    """
    print(f"\n{symbol} için {period} periyodunda detaylı pozitif uyumsuzluk analizi:")
    print("-" * 60)
    
    result = analyze_stock_divergence(symbol, period=period, lookback_days=lookback_days, 
                                     min_window=min_window, divergence_window=divergence_window, 
                                     recent_days=recent_days, verbose=True)
    
    if result is None:
        print("Analiz yapılamadı: Yeterli veri yok veya işlenemedi")
        return
    
    # Uyumsuzluk bilgilerini göster
    print("\nPozitif Uyumsuzluk Analizi:")
    print("-" * 40)
    
    rsi_divergences = result['RSI_Divergences']
    macd_divergences = result['MACD_Divergences']
    
    print(f"RSI Pozitif Uyumsuzluk: {len(rsi_divergences)} adet")
    print(f"MACD Pozitif Uyumsuzluk: {len(macd_divergences)} adet")
    
    # RSI uyumsuzluklarının detaylarını göster
    if rsi_divergences:
        print("\nRSI Pozitif Uyumsuzluk Detayları:")
        for i, div in enumerate(rsi_divergences):
            print(f"  {i+1}. Uyumsuzluk:")
            print(f"    Fiyat Değişimi: {div['price_change']:.2f}%")
            print(f"    RSI Değişimi: {div['indicator_change']:.2f}%")
            print(f"    Uyumsuzluk Gücü: {div['divergence_strength']:.2f}")
    
    # MACD uyumsuzluklarının detaylarını göster
    if macd_divergences:
        print("\nMACD Pozitif Uyumsuzluk Detayları:")
        for i, div in enumerate(macd_divergences):
            print(f"  {i+1}. Uyumsuzluk:")
            print(f"    Fiyat Değişimi: {div['price_change']:.2f}%")
            print(f"    MACD Değişimi: {div['indicator_change']:.2f}%")
            print(f"    Uyumsuzluk Gücü: {div['divergence_strength']:.2f}")
    
    # Grafik çiz
    if plot:
        try:
            # Tek bir figur oluştur
            plt.figure(figsize=(12, 16))
            
            # Fiyat, RSI ve MACD grafiklerini alt alta çiz
            plt.subplot(3, 1, 1)  # Fiyat
            plt.title(f"{symbol} - Pozitif Uyumsuzluk: {result['Data'].index[0].strftime('%Y-%m-%d')} → {result['Data'].index[-1].strftime('%Y-%m-%d')}")
            plt.plot(result['Data'].index, result['Close_Prices'], label='Fiyat', color='blue')
            plt.ylabel('Fiyat')
            plt.grid(True)
            
            # MACD uyumsuzluklarını fiyat grafiğinde işaretle
            for div in macd_divergences:
                price_idx1 = div['price_idx1']
                price_idx2 = div['price_idx2']
                
                # Dip noktalarını işaretle
                plt.plot(result['Data'].index[price_idx1], result['Close_Prices'][price_idx1], 'ro', markersize=5)
                plt.plot(result['Data'].index[price_idx2], result['Close_Prices'][price_idx2], 'ro', markersize=5)
                
                # Dip noktaları arasına çizgi çiz
                plt.plot([result['Data'].index[price_idx1], result['Data'].index[price_idx2]], 
                        [result['Close_Prices'][price_idx1], result['Close_Prices'][price_idx2]], 
                        'r--', linewidth=1.5)
            
            # RSI grafiği
            plt.subplot(3, 1, 2)  # RSI
            plt.title("RSI Göstergesi")
            plt.plot(result['Data'].index, result['RSI_Values'], label='RSI', color='purple')
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
            plt.axhline(y=50, color='b', linestyle='--', alpha=0.3)
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
            plt.ylim(0, 100)
            plt.ylabel('RSI')
            plt.grid(True)
            
            # RSI uyumsuzluklarını işaretle
            for div in rsi_divergences:
                indicator_idx1 = div['indicator_idx1']
                indicator_idx2 = div['indicator_idx2']
                
                # RSI'daki dip noktalarını işaretle
                plt.plot(result['Data'].index[indicator_idx1], result['RSI_Values'][indicator_idx1], 'ro', markersize=5)
                plt.plot(result['Data'].index[indicator_idx2], result['RSI_Values'][indicator_idx2], 'ro', markersize=5)
                
                # RSI'daki dip noktaları arasına çizgi çiz
                plt.plot([result['Data'].index[indicator_idx1], result['Data'].index[indicator_idx2]], 
                        [result['RSI_Values'][indicator_idx1], result['RSI_Values'][indicator_idx2]], 
                        'g-', linewidth=1.5)
            
            # MACD grafiği
            plt.subplot(3, 1, 3)  # MACD
            plt.title("MACD Göstergesi")
            plt.plot(result['Data'].index, result['MACD_Values'], label='MACD', color='green')
            plt.plot(result['Data'].index, result['Signal_Values'], label='Sinyal', color='red')
            
            # Histogram çizimi
            for i in range(len(result['Data'].index)):
                if i < len(result['Histogram_Values']):
                    if result['Histogram_Values'][i] >= 0:
                        plt.bar(result['Data'].index[i], result['Histogram_Values'][i], color='green', alpha=0.3, width=1)
                    else:
                        plt.bar(result['Data'].index[i], result['Histogram_Values'][i], color='red', alpha=0.3, width=1)
            
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            plt.ylabel('MACD')
            plt.grid(True)
            plt.legend(loc='upper left')
            
            # MACD uyumsuzluklarını işaretle
            for div in macd_divergences:
                indicator_idx1 = div['indicator_idx1']
                indicator_idx2 = div['indicator_idx2']
                
                # MACD'deki dip noktalarını işaretle
                plt.plot(result['Data'].index[indicator_idx1], result['MACD_Values'][indicator_idx1], 'bo', markersize=5)
                plt.plot(result['Data'].index[indicator_idx2], result['MACD_Values'][indicator_idx2], 'bo', markersize=5)
                
                # MACD'deki dip noktaları arasına çizgi çiz
                plt.plot([result['Data'].index[indicator_idx1], result['Data'].index[indicator_idx2]], 
                        [result['MACD_Values'][indicator_idx1], result['MACD_Values'][indicator_idx2]], 
                        'g-', linewidth=1.5)
            
            # Grafikleri sıkıştır
            plt.tight_layout()
            
            # Grafiği kaydet
            if save_plot:
                plt.savefig(f"{symbol.replace('.IS', '')}_divergence_{period}.png", dpi=300, bbox_inches='tight')
                print(f"Grafik kaydedildi: {symbol.replace('.IS', '')}_divergence_{period}.png")
            
            # Grafiği göster
            plt.show()
                
        except Exception as e:
            print(f"Grafik çizim hatası: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """
    Ana fonksiyon
    """
    print("Türkiye Borsası (BIST) Pozitif Uyumsuzluk Tarama Sistemi")
    print("=" * 50)
    print("Pozitif uyumsuzluk, fiyat daha düşük seviyede dip yaparken,")
    print("göstergenin (RSI, MACD) daha yüksek dip yapması durumudur.")
    print("Bu genellikle bir alım fırsatı olarak değerlendirilir.")
    print("=" * 50)
    print("Desteklenen periyotlar: 15m (15 dakikalık - son 60 gün), 1h (saatlik), 1d (günlük), 5d, 1wk, 1mo")
    print("=" * 50)
    
    # Komut satırı argümanlarını işleme
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='BIST Pozitif Uyumsuzluk Analizi')
    parser.add_argument('command', choices=['scan', 'analyze'], help='Yapılacak işlem: scan (tüm hisseleri tara) veya analyze (tek hisse analiz et)')
    parser.add_argument('--symbol', help='Analiz edilecek hisse sembolü (sadece analyze komutu için)')
    parser.add_argument('--period', default='1d', choices=['15m', '1h', '1d', '5d', '1wk', '1mo'], help='Veri periyodu (15m: 15 dakikalık - son 60 gün, 1h: saatlik, 1d: günlük, vb.)')
    parser.add_argument('--lookback', type=int, default=180, help='Kaç günlük geçmiş veri alınacağı')
    parser.add_argument('--min-window', type=int, default=5, help='Minimum noktaları bulmak için kullanılacak pencere boyutu')
    parser.add_argument('--div-window', type=int, default=20, help='Uyumsuzluk tespitinde kullanılacak pencere boyutu')
    parser.add_argument('--recent-days', type=int, default=None, help='Son kaç gün içindeki uyumsuzlukları göster (varsayılan: tüm uyumsuzluklar)')
    parser.add_argument('--endeks', default='bist30', choices=['bist30', 'bist100'], help='Taranacak endeks (bist30 veya bist100)')
    parser.add_argument('--verbose', action='store_true', help='Detaylı çıktı göster')
    parser.add_argument('--save-plot', action='store_true', help='Grafikleri kaydet')
    parser.add_argument('--no-plot', action='store_true', help='Grafikleri gösterme')
    
    # Eğer argüman yoksa, yardım mesajını göster
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        # Tüm BIST hisselerini tara
        print(f"\nTüm {args.endeks.upper()} hisseleri {args.period} periyodunda taranıyor...")
        results = scan_bist_for_divergences(period=args.period, endeks=args.endeks, 
                                          lookback_days=args.lookback, min_window=args.min_window, 
                                          divergence_window=args.div_window, recent_days=args.recent_days,
                                          verbose=args.verbose)
        
        if results:
            # En yüksek puanlı hisseyi analiz et
            top_stock = results[0]['Symbol']
            print(f"\nEn güçlü pozitif uyumsuzluk gösteren hisse: {top_stock}")
            analyze_single_stock_divergence(top_stock, period=args.period, lookback_days=args.lookback, 
                                          min_window=args.min_window, divergence_window=args.div_window, 
                                          recent_days=args.recent_days, plot=not args.no_plot, 
                                          save_plot=args.save_plot)
            
    elif args.command == 'analyze':
        if not args.symbol:
            print("Hata: analyze komutu için --symbol parametresi gereklidir")
            sys.exit(1)
            
        # Tek bir hisseyi analiz et
        symbol = args.symbol.upper()
        if not symbol.endswith('.IS'):
            symbol += '.IS'
        analyze_single_stock_divergence(symbol, period=args.period, lookback_days=args.lookback, 
                                      min_window=args.min_window, divergence_window=args.div_window, 
                                      recent_days=args.recent_days, plot=not args.no_plot, 
                                      save_plot=args.save_plot)

if __name__ == "__main__":
    main()
