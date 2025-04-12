import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
from bist_symbols import get_bist30_symbols, get_bist100_symbols

warnings.filterwarnings('ignore')



def analyze_stock(symbol, period='1d', lookback_days=180, verbose=False):
    """
    Hisse senedi için momentum analizi yapar
    
    Parametreler:
    symbol (str): Analiz edilecek hisse senedi sembolü
    period (str): Veri periyodu ('1h': saatlik, '1d': günlük, vb.)
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    verbose (bool): Detaylı çıktı gösterilip gösterilmeyeceği
    """
    if verbose:
        print(f"{symbol} için {period} periyodunda momentum analizi yapılıyor...")
    
    # Veri çekme
    end_date = datetime.now()
    
    # 15 dakikalık veri için Yahoo Finance sınırlaması: son 60 gün
    if period == '15m' and lookback_days > 60:
        actual_lookback = 60
        if verbose:
            print(f"Not: 15 dakikalık veri için Yahoo Finance sadece son 60 gün veri sağlamaktadır. {lookback_days} gün yerine 60 gün kullanılıyor.")
    else:
        actual_lookback = lookback_days
        
    start_date = end_date - timedelta(days=actual_lookback)
    
    try:    
        # Veriyi çek
        data = yf.download(symbol, start=start_date, end=end_date, interval=period, progress=False)
        print(data)
        # Minimum veri sayısı kontrolü
        # Daha kısa periyotlar için daha fazla veri noktası olacaktır
        if period == '15m':
            min_data_points = 400  # 15 dakikalık veriler için daha fazla veri noktası
        elif period == '1h':
            min_data_points = 100  # Saatlik veriler için
        else:
            min_data_points = 50   # Günlük ve diğer periyotlar için
        
        if data.empty or len(data) < min_data_points:
            if verbose:
                print(f"Yeterli veri yok. Sadece {len(data)} veri noktası bulundu, minimum {min_data_points} gerekli.")
            return None
             
        last_date = data.index[-1]

        if verbose:
            print(f"Toplam {len(data)} {period} veri çekildi")
            print(f"En son veri tarihi: {last_date} (Yahoo Finance verileri genellikle 15 dakika gecikmeli)")
        
        # Teknik göstergeleri hesapla
        # RSI hesaplama
        close_prices = data['Close'].values
        delta = np.diff(close_prices)
        delta = np.insert(delta, 0, 0)
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Periyoda göre RSI pencere boyutunu ayarla
        rsi_window = 14
        if period == '15m':
            rsi_window = 96  # 15 dakikalık veriler için 96 (yaklaşık 1 gün)
        elif period == '1h':
            rsi_window = 24  # Saatlik veriler için 24 saat (yaklaşık 1 gün)
        
        # RSI penceresi için ortalama hesaplama
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # İlk pencere için basit ortalama
        if len(gain) >= rsi_window:
            avg_gain[rsi_window-1] = np.mean(gain[0:rsi_window])
            avg_loss[rsi_window-1] = np.mean(loss[0:rsi_window])
            
            # Pencereden sonrası için
            for i in range(rsi_window, len(close_prices)):
                avg_gain[i] = (avg_gain[i-1] * (rsi_window-1) + gain[i]) / rsi_window
                avg_loss[i] = (avg_loss[i-1] * (rsi_window-1) + loss[i]) / rsi_window
        
        # RS ve RSI hesaplama
        rs = np.zeros_like(close_prices)
        rsi = np.zeros_like(close_prices)
        
        for i in range(rsi_window, len(close_prices)):
            if avg_loss[i] == 0:
                rs[i] = 100
            else:
                rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        
        # Periyoda göre hareketli ortalama pencerelerini ayarla
        ma_short_window = 20
        ma_long_window = 50
        
        if period == '15m':
            ma_short_window = 20 * 26  # Yaklaşık 20 günlük MA (günde 6.5 saat işlem, her saat 4 periyot)
            ma_long_window = 50 * 26  # Yaklaşık 50 günlük MA
        elif period == '1h':
            ma_short_window = 20 * 6.5  # Yaklaşık 20 günlük MA (günde 6.5 saat işlem)
            ma_long_window = 50 * 6.5   # Yaklaşık 50 günlük MA
            
        # Tam sayıya yuvarla
        ma_short_window = int(ma_short_window)
        ma_long_window = int(ma_long_window)
            
        # Hareketli ortalamalar
        ma_short = np.zeros_like(close_prices)
        ma_long = np.zeros_like(close_prices)
        
        for i in range(ma_short_window, len(close_prices)):
            ma_short[i] = np.mean(close_prices[i-ma_short_window:i])
            
        for i in range(ma_long_window, len(close_prices)):
            ma_long[i] = np.mean(close_prices[i-ma_long_window:i])
        
        # MACD hesaplama - periyoda göre ayarla
        ema_short_window = 12
        ema_long_window = 26
        signal_window = 9
        
        if period == '1h':
            ema_short_window = 12 * 6.5  # Saatlik için yaklaşık 12 günlük EMA
            ema_long_window = 26 * 6.5   # Saatlik için yaklaşık 26 günlük EMA
            signal_window = 9 * 6.5      # Saatlik için yaklaşık 9 günlük sinyal
            
        # Tam sayıya yuvarla
        ema_short_window = int(ema_short_window)
        ema_long_window = int(ema_long_window)
        signal_window = int(signal_window)
        
        ema_short = np.zeros_like(close_prices)
        ema_long = np.zeros_like(close_prices)
        macd = np.zeros_like(close_prices)
        signal = np.zeros_like(close_prices)
        
        # İlk EMA değerleri
        if len(close_prices) >= ema_short_window:
            ema_short[ema_short_window-1] = np.mean(close_prices[0:ema_short_window])
        if len(close_prices) >= ema_long_window:
            ema_long[ema_long_window-1] = np.mean(close_prices[0:ema_long_window])
        
        # EMA hesaplama
        k_short = 2 / (ema_short_window + 1)
        k_long = 2 / (ema_long_window + 1)
        k_signal = 2 / (signal_window + 1)
        
        for i in range(ema_short_window, len(close_prices)):
            ema_short[i] = close_prices[i] * k_short + ema_short[i-1] * (1 - k_short)
            
        for i in range(ema_long_window, len(close_prices)):
            ema_long[i] = close_prices[i] * k_long + ema_long[i-1] * (1 - k_long)
            macd[i] = ema_short[i] - ema_long[i]
        
        # Signal line
        if len(close_prices) >= ema_long_window + signal_window:
            signal[ema_long_window+signal_window-1] = np.mean(macd[ema_long_window:ema_long_window+signal_window])
            
            for i in range(ema_long_window+signal_window, len(close_prices)):
                signal[i] = macd[i] * k_signal + signal[i-1] * (1 - k_signal)
        
        # Direnç seviyesi - periyoda göre ayarla
        resistance_window = 20
        if period == '1h':
            resistance_window = 20 * 6.5  # Saatlik için yaklaşık 20 günlük direnç
        
        resistance_window = int(resistance_window)
        resistance = np.zeros_like(close_prices)
        high_prices = data['High'].values
        
        for i in range(resistance_window, len(close_prices)):
            resistance[i] = np.max(high_prices[i-resistance_window:i+1])
        
        # Hacim değişimi - periyoda göre ayarla
        volume_window = 5
        if period == '1h':
            volume_window = 5 * 6.5  # Saatlik için yaklaşık 5 günlük hacim
            
        volume_window = int(volume_window)
        volumes = data['Volume'].values
        vol_ma = np.zeros_like(volumes)
        vol_change = np.ones_like(volumes)
        
        for i in range(volume_window, len(volumes)):
            vol_ma[i] = np.mean(volumes[i-volume_window:i])
            if vol_ma[i] > 0:
                vol_change[i] = volumes[i] / vol_ma[i]
        
        # Son iki veri noktasının verisini al
        if len(close_prices) < 2:
            if verbose:
                print("Yetersiz veri")
            return None
            
        last_idx = len(close_prices) - 1
        prev_idx = last_idx - 1
        
        # Momentum kriterlerini kontrol et
        # 1. RSI 50-60 bandında mı?
        rsi_50_60 = 50 <= rsi[last_idx] <= 60
        
        # 2. MACD sinyal çizgisini yukarı kesiyor mu?
        macd_crossover = (macd[prev_idx] < signal[prev_idx]) and (macd[last_idx] > signal[last_idx])
        
        # 3. Fiyat kısa dönem MA üzerinde mi?
        above_ma_short = close_prices[last_idx] > ma_short[last_idx]
        
        # 4. Direnç seviyesine yakın mı?
        close_to_resistance = close_prices[last_idx] / resistance[last_idx] if resistance[last_idx] > 0 else 0
        near_resistance = close_to_resistance > 0.95
        
        # 5. Hacim artıyor mu?
        volume_increasing = vol_change[last_idx] > 1.2
        
        # Kaç kriter karşılanıyor?
        criteria_count = sum([
            rsi_50_60,
            macd_crossover,
            above_ma_short,
            near_resistance,
            volume_increasing
        ])
        
        # Sonuçları bir sözlüğe ekle
        results = {
            'Symbol': symbol,
            'Period': period,
            'Last_Close': float(close_prices[last_idx]),
            'Last_Date': last_date,
            'RSI': float(rsi[last_idx]),
            'MACD': float(macd[last_idx]),
            'Signal': float(signal[last_idx]),
            'MA_Short': float(ma_short[last_idx]),
            'MA_Long': float(ma_long[last_idx]),
            'Resistance': float(resistance[last_idx]),
            'Close_to_Resistance': float(close_to_resistance),
            'Volume_Change': float(vol_change[last_idx]),
            'RSI_50_60': rsi_50_60,
            'MACD_Crossover': macd_crossover,
            'Above_MA_Short': above_ma_short,
            'Near_Resistance': near_resistance,
            'Volume_Increasing': volume_increasing,
            'Criteria_Count': criteria_count
        }
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"Hata: {str(e)}")
        return None

def scan_bist_for_momentum(period='1d', endeks = "bist30", lookback_days=180, verbose=False):
    """
    BIST hisselerini tarar ve momentum kriterlerini karşılayanları bulur
    
    Parametreler:
    period (str): Veri periyodu ('15m': 15 dakikalık, '1h': saatlik, '1d': günlük, vb.)
    endeks (str): BIST hisseleri için endeks ('bist30' veya 'bist100')
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    verbose (bool): Detaylı çıktı gösterilip gösterilmeyeceği
    """
    symbols = get_bist100_symbols() if endeks == "bist100" else get_bist30_symbols()

    momentum_candidates = []
    all_results = []
    latest_data_timestamp = None
    
    print(f"Toplam {len(symbols)} hisse {period} periyodunda taranacak...")
    
    for symbol in symbols:
        print(f"{symbol} taranıyor...", end="")
        
        result = analyze_stock(symbol, period=period, lookback_days=lookback_days)
        
        if result is None:
            print(" Veri yok veya işlenemedi")
            continue
            
        criteria_count = result['Criteria_Count']
        
        # Hangi kriterlerin karşılandığını göster
        criteria_met = []
        if result['RSI_50_60']:
            criteria_met.append("RSI")
        if result['MACD_Crossover']:
            criteria_met.append("MACD")
        if result['Above_MA_Short']:
            criteria_met.append("MA")
        if result['Near_Resistance']:
            criteria_met.append("Direnç")
        if result['Volume_Increasing']:
            criteria_met.append("Hacim")
        
        criteria_str = ", ".join(criteria_met) if criteria_met else "Hiçbiri"
        print(f" {criteria_count}/5 kriter karşılandı: {criteria_str}")
        
        # Tüm sonuçları kaydet
        all_results.append(result)

        # En son veri tarihini güncelle (eğer daha yeni ise)
        if 'Last_Date' in result and (latest_data_timestamp is None or result['Last_Date'] > latest_data_timestamp):
            latest_data_timestamp = result['Last_Date']
            
        if criteria_count >= 3:
            momentum_candidates.append(result)

    # Basit bir şekilde sonuçları yazdır
    print(f"\nTüm BIST Hisselerinin {period} Periyodunda Momentum Kriterleri:")
    print("-" * 80)
    
    # Sonuçları kriterlere göre sırala
    sorted_results = sorted(all_results, key=lambda x: (x['Criteria_Count'], x['RSI']), reverse=True)
    
    for res in sorted_results:
        symbol = res['Symbol']
        count = res['Criteria_Count']
        rsi = res['RSI']
        price = res['Last_Close']
        
        criteria_list = []
        if res['RSI_50_60']:
            criteria_list.append("RSI")
        if res['MACD_Crossover']:
            criteria_list.append("MACD")
        if res['Above_MA_Short']:
            criteria_list.append("MA")
        if res['Near_Resistance']:
            criteria_list.append("Direnç")
        if res['Volume_Increasing']:
            criteria_list.append("Hacim")
            
        criteria_str = ", ".join(criteria_list) if criteria_list else "Hiçbiri"
        print(f"{symbol}: {count}/5 kriter karşılandı - Fiyat: {price:.2f}, RSI: {rsi:.2f} - Kriterler: {criteria_str}")
    
    # En son veri tarihini göster (verbose modunda)
    if verbose :
        print(f"\nEn son veri tarihi: {latest_data_timestamp} (Yahoo Finance verileri genellikle 15 dakika gecikmeli)")
    
    # Momentum adaylarını döndür
    if momentum_candidates:
        # Sonuçları kriterlere göre sırala
        sorted_candidates = sorted(momentum_candidates, key=lambda x: x['Criteria_Count'], reverse=True)
        
        print(f"\nMomentum Potansiyeli Olan Hisseler ({period} periyodu, en az 3 kriter karşılayanlar):")
        print("-" * 80)
        
        for res in sorted_candidates:
            symbol = res['Symbol']
            count = res['Criteria_Count']
            rsi = res['RSI']
            price = res['Last_Close']
            
            criteria_list = []
            if res['RSI_50_60']:
                criteria_list.append("RSI")
            if res['MACD_Crossover']:
                criteria_list.append("MACD")
            if res['Above_MA_Short']:
                criteria_list.append("MA")
            if res['Near_Resistance']:
                criteria_list.append("Direnç")
            if res['Volume_Increasing']:
                criteria_list.append("Hacim")
                
            criteria_str = ", ".join(criteria_list) if criteria_list else "Hiçbiri"
            print(f"{symbol}: {count}/5 kriter karşılandı - Fiyat: {price:.2f}, RSI: {rsi:.2f} - Kriterler: {criteria_str}")
        
        return sorted_candidates
    else:
        print("\nBelirlenen kriterlere uyan hisse bulunamadı.")
        return []

def analyze_single_stock(symbol, period='1d', lookback_days=180):
    """
    Tek bir hisse için detaylı analiz yapar ve sonuçları gösterir
    
    Parametreler:
    symbol (str): Analiz edilecek hisse senedi sembolü
    period (str): Veri periyodu ('1h': saatlik, '1d': günlük, vb.)
    lookback_days (int): Kaç günlük geçmiş veri alınacağı
    """
    print(f"\n{symbol} için {period} periyodunda detaylı analiz:")
    print("-" * 40)
    
    analyze_stock(symbol, period=period, lookback_days=lookback_days, verbose=True)

def main():
    """
    Ana fonksiyon
    """
    print("Türkiye Borsası (BIST) Momentum Tarama Sistemi")
    print("=" * 45)
    print("Belirlenen momentum kriterleri:")
    print("1. RSI'nın 50-60 bandında yükselmeye başlaması")
    print("2. MACD çizgisinin sinyal çizgisini yukarı kesmesi")
    print("3. Fiyatın kısa dönem hareketli ortalamanın üzerinde olması")
    print("4. Önemli direnç seviyelerine yaklaşan veya bu seviyeleri kıran hisseler")
    print("5. Ortalama işlem hacminin son dönemde artış göstermesi")
    print("=" * 45)
    print("Desteklenen periyotlar: 15m (15 dakikalık - son 60 gün), 1h (saatlik), 1d (günlük), 5d, 1wk, 1mo")
    print("=" * 45)
    
    # Komut satırı argümanlarını işleme
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='BIST Momentum Analizi')
    parser.add_argument('command', choices=['scan', 'analyze'], help='Yapılacak işlem: scan (tüm hisseleri tara) veya analyze (tek hisse analiz et)')
    parser.add_argument('--symbol', help='Analiz edilecek hisse sembolü (sadece analyze komutu için)')
    parser.add_argument('--period', default='1d', choices=['15m', '1h', '1d', '5d', '1wk', '1mo'], help='Veri periyodu (15m: 15 dakikalık - son 60 gün, 1h: saatlik, 1d: günlük, vb.)')
    parser.add_argument('--lookback', type=int, default=180, help='Kaç günlük geçmiş veri alınacağı')
    parser.add_argument('--verbose', action='store_true', help='Detaylı çıktı göster')
    
    # Eğer argüman yoksa, yardım mesajını göster
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    if args.command == 'scan':
        # Tüm BIST hisselerini tara
        print(f"\nTüm BIST hisseleri {args.period} periyodunda taranıyor...")
        results = scan_bist_for_momentum(period=args.period, lookback_days=args.lookback, verbose=args.verbose)
        
        if results:
            # En yüksek puanlı hisseyi analiz et
            top_stock = results[0]['Symbol']
            print(f"\nEn yüksek puanlı hisse: {top_stock}")
            analyze_single_stock(top_stock, period=args.period, lookback_days=args.lookback)
            
    elif args.command == 'analyze':
        if not args.symbol:
            print("Hata: analyze komutu için --symbol parametresi gereklidir")
            sys.exit(1)
            
        # Tek bir hisseyi analiz et
        symbol = args.symbol.upper()
        if not symbol.endswith('.IS'):
            symbol += '.IS'
        analyze_single_stock(symbol, period=args.period, lookback_days=args.lookback)

if __name__ == "__main__":
    main()
