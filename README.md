# BIST Momentum Analiz Aracı

Bu araç, Borsa İstanbul (BIST) hisse senetleri için momentum analizi yapmanızı sağlar. Farklı zaman dilimlerinde (15 dakika, 1 saat, günlük, haftalık, aylık) analiz yapabilir ve potansiyel momentum fırsatlarını belirlemenize yardımcı olur.

## Özellikler

- 15 dakikalık, saatlik, günlük, haftalık ve aylık periyotlarda analiz
- Tüm BIST hisselerini tarama veya tek bir hisseyi detaylı analiz etme
- Momentum göstergeleri:
  1. RSI'nın 50-60 bandında yükselmeye başlaması
  2. MACD çizgisinin sinyal çizgisini yukarı kesmesi
  3. Fiyatın kısa dönem hareketli ortalamanın üzerinde olması
  4. Önemli direnç seviyelerine yaklaşan veya bu seviyeleri kıran hisseler
  5. Ortalama işlem hacminin son dönemde artış göstermesi

## Gereksinimler

```
pandas
numpy
yfinance
```

## Kurulum

```bash
pip install pandas numpy yfinance
```

## Kullanım

### Tüm BIST hisselerini taramak için:

```bash
python bist_hourly_momentum_analyzer.py scan --period 15m
```

### Tek bir hisse senedini analiz etmek için:

```bash
python bist_hourly_momentum_analyzer.py analyze --symbol GARAN --period 1h
```

### Ek parametreler:

- `--lookback`: Kaç günlük geçmiş veri alınacağını belirler (varsayılan: 180 gün)
- `--verbose`: Detaylı çıktı gösterir, en son veri tarihini de içerir

Örnek:
```bash
python bist_hourly_momentum_analyzer.py scan --period 1d --lookback 90 --verbose
```

### Desteklenen Periyotlar:

- `15m`: 15 dakikalık (son 60 gün ile sınırlı)
- `1h`: Saatlik
- `1d`: Günlük
- `5d`: 5 günlük
- `1wk`: Haftalık
- `1mo`: Aylık

## Notlar

- 15 dakikalık veri için Yahoo Finance sadece son 60 gün veri sağlamaktadır.
- Yahoo Finance verileri genellikle 15-20 dakika gecikmeli olarak sunulur.
- Analiz sonuçları yatırım tavsiyesi değildir, sadece teknik göstergelere dayalı bir tarama aracıdır.

## Lisans

MIT
