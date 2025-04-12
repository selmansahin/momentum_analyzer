# BIST Teknik Analiz Araçları

Bu repo, Borsa İstanbul (BIST) hisse senetleri için çeşitli teknik analiz araçları içerir. Farklı zaman dilimlerinde (15 dakika, 1 saat, günlük, haftalık, aylık) analiz yapabilir ve potansiyel alım/satım fırsatlarını belirlemenize yardımcı olur.

## İçerikler

Bu repo aşağıdaki analiz araçlarını içerir:

1. **BIST Momentum Analiz Aracı** (`bist_momentum_analyzer.py`): Momentum tabanlı teknik analiz
2. **BIST Divergence Analiz Aracı** (`bist_divergence_analyzer.py`): Pozitif uyumsuzluk (divergence) tabanlı teknik analiz
3. **BIST Sembol Listesi** (`bist_symbols.py`): BIST30 ve BIST100 hisse sembollerini içeren yardımcı modül

## Gereksinimler

```
pandas
numpy
yfinance
matplotlib
scipy
```

## Kurulum

```bash
pip install pandas numpy yfinance matplotlib scipy
```

## 1. BIST Momentum Analiz Aracı

Bu araç, BIST hisselerini momentum kriterlerine göre tarar ve potansiyel alım fırsatlarını belirler.

### Özellikler

- 15 dakikalık, saatlik, günlük, haftalık ve aylık periyotlarda analiz
- Tüm BIST30 veya BIST100 hisselerini tarama veya tek bir hisseyi detaylı analiz etme
- Momentum göstergeleri:
  1. RSI'nın 50-60 bandında yükselmeye başlaması
  2. MACD çizgisinin sinyal çizgisini yukarı kesmesi
  3. Fiyatın kısa dönem hareketli ortalamanın üzerinde olması
  4. Önemli direnç seviyelerine yaklaşan veya bu seviyeleri kıran hisseler
  5. Ortalama işlem hacminin son dönemde artış göstermesi

### Kullanım

#### Tüm BIST hisselerini momentum için taramak için:

```bash
python bist_momentum_analyzer.py scan --period 1d
```

#### Tek bir hisse senedini momentum için analiz etmek için:

```bash
python bist_momentum_analyzer.py analyze --symbol GARAN --period 1d
```

#### Ek parametreler:

- `--lookback`: Kaç günlük geçmiş veri alınacağını belirler (varsayılan: 180 gün)
- `--verbose`: Detaylı çıktı gösterir, en son veri tarihini de içerir

## 2. BIST Divergence Analiz Aracı

Bu araç, BIST hisselerini pozitif uyumsuzluk (divergence) açısından tarar ve potansiyel alım fırsatlarını belirler.

### Özellikler

- 15 dakikalık, saatlik, günlük, haftalık ve aylık periyotlarda analiz
- Tüm BIST30 veya BIST100 hisselerini tarama veya tek bir hisseyi detaylı analiz etme
- Pozitif uyumsuzluk göstergeleri:
  1. Fiyat daha düşük dip yaparken, RSI'nın daha yüksek dip yapması
  2. Fiyat daha düşük dip yaparken, MACD'nin daha yüksek dip yapması
  3. Uyumsuzluk gücünün hesaplanması ve sıralanması
  4. Son belirli gün sayısı içindeki uyumsuzlukların filtrelenmesi
  5. Çoklu teknik gösterge panelleriyle detaylı görselleştirme

### Kullanım

#### Tüm BIST hisselerini pozitif uyumsuzluk için taramak için:

```bash
python bist_divergence_analyzer.py scan --endeks bist30 --period 1d
```

#### Tek bir hisse senedini pozitif uyumsuzluk için analiz etmek için:

```bash
python bist_divergence_analyzer.py analyze --symbol GARAN --period 1d
```

#### Ek parametreler:

- `--lookback`: Kaç günlük geçmiş veri alınacağını belirler (varsayılan: 180 gün)
- `--min-window`: Minimum noktaları bulmak için kullanılacak pencere boyutu (varsayılan: 5)
- `--div-window`: Uyumsuzluk tespitinde kullanılacak pencere boyutu (varsayılan: 20)
- `--recent-days`: Son kaç gün içindeki uyumsuzlukları gösterir (varsayılan: tüm uyumsuzluklar)
- `--endeks`: Taranacak endeks (bist30 veya bist100) (varsayılan: bist30)
- `--verbose`: Detaylı çıktı gösterir, en son veri tarihini de içerir
- `--save-plot`: Grafikleri kaydeder
- `--no-plot`: Grafikleri göstermez

## Desteklenen Periyotlar

Tüm araçlar için aşağıdaki periyotlar desteklenmektedir:

- `15m`: 15 dakikalık (son 60 gün ile sınırlı)
- `1h`: Saatlik
- `1d`: Günlük
- `5d`: 5 günlük
- `1wk`: Haftalık
- `1mo`: Aylık

## Teknik Detaylar

### Momentum Analizi

- RSI (Göreceli Güç Endeksi): Fiyat hareketlerinin hızını ve değişimini ölçer
- MACD (Hareketli Ortalama Yakınsama/Iraksama): Kısa ve uzun dönem hareketli ortalamalar arasındaki ilişkiyi gösterir
- Hareketli Ortalamalar: Fiyat trendini belirlemek için kullanılır
- Direnç Seviyeleri: Fiyatın geçmesi zor olan üst seviyeleri gösterir
- Hacim Analizi: İşlem hacmindeki değişimleri takip eder

### Divergence (Uyumsuzluk) Analizi

- Pozitif uyumsuzluk, fiyat daha düşük seviyede dip yaparken, göstergenin (RSI, MACD) daha yüksek dip yapması durumudur.
- Bu durum genellikle bir alım fırsatı olarak değerlendirilir.
- Uyumsuzluk gücü, fiyat ve gösterge değişimleri arasındaki farkın mutlak değeri olarak hesaplanır.
- Görselleştirme, fiyat grafiği, RSI ve MACD göstergelerini tek bir figür üzerinde üç ayrı panel olarak gösterir.
- Uyumsuzluk noktaları grafikler üzerinde işaretlenir.

## Örnek Kullanım Senaryoları

### Senaryo 1: Son 15 gün içinde pozitif uyumsuzluk gösteren BIST100 hisselerini tarama

```bash
python bist_divergence_analyzer.py scan --endeks bist100 --period 1d --recent-days 15 --verbose
```

### Senaryo 2: Belirli bir hissenin günlük momentum analizini yapma

```bash
python bist_momentum_analyzer.py analyze --symbol THYAO --period 1d --lookback 90
```

### Senaryo 3: Belirli bir hissenin saatlik pozitif uyumsuzluk analizini yapma ve grafikleri kaydetme

```bash
python bist_divergence_analyzer.py analyze --symbol GARAN --period 1h --lookback 30 --save-plot
```

## Notlar

- 15 dakikalık veri için Yahoo Finance sadece son 60 gün veri sağlamaktadır.
- Yahoo Finance verileri genellikle 15-20 dakika gecikmeli olarak sunulur.
- Analiz sonuçları yatırım tavsiyesi değildir, sadece teknik göstergelere dayalı bir tarama aracıdır.
- Gerçek yatırım kararları için profesyonel danışmanlık almanız önerilir.

## Lisans

MIT
