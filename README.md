# Müzik Türü Sınıflandırma

Bu proje, şarkıların demografik ve özelliksel bilgilerine (dans edilebilirlik, enerji, akustiklik, tempo vb.) dayanarak müzik türlerini sınıflandırmayı amaçlar. Doğru sınıflandırma, müzik öneri sistemlerinin geliştirilmesi, müzik analitiği ve kullanıcı tercihlerine dayalı öneriler sunma gibi geniş bir kullanım alanına sahiptir.

---

## Veri Seti Hakkında
Bu projede kullanılan veri seti, [Kaggle](https://www.kaggle.com/datasets/purumalgi/music-genre-classification?select=train.csv) platformundan alınmıştır. Veri seti, 17.996 şarkı örneği ve toplam 17 sütundan oluşmaktadır.

### Sütunlar
- **Artist Name**: Şarkıyı yapan sanatçının adı
- **Track Name**: Şarkının adı
- **Popularity**: Şarkının popülaritesi
- **Danceability**: Şarkının dans edilebilirliği
- **Energy**: Şarkının enerjisi
- **Key**: Şarkının anahtarı
- **Loudness**: Şarkının yüksekliği
- **Mode**: Şarkının modu
- **Speechiness**: Şarkının konuşma oranı
- **Acousticness**: Şarkının akustikliği
- **Instrumentalness**: Şarkının enstrümantal olup olmadığı
- **Liveness**: Şarkının canlılık oranı
- **Valence**: Şarkının duygu durumu
- **Tempo**: Şarkının temposu
- **Duration in milliseconds**: Şarkının süresi (milisaniye)
- **Time Signature**: Şarkının zaman imzası (ritmik yapı)
- **Class**: Şarkının ait olduğu müzik türü (hedef değişken)

---

## Veri Hazırlama Süreci
1. Eksik veriler kontrol edilmiştir:
   - **Popularity** sütunu medyan ile doldurulmuştur.
   - **Key** sütunu en sık görülen değerle (mod) doldurulmuştur.
   - **Instrumentalness** sütunu sıfır ile doldurulmuştur.
2. **Artist Name** ve **Track Name** sütunları kaldırılarak, veri seti 17 sütundan 15 sütuna düşürülmüştür.
3. **Class** sütunu hedef değişken olarak belirlenmiştir.
4. Sayısal sütunlar MinMaxScaler ile 0-1 aralığında normalleştirilmiştir.
5. Aykırı değerler tespit edilerek baskılama yöntemi ile giderilmiştir.
6. Hedef değişken, one-hot encoding yöntemiyle sayısallaştırılmıştır.

---

## Modeller ve Sonuçlar

### Model 1: Eğitim Setini Aynı Zamanda Test Verisi Olarak Kullanma
- **Eğitim Parametreleri**:
  - 100 epoch, batch size = 32, öğrenme katsayısı = 0.001
  - Optimizasyon: Adam
  - Aktivasyon Fonksiyonları: Gizli katmanlar için ReLU, çıkış katmanı için Softmax
  - Katmanlar: 512, 256, 128, 64, 32, 16 nöron (dropout=0.2)

- **Sonuç**: Eğitim seti üzerinde yüksek doğruluk, ancak overfitting riski taşır.

---

### Model 2: %66-%34 Eğitim-Test Ayırma (5 Farklı Rassal Ayırma)
- **Eğitim Parametreleri**:
  - 10 epoch, batch size = 16, öğrenme katsayısı = 0.001
  - Katmanlar: 512, 256, 128, 64, 32, 16 nöron
- **Sonuçlar**:
  - Ortalama Doğruluk: %XX
  - Ortalama F1 Skoru: X.XX

---

### Model 3: 5-Fold Cross Validation
- **Eğitim Parametreleri**:
  - 30 epoch, batch size = 32, öğrenme katsayısı = 0.001
  - Katmanlar: 256, 128, 64, 32, 16 nöron
- **Sonuçlar**:
  - Ortalama Doğruluk: %XX
  - Ortalama F1 Skoru: X.XX

---

### Model 4: 10-Fold Cross Validation
- **Eğitim Parametreleri**:
  - 30 epoch, batch size = 32, öğrenme katsayısı = 0.001
  - Katmanlar: 256, 128, 64, 32, 16 nöron
- **Sonuçlar**:
  - Ortalama Doğruluk: %XX
  - Ortalama F1 Skoru: X.XX

---

## Nasıl Çalıştırılır?
1. Repository’yi klonlayın:
   ```bash
   git clone https://github.com/kullaniciadi/muzik-turu-siniflandirma.git

2. Gerekli kütüphaneleri yükleyin:
    ```bash
   pip install -r requirements.txt
3. Veri setini indirin:
    ```bash
   https://www.kaggle.com/datasets/purumalgi/music-genre-classification?select=train.csv
4. İndirdikten sonra 'train.csv' dosyasını bu proje dizinine yerleştirin.
5. Aşağıdaki komutu çalıştırın:
   ```bash
   python koddosyasi.py

## Veri Seti Hakkında
Bu projede kullanılan veri seti, Kaggle'da yer almaktadır. Veri seti, şarkıların fiziksel ve akustik özelliklerini içerir ve toplamda 17.996 örnek bulunmaktadır.
