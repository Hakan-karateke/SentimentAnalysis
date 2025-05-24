# Ensemble Duygu Analizi Modeli Mimarisi

Bu belge, projede kullanılan ensemble duygu analizi modelinin mimarisini ve çalışma prensibini detaylı olarak açıklamaktadır.

## Ensemble Model Yapısı

Ensemble model, iki ana bileşenden oluşmaktadır:

1. **LSTM Özellik Çıkarıcı**: Metin verilerinden derin özellikler çıkaran bir LSTM tabanlı sinir ağı
2. **Geleneksel Makine Öğrenimi Sınıflandırıcıları**: LSTM'den çıkarılan özellikleri kullanarak nihai duygu tahmini yapan sınıflandırıcılar

### 1. LSTM Özellik Çıkarıcı (`lstm_feature_extractor.h5`)

- LSTM özellik çıkarıcı, metinden zengin özellikler çıkaran bir derin öğrenme modelidir
- Embedding katmanı, LSTM katmanları ve yoğun katmanlardan oluşur
- **Önemli**: Son katman **çıktı katmanı değil**, özellik çıkarıcı olarak kullanılan bir ara katmandır
- `lstm_model.h5` dosyası, LSTM tabanlı duygu analizi için eğitilmiş tam bir modeldir
- `lstm_feature_extractor.h5` ise, bu modelin son sınıflandırma katmanı çıkarılmış halidir ve özellik çıkarıcı olarak kullanılır

### 2. Makine Öğrenimi Sınıflandırıcıları

Projede üç farklı sınıflandırıcı kullanılmıştır:

- **Random Forest** (`best_classifier_random_forest.pkl`)
- **Gradient Boosting** (`best_classifier_gradient_boosting.pkl`)
- **Lojistik Regresyon** (`best_classifier_lojistik_regresyon.pkl`)

Bu sınıflandırıcılar, LSTM özellik çıkarıcısından gelen öznitelikleri kullanarak nihai duygu tahmini yapar.

## Tahmin Süreci Akışı

1. **Metin Ön İşleme**:
   - Temizleme (kullanıcı adları, URL'ler, özel karakterlerin kaldırılması)
   - Küçük harfe dönüştürme
   - Gereksiz kelimelerin (stopwords) kaldırılması

2. **Tokenizasyon ve Padding**:
   - Metindeki kelimeler tamsayı dizilerine dönüştürülür
   - Sabit uzunluğa getirilir (padding)

3. **LSTM Özellik Çıkarımı**:
   - Padding uygulanmış tokenize edilmiş metin LSTM modeline beslenir
   - LSTM modelinin son katmanından çıkan özellikler alınır
   - Bu adımda metin, yüksek boyutlu bir özellik vektörüne dönüştürülür

4. **Sınıflandırıcı Tahmini**:
   - LSTM'den çıkan özellikler, seçilen sınıflandırıcıya (Random Forest, Gradient Boosting veya Lojistik Regresyon) beslenir
   - Sınıflandırıcı bu özellikleri kullanarak bir tahmin skoru üretir

5. **Sonuç Dönüşümü**:
   - Tahmin skoru 0.5'ten büyükse "Pozitif", değilse "Negatif" olarak değerlendirilir
   - Skor değeri güven oranı olarak yorumlanır

## Kod İçerisinde Akış

```python
# 1. Tokenizasyon ve padding
sequences = tokenizer.texts_to_sequences([processed_text])
padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# 2. LSTM özellikleri çıkar
lstm_features = lstm_feature_extractor.predict(padded_sequence)

# 3. Özellikleri sınıflandırıcıya besle
prediction = classifiers[best_classifier_name].predict_proba(lstm_features)[0][1]

# 4. Sonucu yorumla
sentiment = "Pozitif" if prediction >= 0.5 else "Negatif"
confidence = prediction if prediction >= 0.5 else (1.0 - prediction)
```

## Ensemble Yaklaşımının Avantajları

1. **Daha Yüksek Doğruluk**: LSTM ve geleneksel makine öğrenimi yöntemlerinin güçlü yönlerini birleştirir
2. **Daha İyi Genelleme**: Farklı algoritmalar bir araya getirilerek aşırı öğrenme (overfitting) riski azaltılır
3. **Daha Hızlı Eğitim**: Özellik çıkarıcı bir kez eğitildikten sonra, farklı sınıflandırıcılar daha hızlı eğitilebilir
4. **Esneklik**: Farklı sınıflandırıcılar kullanarak modelin davranışı ayarlanabilir

## Kullanılan Dosyalar

- `lstm_feature_extractor.h5`: LSTM özellik çıkarıcı model
- `best_classifier_random_forest.pkl`: Random Forest sınıflandırıcı
- `best_classifier_gradient_boosting.pkl`: Gradient Boosting sınıflandırıcı
- `best_classifier_lojistik_regresyon.pkl`: Lojistik Regresyon sınıflandırıcı
- `tokenizer.pkl`: Kelime tokenizer'ı
- `config.json`: Model konfigürasyon dosyası

## Özetleme

Ensemble duygu analizi modelimiz, LSTM modelinin son katmanını özellik çıkarıcı olarak kullanıp, bu özellikleri geleneksel makine öğrenimi sınıflandırıcılarına girdi olarak veren hibrit bir yaklaşımla çalışmaktadır. Bu hibrit yaklaşım, sadece derin öğrenme veya sadece geleneksel makine öğrenimi yaklaşımlarına göre daha iyi performans göstermektedir.
