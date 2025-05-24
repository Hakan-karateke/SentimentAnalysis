## Ensemble Model Çalışma Mantığı

```
+------------------------+     +----------------------+     +----------------------------+
|                        |     |                      |     |                            |
| Giriş Metni            |---->| Metin Ön İşleme      |---->| Tokenizasyon ve Padding   |
|                        |     |                      |     |                            |
+------------------------+     +----------------------+     +----------------------------+
                                                                        |
                                                                        v
+------------------------+     +----------------------+     +----------------------------+
|                        |     |                      |     |                            |
| Duygu Tahmini          |<----| ML Sınıflandırıcı    |<----| LSTM Özellik Çıkarıcı     |
| (Pozitif/Negatif)      |     | (Random Forest vb.)  |     | (lstm_feature_extractor.h5)|
+------------------------+     +----------------------+     +----------------------------+
```

### Ensemble Model Akışı

1. **Giriş**: Kullanıcı metni alınır
2. **Ön İşleme**: Metin temizlenir, gereksiz karakterler ve stopword'ler kaldırılır
3. **Tokenizasyon**: Tokenizer ile kelimeler sayılara dönüştürülür
4. **Padding**: Tokenize edilmiş diziler sabit uzunluğa getirilir
5. **LSTM Özellik Çıkarımı**: LSTM modeli ile metinden derin özellikler çıkarılır
6. **Sınıflandırıcı Tahmini**: Bu özellikler makine öğrenmesi sınıflandırıcılarından birine (genellikle Random Forest) beslenerek duygu tahmini yapılır
7. **Sonuç**: Tahmin sonucu "Pozitif" veya "Negatif" olarak döndürülür

### Örnek İşleyiş

```python
# Örnek metin
text = "Bu ürün gerçekten harika! Çok beğendim."

# 1. Metin Ön İşleme
processed_text = preprocess_text(text)
# Sonuç: "ürün gerçekten harika beğendim"

# 2. Tokenizasyon ve Padding
sequences = tokenizer.texts_to_sequences([processed_text])
# Sonuç: [[243, 567, 89, 1245]]

padded_sequence = pad_sequences(sequences, maxlen=50)
# Sonuç: [[0, 0, ..., 0, 243, 567, 89, 1245]]

# 3. LSTM Özellik Çıkarımı
lstm_features = lstm_feature_extractor.predict(padded_sequence)
# Sonuç: [[0.23, 0.78, -0.45, ..., 0.12]] (64 veya 128 boyutlu bir vektör)

# 4. Sınıflandırıcı Tahmini
prediction = random_forest_classifier.predict_proba(lstm_features)[0][1]
# Sonuç: 0.87

# 5. Duygu ve Güven Oranı Belirleme
sentiment = "Pozitif"  # Çünkü 0.87 > 0.5
confidence = 0.87      # %87 güven oranı
```

### Neden Bu Yaklaşım?

1. **Daha İyi Performans**: LSTM'nin metin anlamını kavrama yeteneği ile klasik ML algoritmalarının sınıflandırma gücünü birleştirir
2. **Transfer Öğrenme**: LSTM özellik çıkarıcı, genel metin anlama özelliğini sınıflandırıcılara aktarır
3. **Esneklik**: Farklı sınıflandırıcılar denenerek en iyi performans elde edilebilir
4. **Yorumlanabilirlik**: Klasik ML algoritmaları daha kolay yorumlanabilir (örn. Random Forest'ta özellik önem dereceleri)

Bu yapıda LSTM modeli ağırlıklı olarak özellik çıkarımı işlemini gerçekleştirirken, nihai duygu kararını sınıflandırıcı vermektedir. Bu hibrit yaklaşım, tek başına derin öğrenme veya klasik makine öğrenmesi yöntemlerine göre genellikle daha yüksek doğruluk sağlar.
