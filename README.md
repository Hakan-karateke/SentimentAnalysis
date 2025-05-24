# Duygu Analizi Uygulaması

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.9+-orange.svg)
![NLTK](https://img.shields.io/badge/nltk-v3.6+-yellow.svg)

Bu uygulama, önceden eğitilmiş kendim oluşturduğum bir LSTM modeli kullanarak metin veya tweet'leri analiz eden ve duygu durumlarını (pozitif/negatif) belirleyen bir web uygulamasıdır.

## Özellikler

- Tek metin duygu analizi
- Toplu metin analizi (birden fazla metin)
- Web arayüzü ile kolay kullanım
- Güven oranı göstergesi
- Kullanıma hazır pre-trained model

## Kurulum

1. Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

2. Uygulamayı başlatın:

```bash
python app.py
```

3. Tarayıcınızda aşağıdaki adresi açın:

```
http://127.0.0.1:5000
```

## Kullanım

### Tek Metin Analizi

1. "Metin Analizi" sekmesini seçin
2. Analiz etmek istediğiniz metni girin
3. "Analiz Et" butonuna tıklayın
4. Sonucu görüntüleyin

### Toplu Analiz

1. "Toplu Analiz" sekmesini seçin
2. Her satıra bir metin olacak şekilde metinleri girin
3. "Toplu Analiz Et" butonuna tıklayın
4. Sonuçları görüntüleyin

### Twitter Analizi

Bu özelliği kullanabilmek için Twitter API entegrasyonu yapmanız gerekmektedir.

## Model Hakkında

Uygulama `models/ensemble_sentiment_model` klasöründe bulunan ensemble model mimarisini kullanmaktadır. Bu model, LSTM tabanlı öznitelik çıkarıcı ve geleneksel makine öğrenimi sınıflandırıcılarının kombinasyonu şeklindedir (Random Forest, Gradient Boosting ve Lojistik Regresyon). Bu ensemble yaklaşım, tek başına LSTM kullanımına göre daha yüksek doğruluk ve güvenilirlik sağlamaktadır.

Metin önişleme için NLTK kütüphanesi kullanılmaktadır ve tokenleştirme işlemi için önceden eğitilmiş bir tokenizer içermektedir.

## Proje Yapısı

```
SentimentAnalysis/
├── app.py                 # Flask web uygulaması
├── ensemble_model_test.ipynb # Ensemble model test notebook
├── test_ensemble.py       # Ensemble model test script
├── update_preprocessing.py # Script to update preprocessing files
├── ensemble_model_update.md # Update documentation
├── requirements.txt       # Gerekli Python kütüphaneleri
├── templates/             # HTML şablonları
│   ├── index.html         # Ana sayfa
│   └── twitter.html       # Twitter analizi sayfası
├── models/                # Model dosyaları
│   ├── ensemble_sentiment_model/ # Ensemble model dosyaları
│   │   ├── best_classifier_gradient_boosting.pkl # Gradient Boosting sınıflandırıcı
│   │   ├── best_classifier_lojistik_regresyon.pkl # Lojistik Regresyon sınıflandırıcı
│   │   ├── best_classifier_random_forest.pkl # Random Forest sınıflandırıcı 
│   │   ├── config.json    # Model konfigürasyonu
│   │   ├── full_ensemble_model.h5 # Tam ensemble model
│   │   ├── lstm_feature_extractor.h5 # LSTM öznitelik çıkarıcı
│   │   ├── text_preprocessing.py # Metin önişleme fonksiyonları
│   │   └── tokenizer.pkl  # Eğitilmiş tokenizer
│   └── sentiment_model/   # Eski LSTM model (Yedek)
└── sentiment_model/       # Metin önişleme modülü
    ├── text_preprocessing.py  # Metin önişleme fonksiyonları
    └── __pycache__/       # Python önbellek dosyaları
```

## Geliştirme

Twitter analizi özelliğini aktif etmek için Twitter API anahtarları edinmeniz ve bu anahtarları kullanarak bir Twitter API entegrasyonu yapmanız gerekmektedir. Bu entegrasyonu gerçekleştirmek için `app.py` dosyasında `twitter_search` fonksiyonunu güncelleyebilirsiniz.

## Ensemble Model Mimarisi

Bu projede kullanılan ensemble model, aşağıdaki bileşenlerden oluşmaktadır:

1. **LSTM Öznitelik Çıkarıcı**: Metinlerden derin öznitelikler çıkarmak için kullanılan bir LSTM ağı.
2. **Geleneksel Makine Öğrenimi Sınıflandırıcıları**:
   - Random Forest
   - Gradient Boosting
   - Lojistik Regresyon

Ensemble model, iki aşamalı bir süreçle çalışır:
1. Giriş metni önişleme adımlarından geçirilerek temizlenir.
2. Temizlenmiş metin LSTM modelinden geçirilerek derin öznitelikler çıkarılır.
3. Bu öznitelikler, en iyi performans gösteren sınıflandırıcıya (genellikle Random Forest) verilerek nihai duygu tahmini yapılır.

Bu hibrit yaklaşım, tek başına derin öğrenme modellerine göre daha yüksek doğruluk ve daha iyi genelleme yeteneği sağlamaktadır.

### Model Performansı

Ensemble model, standart LSTM modeline göre özellikle:
- Daha kısa metinlerde
- Az görülen ifade kalıplarında
- Belirsiz duygusal bağlamlarda

daha iyi performans göstermektedir.
