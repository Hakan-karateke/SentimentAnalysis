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

Uygulama `sentiment_model` klasöründe bulunan LSTM tabanlı derin öğrenme modelini kullanmaktadır. Bu model metin önişleme için NLTK kütüphanesini kullanmaktadır ve tokenleştirme işlemi için önceden eğitilmiş bir tokenizer içermektedir.

## Proje Yapısı

```
SentimentAnalysis/
├── app.py                 # Flask web uygulaması
├── requirements.txt       # Gerekli Python kütüphaneleri
├── templates/             # HTML şablonları
│   ├── index.html         # Ana sayfa
│   └── twitter.html       # Twitter analizi sayfası
└── sentiment_model/       # Önceden eğitilmiş model dosyaları
    ├── config.json        # Model konfigürasyonu
    ├── lstm_model.h5      # LSTM model dosyası
    ├── text_preprocessing.py  # Metin önişleme fonksiyonları
    └── tokenizer.pkl      # Eğitilmiş tokenizer
```

## Geliştirme

Twitter analizi özelliğini aktif etmek için Twitter API anahtarları edinmeniz ve bu anahtarları kullanarak bir Twitter API entegrasyonu yapmanız gerekmektedir. Bu entegrasyonu gerçekleştirmek için `app.py` dosyasında `twitter_search` fonksiyonunu güncelleyebilirsiniz.
