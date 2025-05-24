#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ensemble Duygu Analizi Modeli İçin Kapsamlı Test Betiği
"""
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sentiment_model.text_preprocessing import preprocess_text

def test_ensemble_model_detail():
    print("Ensemble Duygu Analizi Modelini Detaylı Test Ediliyor...")
    print("-" * 70)
    
    # Dosya yolları
    model_dir = 'models/ensemble_sentiment_model'
    
    # Klasörün var olduğunu kontrol et
    if not os.path.exists(model_dir):
        print(f"Hata: Model klasörü bulunamadı: {model_dir}")
        return False
        
    print(f"Model klasörü bulundu: {model_dir}")
    print("İçerik:", os.listdir(model_dir))
    print("-" * 70)
    
    # TensorFlow/Keras import etmeyi dene
    try:
        import tensorflow as tf
        print(f"TensorFlow sürümü: {tf.__version__}")
        load_model = tf.keras.models.load_model
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    except ImportError:
        try:
            import keras
            print(f"Keras sürümü: {keras.__version__}")
            load_model = keras.models.load_model
            pad_sequences = keras.preprocessing.sequence.pad_sequences
        except ImportError:
            print("TensorFlow veya Keras import edilemedi")
            return False
            
    # Model bileşenlerini yükleme
    try:
        # LSTM özellik çıkarıcıyı yükle
        print("LSTM özellik çıkarıcı yükleniyor...")
        lstm_feature_extractor = load_model(f'{model_dir}/lstm_feature_extractor.h5')
        print("✓ LSTM özellik çıkarıcı yüklendi")
        
        # LSTM model yapısını incele
        print("\nLSTM Model Özeti:")
        lstm_feature_extractor.summary()
        
        # Ensemble modeli yüklemeyi dene
        print("\nTam ensemble modeli yükleniyor...")
        try:
            ensemble_model = load_model(f'{model_dir}/full_ensemble_model.h5')
            print("✓ Tam ensemble modeli yüklendi")
            print("\nEnsemble Model Özeti:")
            ensemble_model.summary()
        except Exception as e:
            print(f"Tam ensemble model yüklenemedi: {str(e)}")
            ensemble_model = None
        
        # Ensemble model yoksa sınıflandırıcıları yükle
        classifiers = {}
        if ensemble_model is None:
            print("\nSınıflandırıcılar yükleniyor...")
            try:
                import joblib
                
                # Random Forest sınıflandırıcı
                rf_path = f'{model_dir}/best_classifier_random_forest.pkl'
                if os.path.exists(rf_path):
                    classifiers["random_forest"] = joblib.load(rf_path)
                    print("✓ Random Forest sınıflandırıcı yüklendi")
                
                # Gradient Boosting sınıflandırıcı
                gb_path = f'{model_dir}/best_classifier_gradient_boosting.pkl'
                if os.path.exists(gb_path):
                    classifiers["gradient_boosting"] = joblib.load(gb_path)
                    print("✓ Gradient Boosting sınıflandırıcı yüklendi")
                
                # Lojistik Regresyon sınıflandırıcı
                lr_path = f'{model_dir}/best_classifier_lojistik_regresyon.pkl'
                if os.path.exists(lr_path):
                    classifiers["logistic_regression"] = joblib.load(lr_path)
                    print("✓ Lojistik Regresyon sınıflandırıcı yüklendi")
                
                # Sınıflandırıcı özellikleri
                for name, clf in classifiers.items():
                    print(f"\n{name.capitalize()} Sınıflandırıcı Bilgileri:")
                    print(f"Tip: {type(clf).__name__}")
                    if hasattr(clf, 'n_estimators'):
                        print(f"Ağaç sayısı: {clf.n_estimators}")
                    if hasattr(clf, 'max_depth'):
                        print(f"Maksimum derinlik: {clf.max_depth}")
                    if hasattr(clf, 'feature_importances_'):
                        print(f"Özellik önemleri: {clf.feature_importances_[:5]}...")
                    
            except Exception as e:
                print(f"Sınıflandırıcı modelleri yüklenirken hata: {str(e)}")
                if len(classifiers) == 0:
                    return False
        
        # Tokenizer ve konfigürasyon yükleme
        print("\nTokenizer ve konfigürasyon yükleniyor...")
        with open(f'{model_dir}/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            print("✓ Tokenizer yüklendi")
            vocab_size = len(tokenizer.word_index) + 1
            print(f"Kelime dağarcığı boyutu: {vocab_size}")
        
        with open(f'{model_dir}/config.json', 'r') as f:
            config = json.load(f)
            print("✓ Konfigürasyon yüklendi:", config)
            
        print("-" * 70)
        print("Tüm model bileşenleri başarıyla yüklendi!")
        print("-" * 70)
        
        # Test metinleri
        test_samples = [
            "This product is amazing! I love it.",
            "I hate this, it's terrible.",
            "Average product, could be better.",
            "The service was fast and helpful.",
            "Completely useless, don't waste your money.",
            "Bu ürün gerçekten harika! Çok beğendim.", # Türkçe örnek
            "Bu kitabı tamamen zaman kaybı buldum.",   # Türkçe örnek
        ]
        
        print("\nLSTM ve Sınıflandırıcı Modelleri Test Ediliyor...")
        print("-" * 70)
        
        results = []
        
        for text in test_samples:
            print(f"Orijinal metin: {text}")
            processed_text = preprocess_text(text)
            print(f"İşlenmiş metin: {processed_text}")
            
            # Tokenize et ve padding uygula
            sequences = tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(
                sequences, 
                maxlen=config.get('max_sequence_length', 50),
                padding='post'
            )
            
            # LSTM özellikleri çıkar
            lstm_features = lstm_feature_extractor.predict(padded_sequence)
            print(f"LSTM özellikleri şekli: {lstm_features.shape}")
            
            # Tüm sınıflandırıcıları dene ve sonuçları karşılaştır
            classifier_results = {}
            
            # Ensemble model varsa onu kullan
            if ensemble_model is not None:
                prediction = ensemble_model.predict(padded_sequence)[0][0]
                classifier_results["ensemble_full"] = prediction
            
            # Her bir sınıflandırıcıyı dene
            for name, clf in classifiers.items():
                try:
                    prediction = clf.predict_proba(lstm_features)[0][1]
                    classifier_results[name] = prediction
                except Exception as e:
                    print(f"{name} tahmini başarısız oldu: {str(e)}")
            
            # En iyi sınıflandırıcıyı ve sonucu belirle
            best_clf = config.get('best_classifier', 'Random Forest').lower().replace(' ', '_')
            if best_clf in classifier_results:
                best_prediction = classifier_results[best_clf]
            else:
                # En iyi sınıflandırıcı yoksa, ilk mevcut olanı kullan
                best_prediction = next(iter(classifier_results.values()))
            
            # Sonucu yorumla
            if best_prediction >= 0.5:
                sentiment = "Pozitif"
                confidence = float(best_prediction)
            else:
                sentiment = "Negatif"
                confidence = float(1.0 - best_prediction)
            
            # Tüm sınıflandırıcı sonuçlarını göster
            print("\nSınıflandırıcı sonuçları:")
            for clf_name, pred in classifier_results.items():
                sentiment_text = "Pozitif" if pred >= 0.5 else "Negatif"
                conf = pred if pred >= 0.5 else (1.0 - pred)
                print(f"- {clf_name}: {sentiment_text} (%{conf*100:.2f})")
            
            print(f"\nNihai Tahmin: {sentiment} (%{confidence*100:.2f})")
            print("-" * 70)
            
            # Sonuçları kaydet
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence,
                'classifier_results': classifier_results
            })
            
        # Sonuçları görselleştir
        try:
            visualize_results(results)
        except Exception as e:
            print(f"Görselleştirme başarısız oldu: {str(e)}")
            
        return results
        
    except Exception as e:
        print(f"Ensemble model test edilirken hata: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def visualize_results(results):
    """Ensemble model sonuçlarını görselleştir"""
    if not results:
        return
        
    # Matplotlib kullanılabilir mi kontrol et
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib kütüphanesi bulunamadı, görselleştirme atlanıyor")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Her sınıflandırıcı için renkler ve etiketler
    colors = {
        'random_forest': 'blue',
        'gradient_boosting': 'green',
        'logistic_regression': 'red',
        'ensemble_full': 'purple'
    }
    
    # Her örnek için tahminleri çiz
    labels = []
    for i, result in enumerate(results):
        clf_results = result['classifier_results']
        x = range(len(clf_results))
        
        # Sınıflandırıcı adlarını ayıkla
        clf_names = list(clf_results.keys())
        
        # Y değerlerini hesapla - pozitif için doğrudan değer, negatif için 1-değer kullan
        y_values = []
        for clf_name in clf_names:
            pred = clf_results[clf_name]
            if pred < 0.5:  # Negatif duygu
                y_values.append(-1 * (1 - pred))  # Negatif değer olarak göster
            else:  # Pozitif duygu
                y_values.append(pred)
        
        plt.subplot(len(results), 1, i+1)
        bars = plt.bar(clf_names, y_values, color=[colors.get(name, 'gray') for name in clf_names])
        
        # Çubukların üzerine değerleri yaz
        for bar, value in zip(bars, y_values):
            if value < 0:
                plt.text(bar.get_x() + bar.get_width()/2., -0.05,
                        f'{abs(value):.2f}', 
                        ha='center', va='top', color='red', rotation=0)
            else:
                plt.text(bar.get_x() + bar.get_width()/2., 0.05,
                        f'{value:.2f}', 
                        ha='center', va='bottom', color='green', rotation=0)
        
        # Eksenleri ayarla
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.ylim(-1, 1)
        plt.title(f"Örnek {i+1}: {result['text'][:30]}..." if len(result['text']) > 30 else result['text'])
        plt.ylabel("Duygu Skoru")
        
        # İlk örnekte açıklamalar göster
        if i == 0:
            labels = clf_names
    
    plt.tight_layout()
    
    # Sonucu kaydet ve göster
    plt.savefig('ensemble_model_test_results.png')
    print("\nSonuç görselleştirmesi 'ensemble_model_test_results.png' dosyasına kaydedildi")

if __name__ == "__main__":
    test_ensemble_model_detail()
