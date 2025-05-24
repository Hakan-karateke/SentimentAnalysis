#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the ensemble sentiment model
"""
import os
import json
import pickle
import numpy as np
from sentiment_model.text_preprocessing import preprocess_text

# This script tests the ensemble model directly
def test_ensemble_model():
    print("Testing ensemble sentiment model...")
    print("-" * 50)
    
    # Paths
    model_dir = 'models/ensemble_sentiment_model'
    
    # Check if model directory exists
    if not os.path.exists(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        return False
        
    print(f"Found model directory: {model_dir}")
    print("Contents:", os.listdir(model_dir))
    print("-" * 50)
    
    # Try to import TensorFlow/Keras
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        load_model = tf.keras.models.load_model
        pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
    except ImportError:
        try:
            import keras
            print(f"Keras version: {keras.__version__}")
            load_model = keras.models.load_model
            pad_sequences = keras.preprocessing.sequence.pad_sequences
        except ImportError:
            print("Failed to import TensorFlow or Keras")
            return False
            
    # Load model components
    try:
        # Load LSTM feature extractor
        print("Loading LSTM feature extractor...")
        lstm_feature_extractor = load_model(f'{model_dir}/lstm_feature_extractor.h5')
        print("✓ LSTM feature extractor loaded")
        
        # Try to load full ensemble model
        print("Loading full ensemble model...")
        try:
            ensemble_model = load_model(f'{model_dir}/full_ensemble_model.h5')
            print("✓ Full ensemble model loaded")
        except Exception as e:
            print(f"Could not load full ensemble model: {str(e)}")
            ensemble_model = None
        
        # Load classifiers if full ensemble model isn't available
        classifiers = {}
        if ensemble_model is None:
            print("Loading individual classifiers...")
            try:
                import joblib
                # Load the individual classifiers
                classifiers["random_forest"] = joblib.load(f'{model_dir}/best_classifier_random_forest.pkl')
                print("✓ Random Forest classifier loaded")
                
                classifiers["gradient_boosting"] = joblib.load(f'{model_dir}/best_classifier_gradient_boosting.pkl')
                print("✓ Gradient Boosting classifier loaded")
                
                classifiers["logistic_regression"] = joblib.load(f'{model_dir}/best_classifier_lojistik_regresyon.pkl')
                print("✓ Logistic Regression classifier loaded")
            except Exception as e:
                print(f"Error loading classifier models: {str(e)}")
                return False
        
        # Load tokenizer and config
        print("Loading tokenizer and config...")
        with open(f'{model_dir}/tokenizer.pkl', 'rb') as f:
            tokenizer = pickle.load(f)
            print("✓ Tokenizer loaded")
        
        with open(f'{model_dir}/config.json', 'r') as f:
            config = json.load(f)
            print("✓ Config loaded:", config)
            
        print("-" * 50)
        print("All model components loaded successfully!")
        print("-" * 50)
        
        # Test with sample texts
        test_samples = [
            "This product is amazing! I love it.",
            "I hate this, it's terrible.",
            "Average product, could be better.",
            "The service was fast and helpful.",
            "Completely useless, don't waste your money."
        ]
        
        print("Testing model with sample texts...")
        print("-" * 50)
        
        for text in test_samples:
            print(f"Original: {text}")
            processed_text = preprocess_text(text)
            print(f"Processed: {processed_text}")
            
            # Tokenize and pad
            sequences = tokenizer.texts_to_sequences([processed_text])
            padded_sequence = pad_sequences(
                sequences, 
                maxlen=config.get('max_sequence_length', 50),
                padding='post'
            )
            
            # Extract LSTM features
            lstm_features = lstm_feature_extractor.predict(padded_sequence)
            
            # Make prediction using either the full ensemble model or the best classifier
            if ensemble_model is not None:
                # Full ensemble model is available
                prediction = ensemble_model.predict(padded_sequence)[0][0]
                model_used = "full_ensemble_model"
            else:
                # Use the best classifier according to config
                best_classifier_name = config.get('best_classifier', 'Random Forest').lower().replace(' ', '_')
                model_used = best_classifier_name
                
                if best_classifier_name == 'random_forest' and 'random_forest' in classifiers:
                    prediction = classifiers['random_forest'].predict_proba(lstm_features)[0][1]
                elif best_classifier_name == 'gradient_boosting' and 'gradient_boosting' in classifiers:
                    prediction = classifiers['gradient_boosting'].predict_proba(lstm_features)[0][1]
                elif best_classifier_name == 'logistic_regression' and 'logistic_regression' in classifiers:
                    prediction = classifiers['logistic_regression'].predict_proba(lstm_features)[0][1]
                else:
                    # If best classifier not found, use any available classifier
                    for classifier_name, classifier in classifiers.items():
                        prediction = classifier.predict_proba(lstm_features)[0][1]
                        model_used = classifier_name
                        break
            
            # Convert prediction to sentiment and confidence
            if prediction >= 0.5:
                sentiment = "Pozitif"
                confidence = float(prediction)
            else:
                sentiment = "Negatif"
                confidence = float(1.0 - prediction)
            
            print(f"Sentiment: {sentiment}, Confidence: {confidence*100:.2f}%, Model used: {model_used}")
            print("-" * 50)
            
        return True
        
    except Exception as e:
        print(f"Error testing ensemble model: {str(e)}")
        return False

if __name__ == "__main__":
    test_ensemble_model()
