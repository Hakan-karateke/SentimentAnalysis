import os
import json
import pickle
import numpy as np
import sklearn
from flask import Flask, render_template, request, jsonify
from sentiment_model.text_preprocessing import preprocess_text

# Define variables that would normally be set by loading the model
lstm_feature_extractor = None
ensemble_model = None
tokenizer = None
config = None
classifiers = {}

app = Flask(__name__)

def load_ensemble_model():
    """Load the pre-trained Ensemble model with LSTM feature extractor and classifiers"""
    global lstm_feature_extractor, ensemble_model, tokenizer, config, classifiers
    
    try:
        # Try to import TensorFlow/Keras with dynamic imports to avoid import errors
        load_model = None
        pad_sequences = None
        
        # Try different import strategies
        try:
            # Try tensorflow first
            import tensorflow as tf
            load_model = tf.keras.models.load_model
            pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
            print("TensorFlow/Keras imports successful")
        except (ImportError, AttributeError):
            # Then try standalone keras
            try:
                import keras
                load_model = keras.models.load_model
                pad_sequences = keras.preprocessing.sequence.pad_sequences
                print("Keras imports successful")
            except (ImportError, AttributeError):
                print("Failed to import Keras or TensorFlow. Will use fallback method.")
                return False
        
        # Make necessary functions globally accessible
        globals()['load_model'] = load_model
        globals()['pad_sequences'] = pad_sequences
        
        # Try to load models
        try:
            # Load LSTM feature extractor
            lstm_feature_extractor = load_model('models/ensemble_sentiment_model/lstm_feature_extractor.h5')
            
            # Load full ensemble model if available
            try:
                ensemble_model = load_model('models/ensemble_sentiment_model/full_ensemble_model.h5')
                print("Full ensemble model successfully loaded.")
            except Exception as e:
                print(f"Could not load full ensemble model: {str(e)}")
                ensemble_model = None
            
            # Load classifiers if full ensemble model isn't available
            if ensemble_model is None:
                try:
                    import joblib
                    # Load the individual classifiers
                    classifiers["random_forest"] = joblib.load('models/ensemble_sentiment_model/best_classifier_random_forest.pkl')
                    classifiers["gradient_boosting"] = joblib.load('models/ensemble_sentiment_model/best_classifier_gradient_boosting.pkl')
                    classifiers["logistic_regression"] = joblib.load('models/ensemble_sentiment_model/best_classifier_lojistik_regresyon.pkl')
                    print("Individual classifiers successfully loaded.")
                except Exception as e:
                    print(f"Error loading classifier models: {str(e)}")
                    return False
            
            # Load tokenizer and config
            with open('models/ensemble_sentiment_model/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            
            with open('models/ensemble_sentiment_model/config.json', 'r') as f:
                config = json.load(f)
            
            print("Ensemble model components successfully loaded.")
            return True
        except Exception as e:
            print(f"Error loading model files: {str(e)}")
            return False
            
    except Exception as e:
        print(f"General error during model setup: {str(e)}")
        return False

# Fallback sentiment analysis for cases where model fails
def fallback_sentiment_analysis(text):
    """Fallback dictionary-based sentiment analysis (English only)"""
    text = text.lower()
    
    # English positive and negative words
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'love', 'best', 'nice', 'perfect', 'happy',
        'wonderful', 'fantastic', 'terrific', 'outstanding', 'superb', 'brilliant', 'awesome',
        'impressive', 'exceptional', 'delightful', 'pleasant', 'superior', 'remarkable',
        'enjoy', 'beautiful', 'positive', 'favorite', 'better', 'recommend', 'satisfied',
        'useful', 'helpful', 'success', 'worth', 'quality', 'reliable', 'valuable'
    ]
    
    negative_words = [
        'bad', 'terrible', 'horrible', 'worst', 'hate', 'awful', 'poor', 'disappointing',
        'unfortunate', 'unpleasant', 'inadequate', 'inferior', 'useless', 'frustrating',
        'annoying', 'irritating', 'pathetic', 'mediocre', 'substandard', 'defective',
        'fail', 'failure', 'horrible', 'dislike', 'problem', 'waste', 'broke', 'broken',
        'terrible', 'disaster', 'avoid', 'sad', 'garbage', 'wrong', 'error', 'issues'
    ]
    
    # Count positive and negative words in the text
    words = text.split()
    positive_count = sum(1 for word in words if any(pos in word for pos in positive_words))
    negative_count = sum(1 for word in words if any(neg in word for neg in negative_words))
    
    # Common phrases that strongly indicate sentiment
    if "highly recommend" in text or "strongly recommend" in text:
        positive_count += 3
    if "waste of money" in text or "waste of time" in text:
        negative_count += 3
    if "never buy" in text or "never use" in text:
        negative_count += 2
      # Determine sentiment based on counts
    if positive_count > negative_count:
        sentiment = "Pozitif"  # Changed to Turkish
        confidence = min(0.5 + (0.5 * (positive_count / (positive_count + negative_count + 0.0001))), 0.95)
    elif negative_count > positive_count:
        sentiment = "Negatif"  # Changed to Turkish
        confidence = min(0.5 + (0.5 * (negative_count / (positive_count + negative_count + 0.0001))), 0.95)
    else:
        # If equal or no sentiment words found, default to neutral
        sentiment = "Nötr"  # Changed to Turkish
        confidence = 0.5
    
    return sentiment, confidence

def predict_sentiment(text):
    """Predict sentiment using the ensemble model or fallback to dictionary approach"""
    global lstm_feature_extractor, ensemble_model, tokenizer, config, classifiers
    
    # Check if required models are loaded
    models_loaded = (lstm_feature_extractor is not None and 
                    tokenizer is not None and 
                    config is not None and 
                    (ensemble_model is not None or len(classifiers) > 0))
    
    if not models_loaded:
        # If models not loaded, use fallback approach
        print("Models not loaded, using fallback sentiment analysis")
        return fallback_sentiment_analysis(text)
    
    try:
        # 1. Tokenize the text
        sequences = tokenizer.texts_to_sequences([text])
        
        # 2. Apply padding
        if 'pad_sequences' not in globals():
            print("pad_sequences not available, using fallback")
            return fallback_sentiment_analysis(text)
            
        pad_sequences_func = globals()['pad_sequences']
        
        padded_sequence = pad_sequences_func(
            sequences, 
            maxlen=config.get('max_sequence_length', 50),
            padding='post'
        )
        
        # 3. Extract LSTM features - bu adım çok önemli
        # LSTM modelinin son katman çıktıları, sınıflandırıcının girdi verileri olarak kullanılıyor
        lstm_features = lstm_feature_extractor.predict(padded_sequence)
        print(f"LSTM özellikleri şekli: {lstm_features.shape}")
        
        # 4. Make prediction using either the full ensemble model or the best classifier
        model_used = ""
        if ensemble_model is not None:
            # Tam ensemble model var ise kullan
            model_used = "full_ensemble_model"
            # Full ensemble model is available - bu durumda padding yapılmış veri doğrudan kullanılır
            prediction = ensemble_model.predict(padded_sequence)[0][0]
        else:
            # LSTM özellikleri ve klasik ML modelini kullan
            # Use the best classifier according to config
            best_classifier_name = config.get('best_classifier', 'Random Forest').lower().replace(' ', '_')
            
            if best_classifier_name == 'random_forest' and 'random_forest' in classifiers:
                model_used = "random_forest"
                # Random forest sınıflandırıcısı, LSTM özelliklerini kullanıyor
                prediction = classifiers['random_forest'].predict_proba(lstm_features)[0][1]
            elif best_classifier_name == 'gradient_boosting' and 'gradient_boosting' in classifiers:
                model_used = "gradient_boosting"
                prediction = classifiers['gradient_boosting'].predict_proba(lstm_features)[0][1]
            elif best_classifier_name == 'logistic_regression' and 'logistic_regression' in classifiers:
                model_used = "logistic_regression"
                prediction = classifiers['logistic_regression'].predict_proba(lstm_features)[0][1]
            else:
                # If best classifier not found, use any available classifier
                for classifier_name, classifier in classifiers.items():
                    model_used = classifier_name
                    prediction = classifier.predict_proba(lstm_features)[0][1]
                    break
                else:
                    # No classifier available, use fallback
                    print("No classifiers available, using fallback")
                    return fallback_sentiment_analysis(text)
        
        print(f"Tahmin için kullanılan model: {model_used}")
        
        # 5. Convert prediction to sentiment and confidence
        if prediction >= 0.5:
            sentiment = "Pozitif"  # Turkish format
            confidence = float(prediction)
        else:
            sentiment = "Negatif"  # Turkish format
            confidence = float(1.0 - prediction)
        
        return sentiment, confidence
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        # If prediction fails, use fallback approach
        return fallback_sentiment_analysis(text)

# Try to load the ensemble model
print("Attempting to load ensemble model...")
try:
    load_ensemble_model()
except Exception as e:
    print(f"Error loading ensemble model: {str(e)}")
    print("Using dictionary-based fallback sentiment analysis")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    """API endpoint for analyzing a single text"""
    try:
        # Verify the request has JSON data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
            
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        try:
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Predict sentiment
            sentiment, confidence = predict_sentiment(processed_text)
            
            return jsonify({
                'text': text,
                'sentiment': sentiment,
                'confidence': round(confidence * 100, 2),
                'raw_score': float(confidence)
            })
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            # Return a default sentiment if something fails
            return jsonify({
                'text': text,
                'sentiment': 'Nötr',
                'confidence': 50,
                'raw_score': 0.5,
                'error': str(e)
            })
    except Exception as e:
        print(f"Critical error in analyze endpoint: {str(e)}")
        return jsonify({'error': 'Server error', 'details': str(e)}), 500

@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """API endpoint for analyzing multiple texts"""
    data = request.json
    texts = data.get('texts', [])
    
    if not texts:
        return jsonify({'error': 'No texts provided'}), 400
    
    results = []
    
    for text in texts:
        # Preprocess text
        processed_text = preprocess_text(text)
        
        # Predict sentiment
        sentiment, confidence = predict_sentiment(processed_text)
        
        results.append({
            'text': text,
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2)
        })
    
    return jsonify(results)

@app.route('/live-twitter', methods=['GET'])
def live_twitter_page():
    """Render the Twitter analysis page"""
    return render_template('twitter.html')

@app.route('/twitter-search', methods=['POST'])
def twitter_search():
    """API endpoint for Twitter search and sentiment analysis"""
    data = request.json
    query = data.get('query', '')
    count = int(data.get('count', 10))
    
    try:
        import tweepy
        # Add your Twitter API keys here
        consumer_key = ""
        consumer_secret = ""
        access_token = ""
        access_token_secret = ""
        
        if not all([consumer_key, consumer_secret, access_token, access_token_secret]):
            # Sample tweets for demonstration
            sample_tweets = [
                {"id": "1", "text": "This product is amazing! Very satisfied.", "user": "User1", "username": "user1", "date": "2025-05-15 10:30:00"},
                {"id": "2", "text": "I hate this, total waste of money.", "user": "User2", "username": "user2", "date": "2025-05-15 11:15:00"},
                {"id": "3", "text": "Average product. Could be better.", "user": "User3", "username": "user3", "date": "2025-05-15 12:00:00"},
                {"id": "4", "text": "The delivery was super fast, thanks!", "user": "User4", "username": "user4", "date": "2025-05-15 13:30:00"},
                {"id": "5", "text": "Package arrived damaged, terrible experience.", "user": "User5", "username": "user5", "date": "2025-05-15 14:45:00"},
            ]
            
            results = []
            for tweet in sample_tweets:
                processed_text = preprocess_text(tweet["text"])
                sentiment, confidence = predict_sentiment(processed_text)
                
                results.append({
                    'id': tweet['id'],
                    'text': tweet['text'],
                    'sentiment': sentiment,
                    'confidence': round(confidence * 100, 2),
                    'user': tweet['user'],
                    'username': tweet['username'],
                    'date': tweet['date']
                })
            
            return jsonify(results[:count])
        else:
            # Twitter API connection
            auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
            auth.set_access_token(access_token, access_token_secret)
            api = tweepy.API(auth)
            
            # Fetch tweets
            tweets = api.search_tweets(q=query, count=count, tweet_mode="extended", lang="en")
            
            results = []
            for tweet in tweets:
                text = tweet.full_text
                processed_text = preprocess_text(text)
                sentiment, confidence = predict_sentiment(processed_text)
                
                results.append({
                    'id': tweet.id,
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': round(confidence * 100, 2),
                    'user': tweet.user.name,
                    'username': tweet.user.screen_name,
                    'date': tweet.created_at.strftime('%Y-%m-%d %H:%M:%S')
                })
            
            return jsonify(results)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True)