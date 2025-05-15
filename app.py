import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from sentiment_model.text_preprocessing import preprocess_text

# Define variables that would normally be set by loading the model
model = None
tokenizer = None
config = None

app = Flask(__name__)

def load_sentiment_model():
    """Load the pre-trained LSTM sentiment model and tokenizer"""
    global model, tokenizer, config
    
    try:
        # Try to import TensorFlow/Keras
        try:
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            
            # Make pad_sequences globally accessible
            globals()['load_model'] = load_model
            globals()['pad_sequences'] = pad_sequences
            print("TensorFlow/Keras imports successful")
        except ImportError:
            try:
                from keras.models import load_model
                from keras.preprocessing.sequence import pad_sequences
                
                # Make pad_sequences globally accessible
                globals()['load_model'] = load_model
                globals()['pad_sequences'] = pad_sequences
                print("Keras imports successful")
            except ImportError:
                print("Failed to import Keras or TensorFlow. Will use fallback method.")
                return False
        
        # Try to load model
        try:
            model = load_model('sentiment_model/lstm_model.h5')
            
            with open('sentiment_model/tokenizer.pkl', 'rb') as f:
                tokenizer = pickle.load(f)
            
            with open('sentiment_model/config.json', 'r') as f:
                config = json.load(f)
            
            print("LSTM model successfully loaded.")
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
    """Predict sentiment using the LSTM model or fallback to dictionary approach"""
    global model, tokenizer, config
    
    # For now, always use the fallback approach since we're having issues with TensorFlow
    print("Using fallback sentiment analysis")
    return fallback_sentiment_analysis(text)
    
    # The following code is kept for future reference if you fix the TensorFlow installation
    """
    if model is None or tokenizer is None or config is None:
        # If model not loaded, try loading it
        if not load_sentiment_model():
            # If loading fails, use fallback approach
            return fallback_sentiment_analysis(text)
    
    try:
        # Tokenize the text
        sequences = tokenizer.texts_to_sequences([text])
        
        # Apply padding
        padded_sequence = pad_sequences(
            sequences, 
            maxlen=config.get('max_sequence_length', 50),
            padding='post'
        )
        
        # Make prediction
        prediction = model.predict(padded_sequence)[0][0]
        
        # Convert prediction to sentiment and confidence
        if prediction >= 0.5:
            sentiment = "Positive"
            confidence = float(prediction)
        else:
            sentiment = "Negative"
            confidence = float(1.0 - prediction)
        
        return sentiment, confidence
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        # If prediction fails, use fallback approach
        return fallback_sentiment_analysis(text)
    """

# Skip model initialization since we're using the fallback approach
print("Using dictionary-based fallback sentiment analysis")
# Commented out to avoid errors: load_sentiment_model()

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