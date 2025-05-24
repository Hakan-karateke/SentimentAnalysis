import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# NLTK kaynaklarını indirme
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

def clean_tweet_text(text):
    # 1. Kullanıcı adlarını (@kullanici) kaldır
    text = re.sub(r'@\w+', '', text)
    # 2. URL'leri kaldır
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # 3. Hashtag sembolünü kaldır
    text = re.sub(r'#\w+', '', text)
    # 4. Özel karakterleri kaldır
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # 5. Küçük harfe dönüştür
    text = text.lower()
    # 6. Fazla boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
    return " ".join(filtered_tokens)

def preprocess_text(text):
    cleaned = clean_tweet_text(text)
    final = remove_stopwords(cleaned)
    return final
