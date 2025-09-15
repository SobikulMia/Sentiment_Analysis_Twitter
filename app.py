
from flask import Flask, request, jsonify, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# Load Model & Vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(word for word in text.split() if word not in stop_words)
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())
    return text

@app.route('/')
def home():
    return render_template('index.html')  

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    texts = data.get('texts', [])
    results = []
    sentiment_map = {0: "Positive", 1: "Negative", 2: "Neutral"}
    for text in texts:
        clean = clean_text(text)
        vect = tfidf.transform([clean])
        pred = model.predict(vect)[0]
        results.append({'text': text, 'sentiment': sentiment_map.get(pred, 'Unknown')})
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
