import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import requests
import csv
import Levenshtein as lev

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def load_dialogs(file_path):
    pairs = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row
        for row in reader:
            question, answer = row
            pairs.append([question, [answer]])  # Format responses as a list
    return pairs

pairs = load_dialogs('dialogs.csv')

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # stop_words = set(stopwords.words('english'))
    # filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def nlp_chatbot_response(user_input, pairs):
    user_input_preprocessed = preprocess_text(user_input)
    
    questions = [preprocess_text(pair[0]) for pair in pairs]
    answers = [pair[1] for pair in pairs]
    
    # Levenshtein distance matching
    distances = [lev.distance(user_input_preprocessed, question) for question in questions]
    min_distance = min(distances)
    index_of_best_match = distances.index(min_distance)
    print(user_input_preprocessed)
    print(questions[index_of_best_match])
    
    if min_distance < 10:  # Adjust threshold as needed
        return random.choice(answers[index_of_best_match])
    else:
        return None

def wikipedia_search(query):
    try:
        url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'format': 'json',
            'titles': query,
            'prop': 'extracts',
            'exintro': True,
            'explaintext': True
        }
        response = requests.get(url, params=params)
        data = response.json()
        pages = data['query']['pages']

        for page_id, page in pages.items():
            if 'extract' in page:
                extract = page['extract']
                sentences = extract.split('. ')
                if len(sentences) > 3:
                    return '. '.join(sentences[:3]) + '.'
                else:
                    return extract

        return "I couldn't find any information about that topic."
    except Exception as e:
        return f"Sorry, I couldn't retrieve the information. Error: {str(e)}"

print("Hi! I am a basic chatbot enhanced with NLP and Wikipedia search. Type 'quit' to exit.")
while True:
    user_input = input("You: ").lower()
    if user_input == 'quit':
        print("Chatbot: Bye, take care!")
        break
    elif "tell me about" in user_input:
        search_query = user_input.replace("tell me about", "").strip()
        print(f"Chatbot: Searching for {search_query}...")
        search_result = wikipedia_search(search_query)
        print(f"Chatbot: {search_result}")
    elif any(phrase in user_input for phrase in ["thank", "thanks"]):
        print("Chatbot: You're welcome!")
    else:
        response = nlp_chatbot_response(user_input, pairs)
        print(f"Chatbot: {response}")
