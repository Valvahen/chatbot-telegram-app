import nltk
from nltk.chat.util import Chat, reflections
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import string

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define pairs of patterns and responses
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how are you today?",]
    ],
    [
        r"(hi|hello|hey)",
        ["Hello!", "Hi there!", "Hey!",]
    ],
    [
        r"how are you ?",
        ["I'm doing good, how about you?",]
    ],
    [
        r"sorry (.*)",
        ["It's alright.", "It's okay, no worries.",]
    ],
    [
        r"I am fine",
        ["Great to hear that!", "Alright, cool!",]
    ],
    [
        r"what is your name ?",
        ["I am a chatbot created by you.", "I'm your friendly chatbot!"]
    ],
    [
        r"quit",
        ["Bye, take care!", "Goodbye!", "It was nice talking to you!"]
    ]
]

# Preprocessing with NLP
def preprocess_text(text):
    # Tokenize text
    tokens = word_tokenize(text.lower())

    # Remove stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]

    # Stem the words
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(word) for word in filtered_tokens]

    return ' '.join(stemmed_tokens)

# Adding some context-based responses
def nlp_chatbot_response(user_input, corpus):
    user_input = preprocess_text(user_input)
    corpus.append(user_input)
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    index_of_best_match = similarity_scores.argsort()[0][-2]
    flat_similarity_scores = similarity_scores.flatten()
    flat_similarity_scores.sort()
    best_match_score = flat_similarity_scores[-2]

    if best_match_score == 0:
        response = "I'm sorry, I don't understand that."
    else:
        response = corpus[index_of_best_match]

    corpus.pop()  # Remove the last added user input to reset the corpus
    return response

# Creating a Chat object
chatbot = Chat(pairs, reflections)

# Small talk corpus for the chatbot to use NLP with
corpus = [
    "hello",
    "hi",
    "how are you",
    "i am fine",
    "what is your name",
    "sorry",
    "bye",
    "thank you",
    "thanks",
    "goodbye"
]

# Greet the user
print("Hi! I am a basic chatbot enhanced with NLP. Type 'quit' to exit.")

# Start chatting
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Bye, take care!")
        break
    elif any(phrase in user_input.lower() for phrase in ["thank", "thanks"]):
        print("Chatbot: You're welcome!")
    else:
        response = chatbot.respond(user_input)
        if response:
            print(f"Chatbot: {response}")
        else:
            print(f"Chatbot: {nlp_chatbot_response(user_input, corpus)}")
