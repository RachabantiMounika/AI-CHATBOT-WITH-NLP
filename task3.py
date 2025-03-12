import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample responses and sentences (training data)
corpus = [
    "Hello",
    "Hi",
    "How are you?",
    "I'm doing good, how about you?",
    "What's your name?",
    "I'm a chatbot created to assist you.",
    "Tell me a joke",
    "Why did the chicken cross the road? To get to the other side!",
    "Goodbye",
    "See you later"
]

responses = [
    "Hello! How can I help you?",
    "Hi! How can I assist you today?",
    "I'm doing well, thank you!",
    "That's great to hear! I'm here to assist you.",
    "I am ChatBot, your virtual assistant!",
    "I can tell you jokes, answer questions, and more!",
    "Haha! That’s a good one, isn't it?",
    "Goodbye! Have a great day!",
    "Take care! See you soon!"
]

# Initialize TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the corpus and transform the corpus into TF-IDF vectors
vectorized_corpus = vectorizer.fit_transform(corpus)

def get_response(user_input):
    # Convert user input into a vector using the same vectorizer
    vectorized_user_input = vectorizer.transform([user_input])
    
    # Compute the cosine similarity between the user input and the corpus
    cosine_similarities = cosine_similarity(vectorized_user_input, vectorized_corpus)
    
    # Get the index of the most similar response
    most_similar_idx = np.argmax(cosine_similarities)
    
    return responses[most_similar_idx]

# Run the chatbot
print("Chatbot: Hello! How can I assist you today?")
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye! Have a nice day!")
        break
    
    response = get_response(user_input)
    print(f"Chatbot: {response}")
