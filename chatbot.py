import nltk
from nltk.chat.util import Chat, reflections

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
    ],
    [
        r"(.*)",
        ["Sorry, I didn't understand that.", "Can you rephrase?", "I'm not sure I follow you."]
    ],
]

chatbot = Chat(pairs, reflections)


print("Hi! I am a basic chatbot. Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        print("Chatbot: Bye, take care!")
        break
    response = chatbot.respond(user_input)
    print(f"Chatbot: {response}")
