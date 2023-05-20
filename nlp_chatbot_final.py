from flask import Flask
import nltk  
nltk.download('punkt')  
nltk.download('wordnet') 
from nltk.stem import WordNetLemmatizer  
lemmatizer = WordNetLemmatizer()
import json  
import pickle 

import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import load_model
import random
from keras.models import load_model

 
model = load_model('chatbot_model.h5')
with open("C://Users//ASUS//Downloads//chatbot atg//intents.json",encoding="utf8") as json_data:
    intents = json.load(json_data)
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def clean(sentence): 
    words_sentence = nltk.word_tokenize(sentence)
    words_sentence = [lemmatizer.lemmatize(word.lower()) for word in words_sentence]
    return words_sentence



def bag_words(sentence, words):
    
    words_sentence = clean(sentence) 
    bag = [0]*len(words) 
    
    for ws in words_sentence:
        for i,w in enumerate(words):
            if w == ws:
                bag[i] = 1 
                
    return(np.array(bag))

def prediction(sentence, model):
   
    p = bag_words(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def final_response(classes_predicted, all_intents):
    tag = classes_predicted[0]['intent'] 
    list_intents = all_intents['intents']
    for i in list_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])  
            break
    return result


##app = Flask(__name__)
##@app.route('/chatbot_output/<string:message>')
def chatbot_output(message):
    classes_predicted = prediction(message, model) 
    res = final_response(classes_predicted, intents) 
    return res
        
print(chatbot_output('hello'))

##if __name__ == "__main__":
    ##app.run()
