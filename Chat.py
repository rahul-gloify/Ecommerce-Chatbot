import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
import pandas as pd

from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
model = load_model('trained_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def tokenize(sentence):
    """ Tokenize the User sentences"""
    sentence_words= nltk.sent_tokenize(sentence)
    sentence_words= [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    """Creates a bag of words vector for each of the word in the User sentence which will later
    be fed to the model"""
    sentence_words= tokenize(sentence)
    bag= [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w==s:
                bag[i]=1

    return (np.array(bag))

def predict_intent(sentence, model):
    """ Predict the intent of the User sentence and returning it"""
    bag= bow(sentence, words)
    pred= model.predict(np.array([bag]))[0]
    Threshold=0.25
    results= [[i,r] for i,r in enumerate(pred) if r>Threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    predicted_intent= classes[results[0][0]]
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})
    return return_list, predicted_intent

def getResponse(ints, intents_json):
    tag= ints[0]['intent']
    list_of_intents= intents_json['intents']
    for i in list_of_intents:
        if (i['tag']==tag):
            response= random.choice(i['responses'])
            break
    return response

def getparticularResponse(ints, intents_json):
    tag= ints
    list_of_intents= intents_json['intents']
    for i in list_of_intents:
        if (i['tag']==tag):
            response= random.choice(i['responses'])
            break
    return response

def chat_response(msg):
    ints, intent= predict_intent(msg, model)
    resp= getResponse(ints, intents)
    return resp

class FeatureExtractor:
    def __init__(self):
        # Use VGG-16 as the architecture and ImageNet for the weight
        base_model = VGG16(weights='imagenet')
        # Customize the model to return features from fully-connected layer
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
    def extract(self, img):
        #img = load_img(('\static\uploads,img_name), target_size=(224, 224))
        # Resize the image
        img = img.resize((224, 224))
        # Convert the image color space
        img = img.convert('RGB')
        # Reformat the image
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract Features
        feature = self.model.predict(x)[0]
        return feature / np.linalg.norm(feature)
fe = FeatureExtractor()

def feature_scores(query):
  features = pd.read_csv('./features/feature_extraction.csv')
  features_data = features.copy()
  features_data = features_data.drop(columns=['image'])
  features_data = features_data.values
  dists = np.linalg.norm(features_data - query, axis=1)

  # Extract images that have lowest distance
  ids = np.argsort(dists)[:5]
  lookalike_imgs = features.iloc[ids, :]['image']
  final_imgs = []
  for img in lookalike_imgs:
      final_imgs.append("./static/images/" + img)
  return final_imgs