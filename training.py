import nltk
from nltk import WordNetLemmatizer
import random
import numpy as np
import json
import pickle
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

words=[]
classes=[]
documents=[]
data_file=open('intents.json').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize the sentences
        w=nltk.sent_tokenize(pattern)
        words.extend(w)
        #add documents to the corpus
        documents.append((w, intent['tag']))

        #add all the intents to classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#lemmatization of words(vocab) and removing duplicates
words= [lemmatizer.lemmatize(word.lower()) for word in words]
words= sorted(list(set(words)))

#sort classes
classes= sorted(list(set(classes)))

#Saving the words & intent classes to pickle files
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]

#create an empty array for output labels
output_labels=[0]*len(classes)

#Creating bag of words (training data) for each user input
for doc in documents:
    bow=[]
    pattern_words= doc[0]
    pattern_words= [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bow.append(1) if w in pattern_words else bow.append(0)

    output_row=list(output_labels)
    output_row[classes.index(doc[1])]=1

    training.append([bow, output_row])

#Shuffle our training data and convert into array
random.shuffle(training)
training=np.array(training)

#Create training features and labels
train_x= list(training[:,0])
train_y= list(training[:,1])
print('Training data created')

#Create a Model
model= Sequential([
    Dense(128, input_shape=(len(train_x[0]),), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(train_y[0]), activation='softmax')
])

#Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-6),
              metrics=['accuracy'])

#Fit the model
history= model.fit(x=np.array(train_x),
                   y=np.array(train_y),
                   epochs=200,
                   batch_size=5,
                   verbose=1)

#Save the model
model.save('trained_model.h5',history)
print('Model saved')
