#Create a GUI
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from Chat import chat_response, getparticularResponse, model, intents, fe, feature_scores
from keras.preprocessing.image import load_img
import pyttsx3
import speech_recognition as sr

filename=''
images=[]
filepath=[]

engine = pyttsx3.init('sapi5')
rate = engine.getProperty('rate')
engine.setProperty('rate', 200)

#Functions
def type():
    msg= EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0", END)
    global img_Insert, images

    if msg:
        text_widget.config(state=NORMAL)
        text_widget.insert(END,'You:'+msg+'\n\n')
        text_widget.config(foreground="#442265", font=("Verdana", 12 ))

        resp= chat_response(msg)
        text_widget.insert(END, 'Bot:' + resp + '\n\n')

        text_widget.config(state=DISABLED)
        text_widget.yview(END)

def speech():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print('Listening')
        audio= r.listen(source)
    try:
        query= r.recognize_google(audio, language='en-in')
        print(query)

        text_widget.config(state=NORMAL)
        text_widget.insert(END,'You:'+query+'\n\n')
        text_widget.config(foreground="#442265", font=("Verdana", 12 ))

        resp= chat_response(query)
        engine.say(resp)
        engine.runAndWait()
        text_widget.insert(END, 'Bot:' + resp + '\n\n')
        engine.stop()

        text_widget.config(state=DISABLED)
        text_widget.yview(END)

    except Exception as e:
        text_widget.config(state=NORMAL)
        text_widget.config(foreground="#442265", font=("Verdana", 12 ))

        ints= 'gibberish'
        resp= getparticularResponse(ints, intents)
        engine.say(resp)
        engine.runAndWait()
        text_widget.insert(END, 'Bot:' + resp + '\n\n')
        engine.stop()

        text_widget.config(state=DISABLED)
        text_widget.yview(END)

def upload():
    global filename
    filename= filedialog.askopenfilename(initialdir='./static/uploads',
                                            title='Select a File', filetypes=(('png files','*.png'),('All files','*.*')))
    text_widget.config(state=NORMAL)
    filepath.append(filename)

    if len(filepath) == 1:
        img = load_img((filepath[0]), target_size=(224, 224))
        query = fe.extract(img)
        result = feature_scores(query)
        filepath.clear()
        engine.say('Here are some products you were looking for')
        engine.runAndWait()
        text_widget.insert(END, 'Bot: Here are some products you were looking for:'+'\n\n')
        engine.stop()
        for i in result:
            new_img = Image.open(i)
            resized = new_img.resize((100, 100), Image.ANTIALIAS)
            img_Insert = ImageTk.PhotoImage(resized)
            text_widget.image_create(END, image=img_Insert, padx=5, pady=5)
            images.append(img_Insert)
    text_widget.insert(END,'\n\n')
    text_widget.config(state=DISABLED)
    text_widget.yview(END)

def clear():
    text_widget.config(state=NORMAL)
    text_widget.delete(1.0, END)
    text_widget.config(state=DISABLED)

#Create Chatbot window
base= Tk()
base.title('Chatbot')
base.geometry('500x500')
base.resizable(width=False, height=False)

#Create Text Window
text_widget= Text(base, bd=0, bg='white', font='Helvetica 14')
text_widget.config(cursor='arrow', state=DISABLED)

#Scroll bar
scrollbar= Scrollbar(base, command=text_widget.yview, cursor='arrow')
text_widget['yscrollcommand']= scrollbar.set


#Send button
SendButton= Button(base, font=("Verdana",12,'bold'), text="Send",
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command=type)

#Microphone image
mic_img = Image.open('./microphone.png')
resized_img = mic_img.resize((40,40), Image.ANTIALIAS)
mic= ImageTk.PhotoImage(resized_img)
mic_label= Label(image=mic)

#Speak button
SpeakButton= Button(base, font=("Verdana",12,'bold'), image=mic,
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command=speech)

#Upload Button
UploadButton= Button(base,font=("Verdana",12,'bold'), text="Upload Files",
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command=upload)

#Clear Button
ClearButton= Button(base,font=("Verdana",12,'bold'), text="Clear",
                    bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
                    command=clear)

#Box to enter the message
EntryBox= Text(base,  bd=0, bg="white",width="29", height="5", font="Helvetica 14")

#Placing all the components
scrollbar.place(x=476, y=6, height=386)
text_widget.place(x=6,y=6, height=386, width=470)
EntryBox.place(x=200, y=401, height=90, width=295)
SendButton.place(x=6, y=401, height=45, width=120)
UploadButton.place(x=6, y=447, height=45, width=120)
SpeakButton.place(x=128, y=433, height=60, width=70)
ClearButton.place(x=128, y=401, height=30, width=70)

base.mainloop()


