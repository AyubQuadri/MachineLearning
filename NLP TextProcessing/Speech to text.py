"""
Created on Tue Aug 22 13:20:06 2017

@author: quadris
@Project: Audio Sentiment analysis.
@ Steps:
    1. Setup the microphone, specifiy which mic to be selected & record sentiments speech about movie or product
    2. feed it Google recognizer so that it convert Speech to text
    3. Using TextBlob perform Sentiment Analysis, POS tagging etc., 
"""

# Libraries which are required.
import speech_recognition as sr 
from textblob import TextBlob

##########################################################################################
## Step 1. Setup the microphone, specifiy which mic to be selected & record sentiments speech about movie or product
    # Fav mic to be used as Plantronics others can use other mic.
mic_name ="Transmit (Plantronics DA40)"  #"Microphone (Realtek Audio)"

# sample rate how often values are recorded. & Chunk size used to store the data in bytes (better to be power of 2 -> 512 or 1024 or 2048) 
sample_rate = 48000
chunk_size = 2048

#Initilize the recorder 
r = sr.Recognizer()

# list of microphones that are connected to the machine & selec the desired one.
mic_list = sr.Microphone.list_microphone_names()
print(mic_list)
for i, microphone_name in enumerate(mic_list):
    	if microphone_name == mic_name:
		    device_id = i
print(microphone_name)
# specify the microphone to use a device, how often to record and data chunk size  
# to reduce background noise & start recording using listen()     
with sr.Microphone(device_index = device_id, sample_rate = sample_rate, chunk_size = chunk_size) as source:
    r.adjust_for_ambient_noise(source)
    print("Say Something")
    audio = r.listen(source)
    # positive sentiment 
        #"The product release was effective, and the over all release was good as well as smooth"
    # Negative sentiment    
        # "The movie was very bad, first half of the movie was ok but the second half was boring"
  
        
###########################################################################################        
## Step 2. feed it Google recognizer so that it convert Speech to text    
#Connect to google Speech to text recongizer using audio file with langauge as english GB -> greate Britan 
       
try:
    text = r.recognize_google(audio, language='en-GB')
    print("You said " + text)
   
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
         print("Could not request results from Google Speech Recognition Service; {0}".format(e))

         
###########################################################################################         
##Step 3. Using TextBlob perform Sentiment Analysis, POS tagging etc.,        
# sentiment analysis using TextBlob.
print(text)
text = TextBlob(text)    

#1. Parts of Speech tagging POS
pos= text.tags
print(pos)

#2. Sentiment Analysis
    # setiments result will be tuple (polarity, subjectivity)
    # polarity ranges from -1 to 1 
        # -1 -> Negative sentiments 
        # 1  -> Positive sentiments
    # Subjectivity ranges from 0 to 1
        # 0 -> very objective
        # 1 -> very subjective
senti = text.sentiment
print(senti)

# appropriatly print the review status.
if(senti.polarity<0):
    print("Negative review")
else:
    print("Positive review")
    

