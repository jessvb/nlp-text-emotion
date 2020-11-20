# Emotion Classification in short texts with BERT
# Applying BERT to the problem of multiclass text classification. Our dataset 
# consists of written dialogs, messages and short stories. Each dialog 
# utterance/message is labeled with one of the five emotion categories: 
# joy, anger, sadness, fear, neutral. 

import pandas as pd
import numpy as np
import re
import os

import ktrain
from ktrain import text

################################################################################
########### Change these depending on what you name the transcripts ############
input_text = 'transcript.txt' # TODO read from file
input_path = os.path.join('input',input_text) # TODO read from file
output_name = input_text.split('.')[0]
################################################################################


class_names = ['joy', 'sadness', 'fear', 'anger', 'neutral']

encoding = {
    'joy': 0,
    'sadness': 1,
    'fear': 2,
    'anger': 3,
    'neutral': 4
}

# #### Testing with other inputs
predictor = ktrain.load_predictor('models/bert_model')
predictor.get_classes()

# Alllll the sentiments:
messages = """I was walking through the forest. It was dark. Something was in 
the shadows. As I looked to the left, I felt something slimy wrapping around 
my ankle. I jumped! It was only my dog giving me a generous lick. I love my 
dog! There's nothing better than sweet surprise doggo kisses. Although the last 
time this happened was on a dark day. I had been crying. And throwing books.
 I failed a test. But that's okay! It just gave me another opportunity to 
 try again :) """

messages_list = re.split('[\.!]', messages)
sentiments = []
i = 1
iter_percent = 0 # for printing percent done
for msg in messages_list:
    # print(f'predicting for sentence: "{msg}"')
    prediction = predictor.predict(msg)
    # print('predicted: {}\n'.format(prediction))
    sentiments.append(prediction)

    # every so often, show percent done
    percent_done = i/len(messages_list)*100
    if (percent_done > iter_percent):
        print('current message index: %.0f' % i)
        print('percent done: %.1f%%' % percent_done)
        iter_percent += 20
    i += 1

# Export predicted emotions to .csv format
df = pd.DataFrame({'EMOTION': sentiments, 'SENTENCE': messages_list})
df.to_csv(os.path.join('output', output_name + '_nlp_er.csv'), sep=',', index=False)

print('🎉 Done! 🎉')
print('See the output file:')
print('output/' + output_name + '_nlp_er.csv')