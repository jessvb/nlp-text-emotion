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
########## Change this depending on where your recordings are located ##########
rec_dir = 'input/'
################################################################################

# #### Testing with other inputs
predictor = ktrain.load_predictor('models/bert_model')
predictor.get_classes()
# The classes are 'joy', 'sadness', 'fear', 'anger', 'neutral'

def getEmotionFromText(input_path):
    # Read in the entire text
    f = open(input_path, 'r')
    messages = f.read()
    f.close()

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
    df.to_csv(os.path.join('output', 'nlp_emotion_' + output_name + '.csv'), sep=',', index=False)

    print('🎉 Done! 🎉')
    print('See the output file:')
    print('output/' + 'nlp_emotion_' + output_name + '.csv')

if __name__ == '__main__':
    # Loop through specific files and analyze their text
    files_in_dir = [f for f in os.listdir(rec_dir) if os.path.isfile(os.path.join(rec_dir, f))]
    i = 0
    for f in files_in_dir:
        if f.split('.')[1] == 'txt':
            input_path = os.path.join(rec_dir,f)
            output_name = f.split('.')[0]

            print(f'Reading from {input_path}')

            getEmotionFromText(input_path)

        i += 1
        print(f"""Number of files to go: {len(files_in_dir) - i}
            Percent files done: {i/len(files_in_dir)*100}\n""")
