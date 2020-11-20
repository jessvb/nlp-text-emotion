# Emotion Classification in short texts with BERT
# Applying BERT to the problem of multiclass text classification. Our dataset 
# consists of written dialogs, messages and short stories. Each dialog 
# utterance/message is labeled with one of the five emotion categories: 
# joy, anger, sadness, fear, neutral. 

## Workflow: 
# 1. Import Data
# 2. Data preprocessing and downloading BERT
# 3. Training and validation
# 4. Saving the model
# Multiclass text classification with BERT and 
# [ktrain](https://github.com/amaiya/ktrain). Use google colab for a free GPU.

import pandas as pd
import numpy as np
import time 

import ktrain
from ktrain import text

data_train = pd.read_csv('data/data_train.csv', encoding='utf-8')
data_test = pd.read_csv('data/data_test.csv', encoding='utf-8')

X_train = data_train.Text.tolist()
X_test = data_test.Text.tolist()

y_train = data_train.Emotion.tolist()
y_test = data_test.Emotion.tolist()

data = data_train.append(data_test, ignore_index=True)

class_names = ['joy', 'sadness', 'fear', 'anger', 'neutral']

print('size of training set: %s' % (len(data_train['Text'])))
print('size of validation set: %s' % (len(data_test['Text'])))
print(data.Emotion.value_counts())

data.head(10)

encoding = {
    'joy': 0,
    'sadness': 1,
    'fear': 2,
    'anger': 3,
    'neutral': 4
}

# Integer values for each class
y_train = [encoding[x] for x in y_train]
y_test = [encoding[x] for x in y_test]


# ## 2. Data preprocessing
# * The text must be preprocessed in a specific way for use with BERT. This is 
# accomplished by setting preprocess_mode to ‘bert’. The BERT model and 
# vocabulary will be automatically downloaded BERT can handle a maximum length 
# of 512, but let's use less to reduce memory and improve speed. 
(x_train,  y_train), (x_test, y_test), preproc = text.texts_from_array(x_train=X_train, y_train=y_train,
                                                                       x_test=X_test, y_test=y_test,
                                                                       class_names=class_names,
                                                                       preprocess_mode='bert',
                                                                       maxlen=350, 
                                                                       max_features=35000)

# ## 2. Training and validation

# Loading the pretrained BERT for text classification 
model = text.text_classifier('bert', train_data=(x_train, y_train), preproc=preproc)


# Wrap it in a Learner object
learner = ktrain.get_learner(model, train_data=(x_train, y_train), 
                             val_data=(x_test, y_test),
                             batch_size=6)

# Train the model. More about tuning learning rates 
# [here](https://github.com/amaiya/ktrain/blob/master/tutorial-02-tuning-learning-rates.ipynb)
learner.fit_onecycle(2e-5, 3)

# Validation
learner.validate(val_data=(x_test, y_test), class_names=class_names)

# #### Testing with other inputs
predictor = ktrain.get_predictor(learner.model, preproc)
predictor.get_classes()

message = 'I just broke up with my boyfriend'

start_time = time.time() 
prediction = predictor.predict(message)

print('predicted: {} ({:.2f})'.format(prediction, (time.time() - start_time)))


# ## 4. Saving Bert model
# let's save the predictor for later use
predictor.save("models/bert_model")

# Done! to reload the predictor use: ktrain.load_predictor
