from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from yake import KeywordExtractor
import numpy as np 
import pandas as pd
import os
import itertools
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow import keras
from io import StringIO
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
layers = keras.layers
models = keras.models

def train_test_split(data, train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train, test

def plot_confusion_matrix(cm, classes,title='Confusion matrix',cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=25)
    plt.xlabel('Predicted label', fontsize=25)


'''
Solution 1: Using YAKE (https://github.com/LIAAD/yake)
----------

This is a handy keyword extraction library so more for tagging but also useful if you rank the tags and then apply the LOWER score that has a match in your category data set. 
Note that the LOWER the score, the MORE relevant the keyword extracted. Subtract the floating point from an integer if you want to see it HIGHER but an unnecessary operation IMHO
'''
def solution1():
    inputText=None
    with open("TESTDATA.TXT",'r') as f:
        inputText=f.read()
    extractor=KeywordExtractor()
    print(filteredText)
    print(extractor.extract_keywords(filteredText))

'''
End of Soultion 1
'''

'''
Solution 2: Just rename the data set from TESTDATA.CSV
----------
'''
def solution2():
    data = pd.read_csv("TESTDATA.CSV")
    train_size = int(len(data) * .8)
    print ("Train size: %d" % train_size)
    print ("Test size: %d" % (len(data) - train_size))
    train_cat, test_cat = train_test_split(data['category'], train_size)
    train_text, test_text = train_test_split(data['text'], train_size)
    max_words = 1000
    tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words,char_level=False)
    #Fit tokenizer to our training text data
    tokenize.fit_on_texts(train_text) 
    x_train = tokenize.texts_to_matrix(train_text)
    x_test = tokenize.texts_to_matrix(test_text)
    #Utility to convert label strings to numbered index
    encoder = LabelEncoder()
    encoder.fit(train_cat)
    y_train = encoder.transform(train_cat)
    y_test = encoder.transform(test_cat)
    #Converts the labels to a one-hot representation
    num_classes = np.max(y_train) + 1
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    '''
    This model trains very quickly and 2 epochs are already more than enough
    Training for more epochs will likely lead to overfitting on this dataset
    You can try tweaking these hyperparamaters when using this model with your own data
    '''
    batch_size = 32
    epochs = 2
    drop_ratio = 0.5
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_split=0.1)

    #Following is to simply visualize the output    
    y_softmax = model.predict(x_test)
    text_labels = encoder.classes_
    y_test_1d = []
    y_pred_1d = []

    for i in range(len(y_test)):
        probs = y_test[i]
        index_arr = np.nonzero(probs)
        one_hot_index = index_arr[0].item(0)
        y_test_1d.append(one_hot_index)

    for i in range(0, len(y_softmax)):
        probs = y_softmax[i]
        predicted_index = np.argmax(probs)
        y_pred_1d.append(predicted_index)

    cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
    plt.figure(figsize=(18,14))
    plot_confusion_matrix(cnf_matrix, classes=text_labels, title="Confusion matrix")
    plt.show()

'''
End of Solution 2
'''

'''
Solution 3: LinearSVCs can at times be more accurate than NaiveBayes, don't forget to rename from TESTDATA.CSV to the uri of your file

WARNING - You may run into some memory errors if you do not have enough RAM (cause of high volume input)
----------
'''
def solution3():
    df = pd.read_csv('TESTDATA.CSV')
    df = df[pd.notnull(df['Consumer complaint narrative'])]
    col = ['Product', 'Consumer complaint narrative']
    df = df[col]
    df['category_id'] = df['Product'].factorize()[0]
    category_id_df = df[['Product', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Product']].values)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

    features = tfidf.fit_transform(df['Consumer complaint narrative']).toarray()
    labels = df.category_id

    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    cv_df = pd.DataFrame(index=range(5 * len(models)))
    entries = []
    for model in models:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=5)
        for fold_idx, accuracy in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    print(cv_df.groupby('model_name').accuracy.mean())

'''
End of Solution 3
'''
#solution1()
#solution2()
#solution3()
