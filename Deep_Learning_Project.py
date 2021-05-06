# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:20:05 2019

@author: Vishnudatha
"""

import numpy as np

def load_data(fname):
    with open(fname,'r') as f:
        utt_ID = [line.split(None, 1)[0] for line in f]
    with open(fname,'r') as f:
        y = [line.split('\t', 2)[1] for line in f]
    with open(fname,'r') as f:
        x_train_org =[line.strip().split('\t', 2)[2] for line in f]
        x_train = [x.split(';') for x in x_train_org]
    return utt_ID,y,x_train 

def load_test_data(fname):
    with open(fname,'r') as f:
        utt_ID = [line.split(None, 1)[0] for line in f]
    with open(fname,'r') as f:
        x_test_org =[line.strip().split('\t', 2)[1] for line in f]
        x_test = [x.split(';') for x in x_test_org]
    return utt_ID,x_test 

input_length = 40

def add_spaces(x):
    return x[:input_length]+['' for i in range(input_length-len(x))]

def process_input_x_for_length(x):
    x_processed=[None]*len(x)
    for i in range(len(x)):
        res = []
        for list in x[i]:
            res.append(str(list).lower().split())
        x_processed[i]=res
    return x_processed

def process_input_x(x):
    x_processed=[None]*len(x)
    for i in range(len(x)):
        res = []
        for list in x[i]:
            res.append(str(list).lower().split())
        x_processed[i]=res
    x_processed_with_spaces= np.array([[add_spaces(x) for x in uttr] for uttr in x_processed])
    return x_processed_with_spaces

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        # ~ print(header)
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        # print(vocab_size)
        for line in range(vocab_size):
            # print(line)
            word = []
            while True:
                ch = f.read(1).decode('iso-8859-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            # print(word)
            if word in vocab:
                # print(word)
                word_vecs[word] = np.frombuffer(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    return word_vecs

def add_unknown_words(word_vecs, vocab, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)
            

def format_x(x):
    l=[]
    for i in x:
        m=[]
        for j in i:
            n=[]
            for k in j:
                n.append(W[word_idx_map[k]])
            m.append(n)
        l.append(m)
    x_em=np.array(l)
    l=[]
    for i in x_em:
        m=[]
        for j in i:
            m.append(j.flatten())
        l.append(m)
    x_format=np.array(l)
    return x_format

utt_ID_train,y_train_org,x_train_org= load_data("utterances.train")
x_train_org[0]
len(x_train_org)

utt_ID_valid,y_valid_org,x_valid_org= load_data("utterances.valid")
x_valid_org[0]
len(x_valid_org)

utt_ID_test,x_test_org= load_test_data("utterances.test")
x_test_org[0]
len(x_test_org)

x_tr_len= process_input_x_for_length(x_train_org)
lengths = []
for i in x_tr_len:
    for j in i:
        lengths.append(len(j))



max_length=max(lengths)
mean_length=np.int(np.mean(lengths))
forty_percentile =np.percentile([np.unique(lengths)],40)
print("max %d" % max_length)
print("mean %d" % mean_length)

print("50 percentile %d" % forty_percentile)

#import matplotlib.pyplot as plt
#plt.plot(lengths)
#plt.ylabel('length')
#plt.show()

#x_train,x_valid,x_test
x_train=process_input_x(x_train_org)
x_valid=process_input_x(x_valid_org)
x_test=process_input_x(x_test_org)

print("Printing Input data sizes")
print("x train shape",len(x_train))
print("x valid shape",len(x_valid))
print("x test shape",len(x_test))


uniqueWords = []
for i in range(len(x_train_org)):
        for j in range(len(x_train[i])):
             for k in x_train[i][j]:
                    if not k.lower() in uniqueWords:
                        uniqueWords.append(k.lower())
for i in range(len(x_valid_org)):
        for j in range(len(x_valid[i])):
             for k in x_valid[i][j]:
                    if not k.lower() in uniqueWords:
                        uniqueWords.append(k.lower())
for i in range(len(x_test_org)):
        for j in range(len(x_test[i])):
             for k in x_test[i][j]:
                    if not k.lower() in uniqueWords:
                        uniqueWords.append(k.lower())
print(len(uniqueWords)) 


import time
start = time.time()
w2v_file = "GoogleNews-vectors-negative300.bin"
w2v = load_bin_vec(w2v_file, uniqueWords)
print("num words found: %d" % len(w2v))
add_unknown_words(w2v, uniqueWords, k=300)
W, word_idx_map = get_W(w2v, k=300)

print("W shape: %s" % str(W.shape))

print("%d seconds to get the embeddings" % (time.time()-start))

#Embedd train, valid and test 
x_em_train= format_x(x_train)
print("X train embedded",x_em_train.shape)
x_em_valid= format_x(x_valid)
print("X valid embedded",x_em_valid.shape)
x_em_test= format_x(x_test)
print("X test embedded",x_em_test.shape)

#Labels
tags = ['%', '%--', '2', 'aa', 'aap', 'ar', 
          'b', 'ba', 'bc', 'bd', 'bh', 'bk', 
          'br', 'bs', 'cc', 'co', 'd', 'fa', 
          'ft', 'g', 'h', 'no', 'qh', 'qo', 
          'qrr', 'qw', 'qy', 's', 't1', 't3','x']

id_to_tags = dict()
for id_i,i in enumerate(tags):
    id_to_tags[i] = id_i

def tag_to_id(y):
    y_id=[None]*len(y)
    for id_i, i in enumerate(y):
        y_id[id_i]=id_to_tags[i]
    return np.array(y_id)

def id_to_tag(y_tag):
    y_id=[None]*len(y_tag)
    for id_i, i in enumerate(y_tag):
        y_id[id_i]=tags[i]
    return np.array(y_id)

#id_to_tags
y_train = tag_to_id(y_train_org)
y_valid = tag_to_id(y_valid_org)

print("y train shape:",y_train.shape)
print("y valid shape:",y_valid.shape)

#one hot representation of y_train, y_valid
from keras.utils import np_utils

y_train_id = np_utils.to_categorical(y_train)
y_valid_id = np_utils.to_categorical(y_valid)

num_classes= y_train_id.shape[1]

print("Num of classes", num_classes)

from keras.models import Model
from keras.layers import LSTM,Dropout
from keras.models import Sequential
from keras.layers import Dense,Bidirectional



model=Sequential()
time_steps=x_em_train.shape[1]
features=x_em_train.shape[2]

model.add(Bidirectional(LSTM(units= 100,  activation='tanh'),input_shape=(time_steps,features)))

model.add(Dense(num_classes, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(x_em_train, y_train_id, validation_data=(x_em_valid, y_valid_id), epochs=5, batch_size=512)

model.save('my_model.h5')

model_eval = model.evaluate(x_em_valid, y_valid_id, batch_size=128, verbose=0)
print("Accuracy: %.2f%%" % (model_eval[1]*100))

y_test_id = model.predict(x_em_test).argmax(axis=1)
print(y_test_id)

y_test = id_to_tag(y_test_id)
data = np.column_stack((utt_ID_test,y_test))

np.savetxt("3250569_Kanjur_topic1_result.txt", data, delimiter='\t', fmt='%s')