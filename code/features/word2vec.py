"""
This program will 

    -->Form the feature vectors and the training and testing dataset
    -->construct the statements which are required for the training and testing 
"""

import pandas as pd
import random
import multiprocessing
import sys
from gensim.models.doc2vec import TaggedDocument,Doc2Vec
from sklearn.svm import SVC
from sklearn import preprocessing

import gensim
import pickle
from gensim import corpora
from gensim import models
from collections import defaultdict
import logging
import os,sys
import re
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
from IdentifyNativeLanguage import readFromFile, readTrainingData
from nltk.tag import pos_tag
from IdentifyNativeLanguage import *

cores = multiprocessing.cpu_count()

nl_to_index={
            "ARA":0,
            "CHI":1,
            "FRE":2,
            "GER":3,
            "HIN":4,
            "ITA":5,
            "JPN":6,
            "KOR":7,
            "SPA":8,
            "TEL":9,
            "TUR":10
    }

number_words=["one","two","twenty","three","thirty","four","forty","five","fifty","six","sixty","seven","seventy","eight","eighty","nine","ninety","thousand","lakhs","crore","million","billion"]


def edits(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return list(set(deletes + transposes + replaces + inserts))


def tagNNPAndNumbers(data):
    all_possibilities=[]
    for word in number_words:
        all_possibilities.extend(edits(word))
    for nl in nl_to_index:
        #print(row['NL'],row['Text'])
        #print(i)
        for identifier in data[nl]:
            row=data[nl][identifier]
            tagged_doc = pos_tag(row.split())
            #print(tagged_sent)
            new_text=""
            for j,tagged_word in enumerate(tagged_doc):
                if tagged_word[1]=="NNP":
                    #print(tagged_word)
                    #tagged_doc[j][0]="NNP"
                    new_text+="NNP "
                elif tagged_word[0] in all_possibilities:
                    new_text+="NUM "
                    #print("happenning")
                else:
                    new_text+=tagged_doc[j][0]+" "
            new_text=new_text.strip()
            data[nl][identifier]=new_text
            """print(data[nl][identifier])
            print("="*100)"""
            """if nl==100:
                break"""
            #sys.exit()
    return data

def writeDataToFile(filename,data):
    fhw=open(filename,"w")
    for nl_num,nl in enumerate(data):
        for each_doc in data[nl]:
            fhw.write(str(nl_to_index[nl])+" ")
            for num,val in enumerate(each_doc):
                fhw.write(str(num+1)+":"+str(val)+" ")
            fhw.write("\n")

def prepareWord2VecModels(training_data,testing_data,num_of_features,choices_of_window,msg):
    entire_text_training=[]
    entire_text_testing=[]
    """fhr=open("wikidump","r")
    entire_text_training=[text for text in fhr.read()]"""
    
    
    for nl in training_data:
        entire_text_training.extend([text.split() for text in training_data[nl].values()])
        #break
        #entire_text_testing.extend([text.split() for text in training_data[nl].values()])
        entire_text_testing.extend([text.split() for text in testing_data[nl].values()])
    """for row in entire_text_training:
        print(row[:3])"""
    #sys.exit()
    #print(entire_text_training[:1])
    #sys.exit()
    #print(len(entire_text))
    #pprint(entire_text)
    corpus_models_training={}
    corpus_models_testing={}
    for f in num_of_features:
        for w in choices_of_window:
            corpus_models_training[str(f)+"_"+str(w)]=models.Word2Vec(entire_text_training, size=f, window=w, min_count=5, workers=30,iter=5)
            corpus_models_testing[str(f)+"_"+str(w)]=models.Word2Vec(entire_text_testing, size=f, window=w, min_count=5, workers=30,iter=5)
            
            print("Corpus Model Training over for ",str(f),"features and ",str(w),"window")
            #fs, temp_path = mkstemp("corpus_word2vec_models/"+str(f)+"_"+str(w)+".model")  # creates a temp file
            
            if msg=="":
                corpus_models_training[str(f)+"_"+str(w)].save("corpus_word2vec_models/"+str(f)+"_"+str(w)+".training.model")  # save the model
                corpus_models_testing[str(f)+"_"+str(w)].save("corpus_word2vec_models/"+str(f)+"_"+str(w)+".testing.model")
            else:
                corpus_models_training[str(f)+"_"+str(w)].save("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".training.model")  # save the model
                corpus_models_testing[str(f)+"_"+str(w)].save("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".testing.model")
            

def calculateWordToVector(training_data,testing_data,num_of_features,choices_of_window,msg="",perform_std=True,recalculate=True,perform_mean=True,perform_sum=True,perform_tfidf=True):
    
    if recalculate:
        prepareWord2VecModels(training_data,testing_data,num_of_features,choices_of_window,msg)
        #return
    for f in num_of_features:
        for w in choices_of_window: 
            print("Starting with ",f,"features and",w,"window")
            if perform_mean:
                data_mean_training={}
                data_mean_testing={}
                for nl in training_data:
                    data_mean_training[nl]=[]
                    data_mean_testing[nl]=[]
            if perform_sum:
                data_sum_training={}
                data_sum_testing={}
                for nl in training_data:
                    data_sum_training[nl]=[]
                    data_sum_testing[nl]=[]
            
            if perform_std:
                data_std_training={}
                data_std_testing={}
                for nl in training_data:
                    data_std_training[nl]=[]
                    data_std_testing[nl]=[]
            
                    
            if perform_tfidf:
                data_tfidf_training={}
                data_tfidf_testing={}
                for nl in training_data:
                    data_tfidf_training[nl]=[]
                    data_tfidf_testing[nl]=[]
                entire_text_training=[]
                entire_text_testing=[]
                for nl in training_data:
                    #print(data[nl].values())
                    entire_text_training.extend([text.split() for text in training_data[nl].values()])
                    entire_text_testing.extend([text.split() for text in training_data[nl].values()])
                    entire_text_testing.extend([text.split() for text in testing_data[nl].values()])
                    
                """print(entire_text)
                sys.exit()"""
                docs_training = [[token for token in text] for text in entire_text_training]
                docs_testing = [[token for token in text] for text in entire_text_testing]
                """print(docs[0])
                print(entire_text[0])"""
                dictionary_training = corpora.Dictionary(entire_text_training)
                dictionary_testing = corpora.Dictionary(entire_text_testing)
                #print(dictionary)
                #dictionary.save_as_text("temp/dictionary")
                
                corpus_training = [dictionary_training.doc2bow(text) for text in docs_training]
                corpus_testing = [dictionary_training.doc2bow(text) for text in docs_testing]
                #print(docs[0])
                tfidf_training = models.TfidfModel(corpus_training)
                tfidf_testing = models.TfidfModel(corpus_testing)
                #print(tfidf)
                #return
            
            #fs, temp_path = mkstemp("corpus_word2vec_models/"+str(f)+"_"+str(w)+".model")  
            if msg=="":
                model_training = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+".training.model")
                model_testing = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+".testing.model")
                #model_testing = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+".training.model")
            else:
                model_training = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".training.model")
                model_testing = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".testing.model")
                #model_testing = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".training.model")
                
            for nl in training_data:
                #nl="ARA"
                all_texts=training_data[nl]
                
                for identifier in all_texts:
                    document=[]
                    surviving_words=[]
                    for word in re.findall(r'\w+',all_texts[identifier]):
                        #print(word)
                        try:
                            document.append(np.array(model_training.wv[word]))
                        except KeyError:
                            continue
                        surviving_words.append(word)
                    document=np.array(document)
                    if perform_mean:
                        # Mean of columns
                        data_mean_training[nl].append(document.mean(axis=0))
                    if perform_sum:
                        # Sum of columns
                        data_sum_training[nl].append(document.sum(axis=0))
                    if perform_std:
                        # Sum of columns
                        row=[]
                        row.extend(document.mean(axis=0))
                        row.extend(document.std(axis=0))
                        data_std_training[nl].append(row)
                    if perform_tfidf:
                        vec_tfidf=tfidf_training[dictionary_training.doc2bow(surviving_words)]
                        #print(vec_tfidf,len(vec_tfidf))
                        weights=[]
                        for word in surviving_words:
                            #print(word,dictionary.token2id[word])
                            try:
                                termid=dictionary_training.token2id[word]
                                idf=tfidf_training.idfs.get(termid)
                                
                                for pairs in vec_tfidf:
                                    if pairs[0]==termid:
                                        #print(pairs[1])
                                        weights.append(pairs[1])
                                        break
                            except KeyError:
                                weights.append(0)
                        data_tfidf_training[nl].append(np.average(document,axis=0,weights=weights))
            
            for nl in testing_data:
                #nl="ARA"
                all_texts=testing_data[nl]
                
                for identifier in all_texts:
                    """print(all_texts[identifier])
                    print("="*50)"""
                    document=[]
                    surviving_words=[]
                    for word in re.findall(r'\w+',all_texts[identifier]):
                        #print(word)
                        try:
                            document.append(np.array(model_testing.wv[word]))
                        except KeyError:
                            continue
                        surviving_words.append(word)
                    document=np.array(document)
                    if perform_mean:
                        # Mean of columns
                        data_mean_testing[nl].append(document.mean(axis=0))
                    if perform_sum:
                        # Sum of columns
                        data_sum_testing[nl].append(document.sum(axis=0))
                    if perform_std:
                        # Sum of columns
                        row=[]
                        row.extend(document.mean(axis=0))
                        row.extend(document.std(axis=0))
                        data_std_testing[nl].append(row)
                    if perform_tfidf:
                        vec_tfidf=tfidf_testing[dictionary_testing.doc2bow(surviving_words)]
                        #print(vec_tfidf,len(vec_tfidf))
                        weights=[]
                        for word in surviving_words:
                            #print(word,dictionary.token2id[word])
                            try:
                                termid=dictionary_testing.token2id[word]
                                idf=tfidf_testing.idfs.get(termid)
                                
                                for pairs in vec_tfidf:
                                    if pairs[0]==termid:
                                        #print(pairs[1])
                                        weights.append(pairs[1])
                                        break
                            except KeyError:
                                weights.append(0)
                                pass
                        if len(document)-len(weights) > 0:
                            for i in range(len(document)-len(weights)):
                                weights.append(0)
                                print("set to 0")
                        print(len(document),len(weights))
                        data_tfidf_testing[nl].append(np.average(document,axis=0,weights=weights))

                #break
            if msg=="":
                if perform_mean:
                    writeDataToFile("training_data/data_mean_"+str(f)+"_"+str(w)+".csv",data_mean_training)
                    #writeDataToFile("actual_testing_data/data_mean_"+str(f)+"_"+str(w)+".csv",data_mean_testing)
                    writeDataToFile("testing_data/data_mean_"+str(f)+"_"+str(w)+".csv",data_mean_testing)
                if perform_sum:
                    writeDataToFile("training_data/data_sum_"+str(f)+"_"+str(w)+".csv",data_sum_training)
                    #writeDataToFile("actual_testing_data/data_sum_"+str(f)+"_"+str(w)+".csv",data_sum_testing)
                    writeDataToFile("testing_data/data_sum_"+str(f)+"_"+str(w)+".csv",data_sum_testing)
                if perform_std:
                    writeDataToFile("training_data/data_std_"+str(f)+"_"+str(w)+".csv",data_std_training)
                    #writeDataToFile("actual_testing_data/data_std_"+str(f)+"_"+str(w)+".csv",data_std_testing)
                    writeDataToFile("testing_data/data_std_"+str(f)+"_"+str(w)+".csv",data_std_testing)
                if perform_tfidf:
                    #writeDataToFile("training_data/data_tfidf_"+str(f)+"_"+str(w)+".csv",data_tfidf_training)
                    writeDataToFile("actual_testing_data/data_tfidf_"+str(f)+"_"+str(w)+".csv",data_tfidf_testing)
            else:
                if perform_mean:
                    writeDataToFile("training_data/data_mean_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_mean_training)
                    writeDataToFile("testing_data/data_mean_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_mean_testing)
                if perform_sum:
                    writeDataToFile("training_data/data_sum_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_sum_training)
                    writeDataToFile("testing_data/data_sum_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_sum_testing)
                if perform_tfidf:
                    writeDataToFile("training_data/data_tfidf_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_tfidf_training)
                    writeDataToFile("testing_data/data_tfidf_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_tfidf_testing)

def performTrainingAndTesting(num_of_features,choices_of_window,option):
    
    parallelize=30
    counter=0
    for f_num,f in enumerate(num_of_features):
        for w_num,w in enumerate(choices_of_window):
            training_data_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            training_data_filename_scaled="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            parameters_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.parameters"
            
            testing_data_filename="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            testing_data_filename_scaled="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            # SVM Scaling of training data
            cmd="svm-scale -l 0 -u 1 -s "+parameters_filename+" "+training_data_filename+" > "+training_data_filename_scaled
            if counter!=parallelize:
                cmd+=" & "
            else:
                counter=0
            #print(cmd)
            counter+=1
    #return
    counter=0
    for f_num,f in enumerate(num_of_features):
        for w_num,w in enumerate(choices_of_window):
            training_data_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            training_data_filename_scaled="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            parameters_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.parameters"
            
            testing_data_filename="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            testing_data_filename_scaled="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            # SVM Scaling of tetsing data
            cmd="svm-scale -l 0 -u 1 -r "+parameters_filename+" "+testing_data_filename+" > "+testing_data_filename_scaled
            if counter!=parallelize:
                cmd+=" & "
            else:
                counter=0
            #print(cmd)
            counter+=1
    #return
    counter=0
    for f in num_of_features:
        for w in choices_of_window:
            
            training_data_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            training_data_filename_scaled="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            parameters_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.parameters"
            
            testing_data_filename="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            testing_data_filename_scaled="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            
            choices_of_gamma=[1,0.1,0.01,0.001,0.0001,0.000001]
            choices_of_r=[0,1,10,50,100,1000]
            choices_of_c=[1,10,100,1000]
            
            
            for g_num,g in enumerate(choices_of_gamma):
                for r_num,r in enumerate(choices_of_r):
                    for c_num,c in enumerate(choices_of_c):
                        model_filename="models/data_"+option+"_"+str(f)+"_"+str(w)+"_"+str(g)+"_"+str(c)+"_"+str(r)+".model"
                        output_filename_training_data="output/training_data/results_"+option+"_"+str(f)+"_"+str(w)+"_"+str(g)+"_"+str(c)+"_"+str(r)+".txt"
                        output_filename_development_data="output/development_data/results_"+option+"_"+str(f)+"_"+str(w)+"_"+str(g)+"_"+str(c)+"_"+str(r)+".txt"
                        # Train system
                        cmd="svm-train -m 5000 -q "
                        cmd+=" -g "+str(g)
                        cmd+=" -c "+str(c)
                        cmd+=" -r "+str(r)
                        cmd+=" "+training_data_filename_scaled+" "+model_filename
                        if counter!=parallelize:
                            cmd+=" & "
                        else:
                            counter=0
                        #print(cmd)
                        counter+=1
                        
    for f in num_of_features:
        for w in choices_of_window:
            
            training_data_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            training_data_filename_scaled="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            parameters_filename="training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.parameters"
            
            testing_data_filename="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv"
            testing_data_filename_scaled="testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv.scaled"
            
            choices_of_gamma=[1,0.1,0.01,0.001,0.0001,0.000001]
            choices_of_r=[0,1,10,50,100,1000]
            choices_of_c=[1,10,100,1000]
            
            
            for g_num,g in enumerate(choices_of_gamma):
                for r_num,r in enumerate(choices_of_r):
                    for c_num,c in enumerate(choices_of_c):                
                        model_filename="models/data_"+option+"_"+str(f)+"_"+str(w)+"_"+str(g)+"_"+str(c)+"_"+str(r)+".model"
                        output_filename_training_data="output/training_data/results_"+option+"_"+str(f)+"_"+str(w)+"_"+str(g)+"_"+str(c)+"_"+str(r)+".txt"
                        output_filename_development_data="output/development_data/results_"+option+"_"+str(f)+"_"+str(w)+"_"+str(g)+"_"+str(c)+"_"+str(r)+".txt"
                        # Test on training data
                        cmd="svm-predict "+training_data_filename_scaled+" "+model_filename+" "+output_filename_training_data+" > "+output_filename_training_data+".quickview &"
                        print(cmd)
                        # Test on testing data
                        cmd="svm-predict "+testing_data_filename_scaled+" "+model_filename+" "+output_filename_development_data+" > "+output_filename_development_data+".quickview "
                        print(cmd)
            
        
def main():
    
    training_data=readFromFile("../data/megaTrain.csv")
    testing_data=readFromFile("../data/megaDev.csv")
                
    training_data=reArrangeByNativeLanguage(training_data)
    testing_data=reArrangeByNativeLanguage(testing_data)
    """
    Please uncomment this section if you wish to tag nouns and numbers.
    """
    """training_data=tagNNPAndNumbers(training_data)
    testing_data=tagNNPAndNumbers(testing_data)"""
    
    num_of_features=[300,400,500,600,700,800,900,1000]
    choices_of_window=[5,9,11,13,17]
    
    # This function will generate the training and the testing data set
    calculateWordToVector(training_data,testing_data,num_of_features,choices_of_window,msg="",recalculate=False,perform_tfidf=False)
    
    # This function will print the commands that should be executed to train and test the data
    performTrainingAndTesting(num_of_features,choices_of_window,"mean")
    performTrainingAndTesting(num_of_features,choices_of_window,"sum")
    performTrainingAndTesting(num_of_features,choices_of_window,"std")

if __name__ == "__main__":
    main()
