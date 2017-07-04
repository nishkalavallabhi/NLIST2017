
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

def readFromFile(filename):
    """
    This function will read from the given file. It assumes that the file is in .csv format where each row
    represents the following: the identifier of the data, the prompt, the native language, the text.
    This function will return a dictionary which will have 11 entries, one for each of the 11 native languages probed
    in this competition. Each of the entries will further have 8 entries, each pertaining to the different prompts. 
    Under each entry the text will be present identified by the name of the file. An example:
    {
        "ARA":
            {
                "P2": {"xxx1.txt":"text1","xxx2.txt":"text2"}
                "P3":
                "P4":
                .
                .
                .
                "P8":
            }
        "CHI":
        .
        .
        .
        "GER":
    }
    """
    fhr=open(filename,"rt",encoding='utf-8')
    data={}
    for line in fhr:
        line=str(line)
        if "Prompt" in line and "Filename" in line and "Text" in line and "NL" in line:
            continue
        """print(line)
        print(len(line.split(",",3)))"""
        identifier,prompt,nl,text=line.split(",",3)
        text=text.strip()
        text=text.strip('"')
        if nl not in data:
            data[nl]={}
        if prompt not in data[nl]:
            data[nl][prompt]={}
            data[nl][prompt][identifier]=text
        else:
            data[nl][prompt][identifier]=text
    """for nl in data:
        for prompt in data[nl]:
            print(nl,prompt,len(data[nl][prompt]))"""
    return data 

def reArrangeByNativeLanguage(input):
    """
    This function will read in the data dictionary created by the function readFromFile and will aggregate the data based on the Native 
    Languages. 
    """
    data=dict()
    for nl in input:
        
        data[nl]={}
        for prompt in input[nl]:
            for identifier in input[nl][prompt]:
                data[nl][identifier]=input[nl][prompt][identifier]
        print(nl,len(data[nl]))
    return data

def calculateStringsToVector(data):
    """
    """
    for nl in data:
        all_texts=data[nl]
        #print(texts)
        
        # remove common words and tokenize
        stoplist = set('for a of the and to in'.split())
        texts = [[word for word in document.lower().split() if word not in stoplist] for document in all_texts]
        frequency = defaultdict(int)
        
        # remove words that appear only once
        for text in texts:
            for token in text:
                frequency[token] += 1
        
        texts = [[token for token in text if frequency[token] > 1] for text in texts]
        #pprint(texts)
        
        dictionary = corpora.Dictionary(texts)
        dictionary.save('temp/'+nl+'.dict')  # store the dictionary, for future reference
        """print(dictionary)
        print(dictionary.token2id)"""
        for each_text in all_texts:
            new_vec=dictionary.doc2bow(each_text.lower().split())
            print(len(new_vec))

def prepareWord2VecModels(data,num_of_features,choices_of_window,msg):
    entire_text=[]
    for nl in data:
        entire_text.extend([text.split() for text in data[nl].values()])
    
    #print(len(entire_text))
    #pprint(entire_text)
    corpus_models={}
    for f in num_of_features:
        for w in choices_of_window:
            corpus_models[str(f)+"_"+str(w)]=models.Word2Vec(entire_text, size=f, window=w, min_count=5, workers=30,iter=25)
            print("Corpus Model Training over for ",str(f),"features and ",str(w),"window")
            #fs, temp_path = mkstemp("corpus_word2vec_models/"+str(f)+"_"+str(w)+".model")  # creates a temp file
            
            if msg=="":
                corpus_models[str(f)+"_"+str(w)].save("corpus_word2vec_models/"+str(f)+"_"+str(w)+".model")  # save the model
            else:
                corpus_models[str(f)+"_"+str(w)].save("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".model")  # save the model

def writeDataToFile(filename,data):
    fhw=open(filename,"w")
    for nl_num,nl in enumerate(data):
        for each_doc in data[nl]:
            fhw.write(str(nl_to_index[nl])+",")
            for val in each_doc:
                fhw.write(str(val)+",")
            fhw.write("\n")

def calculateWordToVector(data,num_of_features,choices_of_window,msg="",train_or_test="",recalculate=True,perform_mean=True,perform_sum=True,perform_tfidf=True):
    
    """num_of_features=[100,150,200,250,300,350,400,450,500,600,700,800]
    choices_of_window=[5,7,9,11,15]"""
    """num_of_features=[100]
    choices_of_window=[5]"""
    
    if recalculate and train_or_test=="training":
        prepareWord2VecModels(data,num_of_features,choices_of_window,msg)
            
    for f in num_of_features:
        for w in choices_of_window: 
            
            if perform_mean:
                data_mean={}
                for nl in data:
                    data_mean[nl]=[]
            if perform_sum:
                data_sum={}
                for nl in data:
                    data_sum[nl]=[]
                    
            if perform_tfidf:
                data_tfidf={}
                for nl in data:
                    data_tfidf[nl]=[]
                entire_text=[]
                for nl in data:
                    #print(data[nl].values())
                    entire_text.extend([text.split() for text in data[nl].values()])
                docs = [[token for token in text] for text in entire_text]
                """print(docs[0])
                print(entire_text[0])"""
                dictionary = corpora.Dictionary(entire_text)
                #print(dictionary)
                #dictionary.save_as_text("temp/dictionary")
                
                corpus = [dictionary.doc2bow(text) for text in docs]
                #print(docs[0])
                tfidf = models.TfidfModel(corpus)
                #print(tfidf)
                #return
            
            #fs, temp_path = mkstemp("corpus_word2vec_models/"+str(f)+"_"+str(w)+".model")  
            if msg=="":
                model = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+".model")
            else:
                model = gensim.models.Word2Vec.load("corpus_word2vec_models/"+str(f)+"_"+str(w)+"_"+msg+".model")
                
            for nl in data:
                #nl="ARA"
                all_texts=data[nl]
                
                for identifier in all_texts:
                    """print(all_texts[identifier])
                    print("="*50)"""
                    document=[]
                    surviving_words=[]
                    for word in re.findall(r'\w+',all_texts[identifier]):
                        #print(word)
                        try:
                            document.append(np.array(model.wv[word]))
                        except KeyError:
                            continue
                        surviving_words.append(word)
                    document=np.array(document)
                    if perform_mean:
                        # Mean of columns
                        data_mean[nl].append(document.mean(axis=0))
                    if perform_sum:
                        # Sum of columns
                        data_sum[nl].append(document.sum(axis=0))
                    if perform_tfidf:
                        vec_tfidf=tfidf[dictionary.doc2bow(surviving_words)]
                        #print(vec_tfidf,len(vec_tfidf))
                        weights=[]
                        for word in surviving_words:
                            #print(word,dictionary.token2id[word])
                            try:
                                termid=dictionary.token2id[word]
                                idf=tfidf.idfs.get(termid)
                                
                                for pairs in vec_tfidf:
                                    if pairs[0]==termid:
                                        #print(pairs[1])
                                        weights.append(pairs[1])
                                        break
                            except KeyError:
                                weights.append(0)
                        data_tfidf[nl].append(np.average(document,axis=0,weights=weights))
                        
                
                #break
            if msg=="":
                if train_or_test=="training":
                    writeDataToFile("training_data/data_mean_"+str(f)+"_"+str(w)+".csv",data_mean)
                    writeDataToFile("training_data/data_sum_"+str(f)+"_"+str(w)+".csv",data_sum)
                    writeDataToFile("training_data/data_tfidf_"+str(f)+"_"+str(w)+".csv",data_tfidf)
                elif train_or_test=="testing":
                    writeDataToFile("testing_data/data_mean_"+str(f)+"_"+str(w)+".csv",data_mean)
                    writeDataToFile("testing_data/data_sum_"+str(f)+"_"+str(w)+".csv",data_sum)
                    writeDataToFile("testing_data/data_tfidf_"+str(f)+"_"+str(w)+".csv",data_tfidf)
            else:
                if train_or_test=="training":
                    writeDataToFile("training_data/data_mean_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_mean)
                    writeDataToFile("training_data/data_sum_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_sum)
                    writeDataToFile("training_data/data_tfidf_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_tfidf)
                elif train_or_test=="testing":
                    writeDataToFile("testing_data/data_mean_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_mean)
                    writeDataToFile("testing_data/data_sum_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_sum)
                    writeDataToFile("testing_data/data_tfidf_"+str(f)+"_"+str(w)+"_"+msg+".csv",data_tfidf)

def taggifyDocuments(documents,tokens_only=False):
    """list_of_classes=list(sorted(nl_to_index.values()))
    list_of_classes=[item for item in list_of_classes for i in range(1000)][:-1]"""
    for i, line in enumerate(documents):
        if tokens_only:
            yield gensim.models.doc2vec.TaggedDocument(line, [i])
        else:
            # For training data, add tags
            """print("line-->",line)
            print(bytes(line,encoding="iso-8859-1"))"""
            yield gensim.models.doc2vec.TaggedDocument(line, [i])
            
            
def calculateDocumentToVector(data,testing_data,num_of_features,choices_of_window,msg="",train_or_test=""):
    """
    """
    
    languages=["ARA","CHI","FRE","GER","HIN","ITA","JPN","KOR","SPA","TEL","TUR"]
    all_texts=[]
    all_texts_train_and_test=[]
    only_test=[]
    for nl in languages:
        #nl="ARA"
        all_texts.extend(data[nl].values())
        all_texts_train_and_test.extend(data[nl].values())
        only_test.extend(testing_data[nl].values())
    
    for nl in languages:
        all_texts_train_and_test.extend(testing_data[nl].values())
    #print(all_texts)
    #all_texts_combined=[text.split() for text in all_texts]
    list_of_classes_training=list(sorted(nl_to_index.values()))
    list_of_classes_training=[item for item in list_of_classes_training for i in range(1000)][:-1]
    train_corpus=list(taggifyDocuments(all_texts))
    
    list_of_classes_testing=list(sorted(nl_to_index.values()))
    list_of_classes_testing=[item for item in list_of_classes_testing for i in range(100)]
    "test_corpus=list(taggifyDocuments(only_test,tokens_only=True))"
    
    
    list_of_classes=list_of_classes_training+list_of_classes_testing
    train_and_test_corpus=list(taggifyDocuments(all_texts_train_and_test))
    #only_test=list(taggifyDocuments(only_test))
    #both_train_and_test_corpus=list(taggifyDocuments(all_texts_train_and_test))
    #print(train_corpus[:2])
    for f in num_of_features:
        for w in choices_of_window:
            """list_of_classes=list(sorted(nl_to_index.values()))
            list_of_classes=[item for item in list_of_classes for i in range(1000)][:-1]"""
            print("Generating doc2vec models for features",f,"and window",w)
            """model = gensim.models.doc2vec.Doc2Vec(size=f, min_count=5, iter=50,workers = 30,window = w)
            testing_model = gensim.models.doc2vec.Doc2Vec(size=f, min_count=5, iter=50,workers = 30,window = w)"""
            
            model = gensim.models.doc2vec.Doc2Vec(size=f, min_count=5, iter=50,workers = 32,window = w)
            #testing_model = gensim.models.doc2vec.Doc2Vec(size=f, min_count=5, iter=50,workers = 32,window = w)
            
            
            model.build_vocab(train_and_test_corpus)
            model.train(train_and_test_corpus)
            
            
            doc2_vec_training=[]
            doc2_vec_testing=[]
            for doc_num,m in enumerate(model.docvecs):
                doc2_vec_training.append(m)
            doc2_vec_testing=doc2_vec_training[-1100:]
            doc2_vec_training=doc2_vec_training[:-1100]
            #print("Corpus count",model.corpus_count)
            """it = LabeledLineSentence(all_texts,list_of_classes)
            model.build_vocab(it)
            for epoch in range(10):
                model.train(it)
                model.alpha -= 0.002 # decrease the learning rate
                model.min_alpha = model.alpha # fix the learning rate, no deca
                model.train(it)"""
            
            """testing_model.build_vocab(test_corpus)
            testing_model.train(test_corpus, total_examples=testing_model.corpus_count)"""
            """testing_model.build_vocab(both_train_and_test_corpus)
            testing_model.train(both_train_and_test_corpus,total_examples=testing_model.corpus_count)"""
            
            #print("Length of testing data ",len(testing_model.docvecs))
            
            """list_of_classes=list(sorted(nl_to_index.values()))
            list_of_classes=[item for item in list_of_classes for i in range(1000)][:-1]"""
            """print(list_of_classes)
            print(len(list_of_classes))"""
            #return
            
            
            fhw=open("training_data/data_doc2vec_"+str(f)+"_"+str(w)+".csv","w")
            for doc_num,m in enumerate(doc2_vec_training):
                fhw.write(str(list_of_classes_training[doc_num])+",")
                for val in m:
                    fhw.write(str(val)+",")
                fhw.write("\n")
            fhw.close()
            
            """list_of_classes=list(sorted(nl_to_index.values()))
            list_of_classes=[item for item in list_of_classes for i in range(100)]"""
            
            
            """temp_test=[]
            for doc_num,m in enumerate(testing_model.docvecs):
                temp_test.append(m)"""
            
            fhw=open("testing_data/data_doc2vec_"+str(f)+"_"+str(w)+".csv","w")
            for doc_num,m in enumerate(doc2_vec_testing):
                fhw.write(str(list_of_classes_testing[doc_num])+",")
                for val in m:
                    fhw.write(str(val)+",")
                fhw.write("\n")
            fhw.close()

def readTrainingData(filename):
    fhr=open(filename,"r")
    X=[]
    Y=[]
    for line in fhr:
        line=line.strip().split(",")
        Y.append(int(line[0]))
        #try:
        #print(line[1:])
        """for val in line[1:-1]:
            if "e" in val:
                print(val,float(val))
                print(type(val),type(float(val)))"""
        #print([float(val) for val in line[1:-1]])
        X.append([float(val) for val in line[1:-1]])
    return X,Y
    
def performCrossValidationAndTrainingSVM(options,num_of_features,choices_of_window,onlyTrain=False,onlyTest=False,msg=""):
    #num_of_features=[100,150,200,250,300,350,400,450,500,600,700,800]
    #choices_of_window=[5,7,9,11,15]
    # Set the parameters by cross-validation
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9],'C': [1, 10, 100, 500, 1000]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 500, 1000]},
                        ]
    #tuned_parameters=[{'kernel': ['rbf'], 'gamma': [1e-1],'C': [10]}  ]
    min_max_scaler=preprocessing.MinMaxScaler()
    for f in num_of_features:
        for w in choices_of_window:
            for option in options:
                if onlyTrain==True:
                    print("# Tuning hyper-parameters for ","Number of features==>",str(f),"Window Size==>",str(w),"Option==>",str(option))
                    print()
                    if msg=="":
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                    else:
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".csv")
                    #X_test,Y_test=readTrainingData("testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                    """print(len(X),len(Y))
                    print(len(X_test),len(Y_test))
                    del X,Y,X_test,Y_test
                    continue"""
                    
                    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,n_jobs=30)
                    #scaler=preprocessing.StandardScaler().fit(X)
                    #X=scaler.transform(X)
                    scaler = min_max_scaler.fit(X)
                    X=scaler.transform(X)
                    #X_test = scaler.transform(X_test)
                    # min_max_scaler.transform(X_test)
                    clf.fit(X, Y)
                
                    print("Best parameters set found on development set:")
                    print()
                    print(clf.best_params_)
                    print()
                    print("Grid scores on development set:")
                    print()
                    means = clf.cv_results_['mean_test_score']
                    stds = clf.cv_results_['std_test_score']
                    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                        print("%0.3f (+/-%0.03f) for %r"
                              % (mean, std * 2, params))
                    if msg=="":
                        pickle.dump(clf,open("trained_models/"+option+"_"+str(f)+"_"+str(w)+".svm.model","wb"))
                    else:
                        pickle.dump(clf,open("trained_models/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".svm.model","wb"))
                    
                
                if onlyTest==True:
                    print("Number of features==>",str(f),"Window Size==>",str(w),"Option==>",str(option))
                    print()
                    if msg=="":
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                        X_test,Y_test=readTrainingData("testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                    else:
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".csv")
                        X_test,Y_test=readTrainingData("testing_data/data_"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".csv")
                    #scaler=preprocessing.StandardScaler().fit(X)
                    scaler=min_max_scaler.fit(X)
                    X = scaler.transform(X)
                    X_test = scaler.transform(X_test)
                    print("Detailed classification report:")
                    print()
                    print("The model is trained on the full training set.")
                    print("The scores are computed on the full training set.")
                    print()
                    if msg=="":
                        clf=pickle.load(open("trained_models/"+option+"_"+str(f)+"_"+str(w)+".svm.model","rb"))
                    else:
                        clf=pickle.load(open("trained_models/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".svm.model","rb"))
                    y_true, y_pred = Y, clf.predict(X)
                    print(classification_report(y_true, y_pred))
                    #y_true, y_pred = Y_test, clf.predict_proba(X)
                    if msg=="":
                        pickle.dump(Y,open("output/training_data/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".output","wb"))
                    else:
                        pickle.dump(Y,open("output/training_data/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".output","wb"))
                    
                    print("The model is trained on the full training set.")
                    print("The scores are computed on the full development set.")
                    print()
                    #clf=pickle.load(open("trained_models/"+option+"_"+str(f)+"_"+str(w)+".svm.model","rb"))
                    y_true, y_pred = Y_test, clf.predict(X_test)
                    print(classification_report(y_true, y_pred))
                    
                    #y_true, y_pred = Y_test, clf.predict_proba(X_test)
                    if msg=="":
                        pickle.dump(Y_test,open("output/development_data/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".output","wb"))
                    else:
                        pickle.dump(Y_test,open("output/development_data/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".output","wb"))
                    print()
                print("="*150)
                
                #return
def performCrossValidationAndTrainingRF(options,num_of_features,choices_of_window,onlyTrain=True,onlyTest=True,msg=""):
    
    tuned_parameters = [
                        {'n_estimators':[500,750,1000,1500]}
                        ]
    min_max_scaler=preprocessing.MinMaxScaler()
    for f in num_of_features:
        for w in choices_of_window:
            for option in options:
                if onlyTrain==True:
                    print("# Tuning hyper-parameters for ","Number of features==>",str(f),"Window Size==>",str(w),"Option==>",str(option))
                    print()
                    if msg=="":
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                    else:
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".csv")
                    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=4,n_jobs=30)
                    scaler = min_max_scaler.fit(X)
                    X=scaler.transform(X)
                    
                    clf.fit(X, Y)
                
                    print("Best parameters set found on development set:")
                    print()
                    print(clf.best_params_)
                    print()
                    print("Grid scores on development set:")
                    print()
                    means = clf.cv_results_['mean_test_score']
                    stds = clf.cv_results_['std_test_score']
                    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                        print("%0.3f (+/-%0.03f) for %r"
                              % (mean, std * 2, params))
                    if msg=="":
                        pickle.dump(clf,open("trained_models/"+option+"_"+str(f)+"_"+str(w)+".rf.model","wb"))
                    else:
                        pickle.dump(clf,open("trained_models/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".rf.model","wb"))
                    
                if onlyTest==True:
                    print("Number of features==>",str(f),"Window Size==>",str(w),"Option==>",str(option))
                    print()
                    if msg=="":
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                        X_test,Y_test=readTrainingData("testing_data/data_"+option+"_"+str(f)+"_"+str(w)+".csv")
                    else:
                        X,Y=readTrainingData("training_data/data_"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".csv")
                        X_test,Y_test=readTrainingData("testing_data/data_"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".csv")
                    #scaler=preprocessing.StandardScaler().fit(X)
                    scaler=min_max_scaler.fit(X)
                    X = scaler.transform(X)
                    X_test = scaler.transform(X_test)
                    print("Detailed classification report:")
                    print()
                    print("The model is trained on the full training set.")
                    print("The scores are computed on the full training set.")
                    print()
                    if msg=="":
                        clf=pickle.load(open("trained_models/"+option+"_"+str(f)+"_"+str(w)+".rf.model","rb"))
                    else:
                        clf=pickle.load(open("trained_models/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".rf.model","rb"))
                    y_true, y_pred = Y, clf.predict(X)
                    print(classification_report(y_true, y_pred))
                    #y_true, y_pred = Y_test, clf.predict_proba(X)
                    if msg=="":
                        pickle.dump(Y,open("output/training_data/"+option+"_"+str(f)+"_"+str(w)+".rf.output","wb"))
                    else:
                        pickle.dump(Y,open("output/training_data/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".rf.output","wb"))
                    
                    print("The model is trained on the full training set.")
                    print("The scores are computed on the full development set.")
                    print()
                    #clf=pickle.load(open("trained_models/"+option+"_"+str(f)+"_"+str(w)+".svm.model","rb"))
                    y_true, y_pred = Y_test, clf.predict(X_test)
                    print(classification_report(y_true, y_pred))
                    
                    #y_true, y_pred = Y_test, clf.predict_proba(X_test)
                    if msg=="":
                        pickle.dump(Y_test,open("output/development_data/"+option+"_"+str(f)+"_"+str(w)+".rf.output","wb"))
                    else:
                        pickle.dump(Y_test,open("output/development_data/"+option+"_"+str(f)+"_"+str(w)+"_"+msg+".rf.output","wb"))
                    print()
                print("="*150)
    
def main():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    training_data=readFromFile("data/train2017.csv")
    testing_data=readFromFile("data/dev2017.csv")
    
    #print(testing_data)
    training_data=reArrangeByNativeLanguage(training_data)
    testing_data=reArrangeByNativeLanguage(testing_data)
    #return
    #calculateStringsToVector(training_data)

    
    num_of_features=[100,200,300,400,500,600,700,800]
    #num_of_features=[200]
    
    choices_of_window=[5,7,9,11,15] 
    #choices_of_window=[5]
    calculateDocumentToVector(training_data,testing_data,num_of_features,choices_of_window)   
    performCrossValidationAndTrainingSVM(["doc2vec"],num_of_features,choices_of_window,onlyTrain=True,onlyTest=True,msg="")
    return
    
    """performCrossValidationAndTrainingSVM(["mean","sum","tfidf"],num_of_features,choices_of_window,onlyTrain=True,onlyTest=True,msg="")
    num_of_features=[100]
    choices_of_window=[5,7] """
    #performCrossValidationAndTrainingSVM(["mean","sum","tfidf"],num_of_features,choices_of_window,onlyTest=True)
    """num_of_features=[100]
    choices_of_window=[5]"""
    
    #performCrossValidationAndTrainingRF(["mean","sum","tfidf"],num_of_features[::-1],choices_of_window,onlyTrain=True,onlyTest=True,msg="")
    
    #pass
    
    # Original Data with negative sampling equal to 10
    """num_of_features=[500,600,800,1000]
    choices_of_window=[5,10,15,20]
    calculateWordToVector(training_data,num_of_features,choices_of_window,msg="neg_10",train_or_test="training")
    calculateWordToVector(testing_data,num_of_features,choices_of_window,msg="neg_10",train_or_test="testing")
    performCrossValidationAndTrainingSVM(["mean","sum"],num_of_features[::-1],choices_of_window,onlyTrain=True,onlyTest=True,msg="neg_10")"""
    
    """# Run on spelling corrected data
    training_data=readFromFile("data/train2017_spelling_error_corrected.csv")
    testing_data=readFromFile("data/dev2017_spelling_error_corrected.csv")
    
    #print(testing_data)
    training_data=reArrangeByNativeLanguage(training_data)
    testing_data=reArrangeByNativeLanguage(testing_data)
    num_of_features=[250,500,750,1000]
    choices_of_window=[5,10,15,20]
    calculateWordToVector(training_data,num_of_features,choices_of_window,msg="spell_corrected",train_or_test="training")
    calculateWordToVector(testing_data,num_of_features,choices_of_window,msg="spell_corrected",train_or_test="testing")
    performCrossValidationAndTrainingSVM(["mean","sum","tfidf"],num_of_features[::-1],choices_of_window,onlyTrain=True,onlyTest=True,msg="spell_corrected")
"""



if __name__ == "__main__":
    main()