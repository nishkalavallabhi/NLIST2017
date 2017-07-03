
'''
Ngram features based classification experiments for Word and Character based Ngrams.
Right now only using Logistic Regression
Note: Feature reps are different in sklearn and say, using our own functions or using WEKA, LightSide implementations etc.
'''

# NLTK modules
import nltk
from gensim import utils
import pprint
import numpy as np

# BoW feature extractor
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

#Read data from the mega csv files and store them into texts, labels and fileids seperately
def readData(csvfilepath):
    fh = open(csvfilepath)
    fh.readline()
    data = []
    labels = []
    fileids = []
    for line in fh:
        # Reading each sentence from file
        text = utils.to_unicode(line).strip().lower().split(',"')[1][0:-1]
        label = utils.to_unicode(line).strip().split(',"')[0].split(",")[2]
        fileid = utils.to_unicode(line).strip().split(',"')[0].split(",")[0]
        text_string = " ".join(nltk.tokenize.word_tokenize(text))
        data.append(text_string)
        labels.append(label)
        fileids.append(fileid)
    return data, labels, fileids

#Predict with a given model, for a given data, and store output predictions as a csv file
def predict_with_model(model,data,ids,labels,file_path):
    fw1 = open(file_path,"w")
    fw1.write("test_taker_id,prediction")
    fw1.write("\n")
    predicted = text_clf.predict(data)
    for i in range(0,len(predicted)):
        fw1.write(ids[i].replace(".txt","") + "," + str(predicted[i]))
        fw1.write("\n")
    print(np.mean(predicted == labels,dtype=float))
    fw1.close()

print("starting")

#Read the data - change these input paths depending on what you want.
train_data, train_labels, train_ids = readData("../data/megaTrain.csv")
dev_data, dev_labels, dev_ids = readData("../data/megaDev.csv")
test_data, test_labels, test_ids = readData("../data/megaTest.csv")
print("Reading the data done")

#Paths to store output:
dev_predictions_path = "../results/unprocessed/dev-predictions-MixedWordTagRep2.csv"
test_predictions_path = "../results/unprocessed/test-speech-predictions-wordNgrams.csv"

#Extract necessary features
print("preparing features: ")

unigram_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,  stop_words = None, ngram_range=(1,1), min_df=10, max_features = 100000)
bigram_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None,  stop_words = None, ngram_range=(2,2), min_df=10, max_features = 100000)
trigram_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(3,3), min_df=10, max_features = 100000)
fourgram_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(4,4), min_df=10, max_features = 100000)
fivegram_vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(5,5), min_df=10, max_features = 100000)
uni_to_tri_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,3), min_df=10, max_features = 200000)
tri_to_five_vectorizer =  CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(1,8), min_df=10, max_features = 300000)
withCharNgrams = CountVectorizer(analyzer = "char", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(2,10), min_df=10, max_features = 100000)
withCharNgrams2 = CountVectorizer(analyzer = "char_wb", tokenizer = None, preprocessor = None, stop_words = None, ngram_range=(2,10), min_df=10, max_features = 100000)

#Put whatever vectorizers you want.
#vectorizers = [unigram_vectorizer,bigram_vectorizer,trigram_vectorizer,fourgram_vectorizer,fivegram_vectorizer,uni_to_tri_vectorizer,tri_to_five_vectorizer,withCharNgrams,withCharNgrams2]
vectorizers = [uni_to_tri_vectorizer,tri_to_five_vectorizer]

print("all features created")

print("logRegClassifier output, with dev data")

classifier = LogisticRegression(C=0.1)
#classifier = LinearSVC(multi_class='crammer_singer')

#For each group of features, train a model, get predictions for dev and test data and store the output csvs.
for vectorizer in vectorizers:
    text_clf = Pipeline([('vect', vectorizer), ('clf', classifier)])
    text_clf = text_clf.fit(train_data,train_labels)
    print("Done with model building: ", str(vectorizer.get_params()['analyzer']), str(vectorizer.get_params()['ngram_range']))
    print("prediction accuracy for this model on dev data: " )
    predict_with_model(text_clf,dev_data,dev_ids,dev_labels,dev_predictions_path)
    predicted = text_clf.predict(dev_data)
    print(np.mean(predicted == dev_labels,dtype=float))
    print("prediction on test data")
    print("prediction accuracy for this model: " )
    predict_with_model(text_clf,test_data,test_ids,test_labels,test_predictions_path)

    print("************")
#DONE
print("wrote predictions out")


'''
#This is to do a Cross-Validation setup
#Just doing 5 fold CV for now. But we should get results on Dev. set.
logRegClassifier_withSW = LogisticRegression(C=0.1)
scores = cross_val_score(logRegClassifier_withSW, train_data_features_withSW, train_labels, cv=5)
predicted = cross_val_predict(logRegClassifier_withSW, train_data_features_withSW, train_labels, cv=5)

#Printing some numbers:
print("logRegClassifier - with stopwords - with cross validation")
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Overall accuracy {:.3g}%".format(metrics.accuracy_score(train_labels, predicted)*100))
print(classification_report(train_labels, predicted))
#print(confusion_matrix(train_labels, predicted, labels=class_names))
#print("\n")

'''