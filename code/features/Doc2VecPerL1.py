'''
Purpose: 
1. Build one doc2vec model per L1 using training data 
(unsupervised - yet to figure out supervised Doc2Vec in Gensim)
2. Use those models to infer doc vectors for training and dev sets
3. Feature rep: concatenated vector from 11 L1 Doc2vecs (should find out other ways)
3. Train a model with training vectors and use it to predict dev. vectors.
'''

from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

import nltk
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


#Builds a doc2vec model with the chosen parameters
def build_doc2vec_model(file_path, output_path, total_examples):
    # Creating labeled sentences from input.txt
    sentences = TaggedLineDocument(file_path)
    #Parameters are explored using cross validation in the other code. Here, they are directly given
    model = Doc2Vec(alpha=0.1, size=500, window=25, min_count=10, dm=0, dbow_words=1, iter=10)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=total_examples,epochs=10)
    model.save(output_path)

#Use the TaggedDocument class to build a supervised doc2vec model instead of an unsupervised one.
def build_doc2vec_supervised_model(file_path, output_path, labels_path, total_examples):
    print("To Do")

#Creates a data array based on 11 doc2vec matrices, 1 for L1.
#The final data array per text is a concatenated version from 11 matrices in this implementation
#Yet to figure out what is the best way to combine these. 
def create_data_array_mult(path):
    d2vpaths = ['ARA.d2v', 'CHI.d2v', 'FRE.d2v', 'GER.d2v', 'HIN.d2v', 'ITA.d2v', 'JPN.d2v', 'KOR.d2v', 'SPA.d2v', 'TEL.d2v', 'TUR.d2v']
    d2vs = []
    for i in d2vpaths: 
       d2vs.append(Doc2Vec.load(i))
    fh = open(path)
    text_vector = [] 
    for line in fh:
        text = nltk.word_tokenize(utils.to_unicode(line).strip().lower())
        temp_vector = []
        for d2v in d2vs:
            d2v.random.seed(0) 
	    #To ensure consistency in the inferred vector each time it is called for a given model and reduce random elements in prediction
            temp_vector.extend(d2v.infer_vector(text))
        text_vector.append(temp_vector)
    fh.close()
    return text_vector

def create_labels_array(length, labels_path):
    labels = numpy.chararray(length,itemsize=5)
    with utils.smart_open(labels_path) as fin:
       for i, line in enumerate(fin):
          labels[i] = line.strip().lower()
    return labels

#Takes the main file and splits it into 11 L1 based files.
def getL1specificData(category, file_path):
    output_path = category + ".txt"
    fh = open(file_path)
    fw = open(output_path, "w")
    for line in fh:
       l1 = line.split(",")[2]
       if l1 == category:
         text = line.split('"')[1]
         fw.write(text.lower())
         fw.write("\n")
    fw.close()
    fh.close()

#Predict with a given model, for a given data, and store output predictions as a csv file
def predict_with_model(model,data,labels,file_path):
    fw1 = open(file_path,"w")
    fw1.write("test_taker_id,prediction")
    fw1.write("\n")
    predicted = model.predict(data)
    for i in range(0,len(predicted)):
        fw1.write(str(predicted[i]) +"," + str(labels[i]))
        fw1.write("\n")

    print(numpy.mean(predicted == labels,dtype=float))
    fw1.close()

def main():
    cats = ['ARA', 'CHI', 'FRE', 'GER', 'HIN', 'ITA', 'JPN', 'KOR', 'SPA', 'TEL', 'TUR']

    #This is to get language specific files that are used to build custom Doc2Vec models. Used only once to create the files.
    '''
    for cat in cats:
        getL1specificData(cat,"megaTrain.csv")
    print("DONE")
    '''
    #This loop builds 11 D2v models, one for each L1 - you can do it only once if you are only doing the classification part.
    #Build new models by changing the hyperparameters in build_doc2vec_model() function
    
    for cat in cats:
       file_path = cat + ".txt"
       output_path = cat + ".d2v"
       total_examples = 1000
       build_doc2vec_model(file_path, output_path, total_examples)
       print("Finished d2v model for: ", cat)
    

    print("Model training done")


    # Creating training data
    train_arrays = create_data_array_mult('megaTrain-TextOnly.txt')
    train_labels = create_labels_array(len(train_arrays), "megaTrain-LabelOnly.txt")

    print(len(train_arrays))
    print(len(train_labels))

    #Create testing data
    dev_arrays = create_data_array_mult('megaDev-TextOnly.txt')
    dev_labels = create_labels_array(len(dev_arrays), "megaDev-LabelOnly.txt")

    print(len(dev_arrays))
    print(len(dev_labels))

    print("Training, Test data created")

    # -----------------------------------------
    # Training the LogisticRegression classifier
    classifier = LogisticRegression(C=0.01)
    classifier.fit(train_arrays, train_labels)
    print("training done. ")

    print("Testing on dev data")
    # Classifier metrics
    predictedLabels = classifier.predict(dev_arrays)
    print(numpy.mean(predictedLabels == dev_labels,dtype=float))
    print(f1_score(predictedLabels, dev_labels,average="macro"))
    # -----------------------------------------

    '''
    print("testing on train data itself")
    predictedLabels = classifier.predict(train_arrays)
    print(numpy.mean(predictedLabels == train_labels,dtype=float))
    print(f1_score(predictedLabels, train_labels,average="macro"))
    '''
     #saves predictions in some file.
   # predict_with_model(classifier,dev_arrays,dev_labels,"../results/devpreds-dvt2.csv")
   

if __name__ == '__main__':
    main()
