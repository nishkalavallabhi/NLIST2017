from gensim import utils
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

# random
from random import shuffle

import nltk

# numpy
import numpy

# classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report


def build_doc2vec_model():
    # Creating labeled sentences from training data
    sentences = TaggedLineDocument('../data/megaTrain-TextOnly.txt')
    model = Doc2Vec(alpha=0.1, size=500, window=25, min_count=10, dm=0, dbow_words=1, iter=50)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=10999,epochs=50)
    model.save('../data/nli.d2v')

#Using the unsupervised doc2vec model to even infer training vectors.
def create_data_array(path):
    loaded_model = Doc2Vec.load('../data/nli.d2v')
    text_vector = []
    fh = open(path)
    for line in fh:
        text = nltk.word_tokenize(utils.to_unicode(line).strip().lower())
        loaded_model.random.seed(0) #so that doc2vec model returns same inferred vector consistently for a given text.
        text_vector.append(loaded_model.infer_vector(text))
    fh.close()
    return text_vector

def create_labels_array(length, labels_path):
    labels = numpy.chararray(length,itemsize=5)
    with utils.smart_open(labels_path) as fin:
       for i, line in enumerate(fin):
          labels[i] = line.strip().lower()
    return labels

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

    print("building doc2vec model first")
    build_doc2vec_model()

    print("Model training done")

    # Creating training data
    train_arrays = create_data_array('../data/megaTrain-TextOnly.txt')
    train_labels = create_labels_array(len(train_arrays), "../data/megaTrain-LabelOnly.txt")

    print(len(train_arrays))
    print(len(train_labels))

    #Create testing data
    dev_arrays = create_data_array('../data/megaDev-TextOnly.txt')
    dev_labels = create_labels_array(len(dev_arrays), "../data/megaDev-LabelOnly.txt")

    #test_arrays = create_data_array('../data/megaTest-spelled-TextOnly.txt')
    #test_labels = create_labels_array(len(test_arrays), "../data/megaTest-LabelOnly.txt")

    print(len(dev_arrays))
    print(len(dev_labels))

    print("Training, Test data created")

    # -----------------------------------------
    # Training the LogisticRegression classifier
    classifier = LogisticRegression(C=0.1)
    classifier.fit(train_arrays, train_labels)

    print("training done. ")

    print("Testing on dev data")
    # Classifier metrics
    predictedLabels = classifier.predict(dev_arrays)
    print(numpy.mean(predictedLabels == dev_labels,dtype=float))
    print(f1_score(predictedLabels, dev_labels,average="macro"))
    # -----------------------------------------

   # predict_with_model(classifier,dev_arrays,dev_labels,"../results/devpreds-dvt2.csv")

if __name__ == '__main__':
    main()
