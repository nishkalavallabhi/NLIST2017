
from IdentifyNativeLanguage import readFromFile,reArrangeByNativeLanguage
from gensim.models.wrappers import FastText
from IdentifyNativeLanguage import *
import sys
import math
import numpy as np

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


def generateData(data,f,w,msg,ECO) :
    if "training" in msg:
        fhw_sum=open("training_data/data_sum_"+str(f)+"_"+str(w)+"_ECO_wiki.csv","w")
        fhw_mean=open("training_data/data_mean_"+str(f)+"_"+str(w)+"_ECO_wiki.csv","w")
    else:
        fhw_sum=open("testing_data/data_sum_"+str(f)+"_"+str(w)+"_ECO_wiki.csv","w")
        fhw_mean=open("testing_data/data_mean_"+str(f)+"_"+str(w)+"_ECO_wiki.csv","w")
    for nl in data:
        for identifier in data[nl]:
            text=data[nl][identifier].strip()
            text.replace("."," ")
            text=text.split()
            row=[]
            for i in range(w//2,len(text)-w//2):
                lrow=[]
                flag=0
                for j in range(-w//2,0):
                    if str(j)+"_"+text[i+j] not in ECO:
                        flag=1
                        break
                    lrow.append(np.array(ECO[str(j)+"_"+text[i+j]]))
                rrow=[]
                for j in range(1,w//2+1):
                    if str(j)+"_"+text[i+j] not in ECO or flag==1:
                        flag=1
                        break
                    rrow.append(np.array(ECO[str(j)+"_"+text[i+j]]))
                if flag==1:
                    continue
                lrow=np.array(lrow)
                rrow=np.array(rrow)
                finalrow=[]
                finalrow.extend(list(np.mean(lrow,axis=0)))
                finalrow.extend(list(np.mean(rrow,axis=0)))
                #print(len(finalrow))
                row.append(np.array(finalrow))
            if len(row)>1:
                row=np.array(row)
                fhw_sum.write(str(nl_to_index[nl])+",")
                for val in list(np.sum(row,axis=0)):
                    fhw_sum.write(str(val)+",")
                fhw_sum.write("\n")
                
                fhw_mean.write(str(nl_to_index[nl])+",")
                for val in list(np.mean(row,axis=0)):
                    fhw_mean.write(str(val)+",")
                fhw_mean.write("\n")
    print()

def loadECOEmbeddings(dim,c):
    j=list(range(-c//2,c//2+1))
    j.remove(0)
    all=dict()
    for val in j:
        filename="wiki/dim"+str(dim)+"_c"+str(c)+"/cocoon.mincount~5.dim~"+str(dim)+".window~"+str(val)+".dim_divide~"+str(c)+".embeds"
        fhr=open(filename,"r")
        for line_num,line in enumerate(fhr):
            if line_num==0:continue
            line.split()[-dim//c:]
            index=str(val)+"_"+' '.join(line.split()[:-dim//c])
            #if ">" in index:print(index)
            row=[]
            for x in line.split()[-dim//c:]:
                row.append(float(x))
            all[index]=row
        if len(all)%10000:
            print("Length of dictionary",len(all))
            #return all
        #print(all.keys())
    return all
    
def main():
    """training_data=readFromFile("data/train2017_spelling_error_corrected.csv")
    testing_data=readFromFile("data/dev2017_spelling_error_corrected.csv")"""
    training_data=readFromFile("../data/megaTrain.csv")
    testing_data=readFromFile("../data/megaDev.csv")
    
    training_data=reArrangeByNativeLanguage(training_data)
    testing_data=reArrangeByNativeLanguage(testing_data)
    """
    Please uncomment this section if you wish to tag nouns and numbers.
    """
    """training_data=tagNNPAndNumbers(training_data)
    testing_data=tagNNPAndNumbers(testing_data)"""
    #print(testing_data)
    
    num_of_features=[500,100,700]
    choices_of_window=[4,10]
    
    for f in num_of_features:
        for w in choices_of_window:
            if f==700 and w==10:
                continue
            print("Loading ECO embeddings for dimension",f,"and window ",w)
            ECO=loadECOEmbeddings(f,w)
            print("Generating training data for dimension",f,"and window ",w)
            generateData(training_data,f,w,"training",ECO) 
            print("Generating testing data for dimension",f,"and window ",w)
            generateData(testing_data,f,w,"testing",ECO) 
            #sys.exit()
    """combineTextsIntoSingleFile("data/training_corpus",training_data)
    combineTextsIntoSingleFile("data/development_corpus", testing_data)"""
    performCrossValidationAndTrainingSVM(["mean","sum"],num_of_features,choices_of_window,onlyTrain=True,onlyTest=True,msg="ECO_wiki")
    #performCrossValidationAndTrainingSVM(["mean","sum"],num_of_features,choices_of_window,onlyTrain=False,onlyTest=True,msg="fasttext_cbow")


if __name__ == "__main__":
    main()