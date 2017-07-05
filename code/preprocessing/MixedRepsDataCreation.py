#Mixed N-gram representations dataset creation.

import nltk
from gensim import utils
from sklearn.feature_extraction.text import CountVectorizer

#replace Nouns and NNPs, adjectives, but keep the rest and do ngram modeling
'''
Noun tags: NN, NNP, NNS, NNPS (replace with nountag)
Pronoun tags: PRP, PRP$ (replace with prontag)
LS (replace with lstag)
CD (replace with cdtag)
SYM (replace with symtag)
'''

def processText(text_string):
    tokenized_text_string = nltk.tokenize.word_tokenize(text_string)
    tagged_string = nltk.pos_tag(tokenized_text_string)
    result_string = []
    for word_tag in tagged_string:
        word = word_tag[0]
        tag = word_tag[1]
        #Depending on what you want to keep - use some of these if statements and comment others.
        '''
        if tag.startswith('NN'):
            result_string.append("othertag")
        elif tag.startswith('PRP'):
            result_string.append("othertag")
        elif tag.startswith('VB'):
            result_string.append("othertag")
        elif tag.startswith('LS') or tag.startswith('SYM') or tag.startswith('CD'):
            result_string.append("othertag")
        elif tag.startswith(","):
            result_string.append("comma")
        elif tag.startswith('?') or tag.startswith('!') or tag.startswith(':') or tag.startswith(';'):
            result_string.append("puncttag")
        elif tag.startswith('"') or tag.startswith("'"):
            result_string.append("othertag")
        elif tag.startswith('JJ') or tag.startswith('RB'):
            result_string.append("othertag")
        else:
            result_string.append(word)
        '''
        if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ') or tag.startswith('CD'):
            result_string.append(tag)
        else:
            result_string.append(word)

    return " ".join(result_string)

def returnPOSTags(text_string):
    tagged_string = nltk.pos_tag(nltk.tokenize.word_tokenize(text_string))
    tags = []
    for tuple in tagged_string:
        if tuple[1] == ",":
            tags.append("COMMA")

        else:
            tags.append(tuple[1])
    return " ".join(tags)

def POSTaggedRep(inputcsvfilepath, outputcsvfilepath):
    fh = open(inputcsvfilepath)
    header = fh.readline()
    fw = open(outputcsvfilepath, "w")
    fw.write(header)

    for line in fh:
        text_string = utils.to_unicode(line).strip().lower().split(',"')[1][0:-1]
        label = utils.to_unicode(line).strip().split(',"')[0].split(",")[2]
        fileid = utils.to_unicode(line).strip().split(',"')[0].split(",")[0]
        prompt = utils.to_unicode(line).strip().split(',"')[0].split(",")[1]
        processedString = returnPOSTags(text_string)
        towrite = fileid+","+prompt+","+label+',"'+processedString+'"' + "\n"
        fw.write(towrite)
    fw.close()
    fh.close()

def createMixedFile(inputcsvfilepath, outputcsvfilepath):
    fh = open(inputcsvfilepath)
    header = fh.readline()
    fw = open(outputcsvfilepath, "w")
    fw.write(header)

    for line in fh:
        # Reading each sentence from file
        text_string = utils.to_unicode(line).strip().lower().split(',"')[1][0:-1]
        label = utils.to_unicode(line).strip().split(',"')[0].split(",")[2]
        fileid = utils.to_unicode(line).strip().split(',"')[0].split(",")[0]
        prompt = utils.to_unicode(line).strip().split(',"')[0].split(",")[1]
        processedString = processText(text_string)
        towrite = fileid+","+prompt+","+label+',"'+processedString+'"' + "\n"
        fw.write(towrite)

    fw.close()
    fh.close()

createMixedFile("../data/megaTrain.csv", "../data/megaTrain-MixedWordTagRep5.csv")
createMixedFile("../data/megaDev.csv", "../data/megaDev-MixedWordTagRep5.csv")

#POSTaggedRep("../data/megaTrain.csv", "../data/megaTrain-POS.csv")
#POSTaggedRep("../data/megaDev.csv", "../data/megaDev-POS.csv")
