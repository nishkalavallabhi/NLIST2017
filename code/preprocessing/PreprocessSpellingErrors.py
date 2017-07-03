#Preprocess spelling errors and replace them with "abcdef"
#Lines are of the form: fileid,prompt,nl,text

import enchant
import nltk
from gensim import utils

my_dict =  enchant.Dict("en_US")
error_code = "abcdef"

input_file_path = "../data/megaTrain-Speech.csv"
output_file_path = "../data/megaTrain-Speech-spelled.csv"

fh = open(input_file_path)
header = fh.readline() #The header

fw = open(output_file_path,"w")
fw.write(header)
#fw.write("\n")

for line in fh:
    text = utils.to_unicode(line).strip().lower().split(',"')[1][0:-1]
    nl = utils.to_unicode(line).strip().split(',"')[0].split(",")[2]
    prompt = utils.to_unicode(line).strip().split(',"')[0].split(",")[1]
    fileid = utils.to_unicode(line).strip().split(',"')[0].split(",")[0]
    tokens = nltk.tokenize.word_tokenize(text)
    corrected_tokens = []
    for token in tokens:
        if token.isalpha():
            if my_dict.check(token):
                corrected_tokens.append(token)
            else:
                corrected_tokens.append(error_code)
        else:
            corrected_tokens.append(token)
    corrected_text_string = " ".join(corrected_tokens)
    line_to_write = fileid + "," + prompt + "," + nl + "," + '"' + corrected_text_string.replace('"',"'") + '"'
    fw.write(line_to_write)
    fw.write("\n")

fh.close()
fw.close()
