import nltk
from pycorenlp import StanfordCoreNLP
import csv
import sys
import glob
import os
from nltk.tokenize import sent_tokenize
import glob
import os
from nltk.corpus import wordnet
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize as tokenize

from neuralcoref import Coref
from nltk.parse.stanford import StanfordParser
from nltk.tokenize import RegexpTokenizer

sp_err=[] 
pos_mist = []
vb_mist= []
score_1 = []
score_2 = []
sent_form = 0
final_scores = []
nlp1 = StanfordCoreNLP('http://localhost:9000')
count = 0
coher = []
grade_essay = []
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

c_list = []
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


with HiddenPrints():    
    coref = Coref()
count = 0
list1=['i', 'me', 'you', 'we', 'he', 'myself', 'us', 'yourself','ours','my', 'our','your','it']
list2=[]
f_list1=[]
f_list2=[]
c_list = []
''' Part a, b and c are in this function and the code has been explained using comments '''

def essay_grader(f_name,data,topic):
    wrong=0
    sent_count = 0   
    sentences = sent_tokenize(data)
    result = ""

    for sentence in sentences:
        sent_count+=1        
    

    from nltk.tokenize import word_tokenize
    from nltk.tokenize import RegexpTokenizer

    spelling_error = 0

    # Regexptokenizer is used for tokenizing effectively
    tokenizer = RegexpTokenizer('[A-Za-z0-9\']+')

    tk1 = tokenizer.tokenize(data)
    result = ""
    for token in tk1:
        result += "[" + token + "] "


########################### b. Spelling mistakes ##################################
    
    
    from nltk import pos_tag
    
    # Making use of two dictionaries using pyenchant to compare spellings
    d_US = enchant.Dict("en_US")
    d_UK = enchant.Dict("en_UK")
    tagged_tokens = pos_tag(tk1)
    result = ""
    spelling_error=0
    serror=[]
    
    # This is done to make sure that the proper noun is not considered as a spelling error
    crosscheck = ['NNP','NNPS']  
    
    # Checking the spelling error for each word in the essay
    for token in tagged_tokens:
        result += '[' + token[0] + '/' + token[1] + ']'
        flag = 0
        for ind, tag_val in enumerate(crosscheck):
            if (token[1] == crosscheck[ind]):
                flag = 1
        if flag != 1:
            val_US = d_US.check(token[0])
            val_UK = d_UK.check(token[0])
            if (val_US == False and val_UK == False):
                serror.append(token[0])
                spelling_error += 1

    
########################## c.(i) Subject verb agreement ########################

    # Here we check for the most common type of mistake which is the mistake of this and these
    gramm_mist = 0
    for k in sentences:
        tokens = word_tokenize(k)
        for i, j in enumerate(tokens):#for a given set of tokens in a given sentence
            if j == 'this':
                list_temp = nltk.tag.pos_tag([tokens[i+1]])
                for tag in list_temp:
                    if(tag[1] == 'NNS'):
                        gramm_mist+=1
                        
            if j == 'these':
                list_temp1 = nltk.tag.pos_tag([tokens[i+1]])
                for tag in list_temp1:
                    if(tag[1] == 'NN'):
                        gramm_mist+=1 
    
    tokens = word_tokenize(data)
    result = ""
    for token in tokens:
        result += "[" + token + "] "
    
    from nltk import pos_tag
    tagged_tokens = pos_tag(tokens)
    result = ""
    
    # In this case we check for subject verb agreement using different pairs of tags
    # We are detecting whether the user has entered comma or not by using pairs of tags that cannot come together without a comma
    # We also found that two determiners cannot be together 
    crosscheck=['NNP VBP','MD VBN', 'DT DT', 'DT VBP', 'DT VB', 'DT PRP', 'MD VBD', 'JJS PRP' ]
    
    previousTag = '.'
    previousWord = ''
    pairs_mtake = 0
    for token in tagged_tokens:
        result += '[' + token[0] + '/' + token[1] + ']'

        previousTag_tag = previousTag + ' ' + token[1]
        previousTag = token[1]
    
        previousWord_word = previousWord + ' ' + token[0]
        previousWord = token[0]
    
        # The bigram pos pairs are checked with the pairs in the crosschecked list
    
        flag=0
        for ind,tag_val in enumerate(crosscheck):
            if(previousTag_tag == crosscheck[ind]):
                flag=1
                pairs_mtake += 1
                
    pos_gramm_mistakes = pairs_mtake + gramm_mist 
    pos_mist.append(pos_gramm_mistakes)
    
    
################# c.(ii) - Detecting missing verbs and tense mistakes ###########


    verb_mist = 0
    
    #individual sentences in the list
    for k in sentences:
        doc = nlp(k)
        str = ""
        #tokenize individual senteneces
        for token in doc:
            str = str + token.pos_ + " "

        if str.find("VERB") == -1:
            verb_mist+=1
            
    # In this case we check tense mistakes and missing verbs by making doing a crosscheck of pairs        
    crosscheck=['NNP VBP', 'NNS VBZ','VBZ NNP','VBP NNP']
    
    previousTag = '.'
    previousWord = ''
    tense_mtake = 0
    
    # Pairs of tokens are checked each time by making use of crosscheck array in order to find mistakes in pairs
    for token in tagged_tokens:
        result += '[' + token[0] + '/' + token[1] + ']'

        previousTag_tag = previousTag + ' ' + token[1]
        previousTag = token[1]
    
        previousWord_word = previousWord + ' ' + token[0]
        previousWord = token[0]
    
    
        flag=0
        for ind,tag_val in enumerate(crosscheck):
            if(previousTag_tag == crosscheck[ind]):
                flag=1
                tense_mtake += 1
                
    verb_tensemist = verb_mist + tense_mtake  
    vb_mist.append(verb_tensemist)       
    
########################## c.(iii) Sentence Formation ######################################

    error_frag = 0
    for k in sentences:
        output = nlp1.annotate(k, properties={
            'annotators': 'tokenize,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
            })
          
        sbar_flag = 0
        s_flag = 0
        if(count<=83):
            for i,p in enumerate([s['parse'] for s in output['sentences']]):#Returns a parse tree for a particular sentence
                index_s = p.find('(S')
                if (p[index_s+2] == '\n' or p[index_s+2] == ' '):
                    s_flag = 1
    
                index_sbar = p.find('SBAR')
                if (p[index_sbar+4] == " " or p[index_sbar+4] == "\n"):
                    sbar_flag = 1
                    
                if "FRAG" in p:
                    if(sbar_flag == 1 and s_flag == 0):  
                        #print(p)
                        error_frag += 1                  

############################ d.(i) Is the essay coherent?       #######################################
    tokenizer = RegexpTokenizer('[A-Za-z0-9\']+')
                
    sentences = sent_tokenize(data)        

    prev_sent=""
    for ind,s in enumerate(sentences):
        
        if(ind!=0):
            tk1 = tokenizer.tokenize(s)
            tagged_tokens = pos_tag(tk1)
            for token in tagged_tokens:
                if(token[1]=='PRP' or token[1]=='PRP$'):   #Looking for pronouns in 3rd person
                    if(token[0].casefold() not in list1 and token[0] not in f_list1):
                        f_list1.append(token[0])   
            
            prev_sent=sentences[ind-1]
            utterances=s       #Current sentence
            context=prev_sent  #Previous sentence for conflict resolution 
            
            clusters = coref.one_shot_coref(utterances, context)
            
            most = coref.get_most_representative()   #Generates links between context and utterance
            most1 = repr(most)
            for x in f_list1:
                if x not in most1:
                    #print("%s\n" %context)
                    #print("%s\n" %utterances)
                    #print("%s\n\n\n" %x)
                    
                    wrong+=1
                    break
            f_list1.clear()
    c_list.append(wrong)
    coref_mist = wrong       
    wrong=0                                    

############################ d.(ii) Does the essay stay on topic       #######################################
    tk1 = tokenize(topic)
    tagged_tokens = pos_tag(tk1)
    new_top = ''
    
    #Here I check for all the noun occurences in the essay
    for token in tagged_tokens:
        if(token[1]=='NNS' or token[1] == 'NN' or token[1] == 'NNP' or token[1] == 'NNPS'):
            new_top = new_top + token[0] + " "
        
    nouns = {x.name().split('.', 1)[0] for x in wordnet.all_synsets('n')}
    
    dic = {}
    #I use wordnet to find the main words in the topic which are used later to find similar words in essay
    for i,k in enumerate(topic.split()):
        synonyms = []
        for syn in wordnet.synsets(k):
            for l in syn.lemmas():
                synonyms.append(l.name())       
        dic.update({i:set(synonyms)})
     
    word_set = set()  
    
    #Now using the synonyms of the words in the topics I find the match in the essay
    for k in data.split(' '):#Each of the words in essay
        for val in dic:
            if (k in dic[val] and k in nouns):
                word_set.add(k)
    
    for i,k in enumerate(new_top.split()):
        synonyms = []
        for syn in wordnet.synsets(k):
            for l in syn.lemmas():
                synonyms.append(l.name())       
        dic.update({i:set(synonyms)})
     
    
    for k in data.split(' '):#Each of the words in essay
        for j in new_top.split(' '): 
            if (k == j):
                word_set.add(k) 
   
    new_set = set()
    for k in word_set:
        if(k != ''):
            new_set.add(lemmatizer.lemmatize(k))
    #print(len(new_set))    
    
    #The length gives the number of words that are related to the topic
    ess_coher = len(new_set)
    coher.append(len(new_set))
                        
                        
################################################################################                        
                        
    scores(f_name,sent_count,spelling_error,serror,pos_gramm_mistakes, verb_tensemist,error_frag,ess_coher,coref_mist)                        

''' Append the scores of the essays based on various score ranges '''

def scores(f_name,count, spelling_error, serror, pos_gramm_mistakes,verb_tensemist,error_frag,ess_coher,coref_mist):
    
    if(1<=count<=4):
        a = 1
    elif(5<=count<=9):
        a = 2
    elif(10<=count<=14):
        a = 3
    elif(15<=count<=19):
        a = 4
    elif(count>=20):
        a = 5
    
    if(0<=spelling_error <= 1):
        b = 4
    elif(spelling_error == 2):
        b = 3
    elif(3<=spelling_error<=4):
        b = 2
    elif(5<=spelling_error<=5):
        b = 1
    elif(spelling_error>=6):
        b = 0
        
    if(pos_gramm_mistakes >= 8):
        c1 = 1
    elif(5<=pos_gramm_mistakes<=7):
        c1 = 2
    elif(3<=pos_gramm_mistakes<=4):
        c1 = 3
    elif(1<=pos_gramm_mistakes<=2):
        c1 = 4
    elif(pos_gramm_mistakes==0):
        c1 = 5    
        
    score_1.append(c1)    
    
    c2 = 0
    if(verb_tensemist>3):
        c2 = 1
    elif(verb_tensemist == 3):
        c2 = 2
    elif(verb_tensemist==1):
        c2 = 4
    elif(verb_tensemist==0):
        c2 = 5
        
        
    c3 = 0
    if(error_frag == 2):
        c3 = 2
    elif(error_frag == 1):
        c3 = 3
    elif(error_frag==0):
        c3 = 4   
        
        
    score_2.append(c2)    
    d1=0
    if(coref_mist >= 8):
        d1 = 1
    elif(5<=coref_mist<=7):
        d1 = 2
    elif(3<=coref_mist<=4):
        d1 = 3
    elif(1<=coref_mist<=2):
        d1 = 4
    elif(coref_mist==0):
        d1 = 5
    
    d2=0
    if(ess_coher >= 8):
        d2 = 5
    elif(5<=ess_coher<=7):
        d2 = 4
    elif(3<=ess_coher<=4):
        d2 = 3
    elif(1<ess_coher<=2):
        d2 = 2
    elif(ess_coher==1):
        d2 = 1
        
    # Calculating the final score based on the given formula
    f_score = 2*a - 2*b + c1 + c2 + 2*c3 + d1 + 3*d2
    final_scores.append(f_score)
    #Some other formulae considered forscoring are:
#    f_score =  2*a - 2*b + 2*c1 + c2
#    f_score =  2*a - 2*b + 2*c1 + 0.5*c2
    
    grade = ''
    
    # Checking the total value of score that is calculated using individual scores by setting a threshold limit
    
    if f_score>38:
        grade ='high'
    else:   
        grade ='low'
        
    grade_essay.append(grade)
    
    f.write("%s;" % f_name)
    f.write("%s;" % a)
    f.write("%s;" % b)
    f.write("%s;" % c1)
    f.write("%s;" % c2)
    f.write("%d;" % c3)
    f.write("%d;" % d1)
    f.write("%d;" % d2)
    f.write("%s;" % f_score)
    f.write("%s\n" % grade)
       


# Download all the required packages
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('treebank')

# Download spacy if not downloaded
import sys
try:
    import enchant
except ImportError:
    print("Installing the required 'enchant' module\n")
    os.system('pip install pyenchant') 
import enchant    
    
try:	
    import spacy	
except ImportError:		
    print("Installing the required 'spacy' module\n")
    os.system('pip install -U spacy')
    os.system('python -m spacy download en')
import spacy
nlp = spacy.load('en')

# Open file and store it in variable f
f = open("../output/results.txt", "w+")

print('\nEssay grader running please wait for atleast 2 minutes...')

#Open each essay and call the essay grader function
print('Current Essay Running:')
for filename in glob.glob(os.path.join('../input/testing/essays/', '*.txt')):
    print(count)
    count +=1
    f_name = (filename.split("../input/testing/essays/"))[1]    
    essay_name = (filename.split("../input/testing/essays/"))[1]    
    mycsv = csv.reader(open('../input/testing/index.csv'), delimiter=';')
    with open(filename, 'r') as myfile:
        for row in mycsv:
            if(essay_name == row[0]):
                topic = (row[1][56:].split('.', 1))[0]
                break
        data = myfile.read()  
        essay_grader(f_name,data,topic)
        
f.close()
print('\nEssay graded, open results.txt in output folder to view the results')

#print(final_scores)
#print('============')
#print(coher)
#print(c_list)
#print(grade_essay)