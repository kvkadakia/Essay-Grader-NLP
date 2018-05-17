import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

scores_Y = [3, 3, 4, 3, 2, 4, 3, 2, 3, 4, 3, 4, 1, 2, 4, 3, 5, 2, 5, 5, 3, 5, 4, 2, 4, 1, 5, 2, 3, 5, 3, 4, 4, 2, 2, 2, 2, 4, 4, 5, 2, 4, 4, 1, 5, 5, 4, 2, 5, 3, 1, 3, 4, 4, 1, 4, 5, 5, 5, 3, 3, 4, 5, 2, 5, 2, 5, 2, 2, 2, 3, 2, 4, 5, 2, 2, 2, 4, 5, 5, 4, 3, 2, 4, 3, 5, 5, 5, 4, 2, 2, 5, 5, 5, 4, 4, 2, 4, 4, 5]
length_X = []

        
def regression(scores_Y,length_X): 
    y = scores_Y 
    x = length_X
    xsq= [i ** 2 for i in length_X]
    squared_Y= [i ** 2 for i in scores_Y]
    xy = []
    
    for j,i in enumerate(x):
        xy.append(i * y[j])
    
    a = (sum(y)*sum(xsq))-(sum(x)*sum(xy))
    a = a / ((len(x)*sum(xsq)) - sum(x)**2)
    
    b = (len(x)*sum(xy)) - (sum(x)*sum(y))
    b = (b)/((len(x)*sum(xsq)) - (sum(x)**2))
    
    print('{}{}{}{}{}{}'.format('y',' = ', a ,' + ', b , ' * x '))

import glob
import os
from nltk.tokenize import sent_tokenize

for filename in glob.glob(os.path.join('/home/karan/Essay-Grader-NLP/essays', '*.txt')):
    with open(filename, 'r') as myfile:
        data = myfile.read()  
        sentences = sent_tokenize(data)
        result = ""
        count = 0
        for sentence in sentences:
            count+=1
        length_X.append(count)
        
regression(scores_Y ,length_X)

###############################################################################

spell_err=[]
b = []
for filename in glob.glob(os.path.join('/home/karan/Essay-Grader-NLP/essays', '*.txt')):
    with open(filename, 'r') as myfile:
        data = myfile.read()  
        sentences = sent_tokenize(data)
        result = ""
        count = 0
        from nltk.tokenize import word_tokenize
        import enchant
        d_US = enchant.Dict("en_US")
        d_UK = enchant.Dict("en_UK")
        from nltk.tokenize import RegexpTokenizer
    
        spelling_error = 0
        tokenizer = RegexpTokenizer('[A-Za-z0-9\']+')
    
        tk1 = tokenizer.tokenize(data)
        for token in tk1:
            val_US = d_US.check(token)
            val_UK = d_UK.check(token)
            if (val_US == False and val_UK == False):
                spelling_error += 1        
        spell_err.append(spelling_error)
        if(0<=spelling_error <= 1):
            b.append(4)
        elif(spelling_error == 2):
            b.append(3)
        elif(3<=spelling_error<=4):
            b.append(2)
        elif(5<=spelling_error<=6):
            b.append(1)
        elif(spelling_error>=7):
            b.append(0)
        
regression(b,spell_err)
print(b)        
############################################################################

            
            
