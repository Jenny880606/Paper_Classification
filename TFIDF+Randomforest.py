import csv
#import requests
import nltk
import math
import string
from nltk.stem.porter import *
from sklearn.feature_extraction.text import TfidfVectorizer
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from collections import Counter
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import ensemble, metrics


vectorizer = CountVectorizer()
pst = PorterStemmer()


testing = 8000
category = ['theoretical', 'engineering', 'empirical', 'others']

catch = [0.05,0.1]

def TTT(num):
    if num>0:
        return 1
    else:
        return 0

def turn_TASK(T):
    TT = [0,0,0,0]
    num=0
    for t in T:
        for c in range(len(category)):
            if t == category[c]:
                TT[c] = c+1
                num+=1
                break
    return TT,num

def get_tokens(text):
    lowers = text.lower()
    # remove the punctuation using the character deletion step of translate
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    no_punctuation = lowers.translate(remove_punctuation_map)
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tf(word, count):
    return count[word] / sum(count.values())


def n_containing(word, count_list):
    return sum(1 for count in count_list if word in count)
    # return sum([1 for count in count_list if word in count])           #游-解


def idf(word, count_list):
    return math.log(len(count_list) / (1+n_containing(word, count_list)))


def tfidf(word, count, count_list):
    return tf(word, count) * idf(word, count_list)


def add_key(w, key):
    inhere = 0
    for k in key:
        if w == k:
            inhere = 1
            return False
    if inhere == 0:
        return True
    else:
        return False


nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

TITLE_train = []
ABSTRACT_train = []
TASK_train = []
CATE_train=[]

TITLE_test = []
ABSTRACT_test = []
TASK_test = []
CATE_test=[]


combin_train=[]
combin_test=[]

keyword_C=[]

ALL_TITLE = []
ALL_ABSTRACT = []

out_C_train=[]
out_C_test=[]

ANSWER=[]

print("----------------------------------讀檔 train-----------------")
with open("task2_trainset.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    n = 0
    for col in reader:
        n += 1

        if n == 1:
            continue

        task = get_tokens(col[6])
        task,num = turn_TASK(task)
        

        ALL_TITLE.append(col[1])
        ALL_ABSTRACT.append(col[2])

        title = get_tokens(col[1])
        title_filtered = [w for w in title if not w in stopwords.words('english')]
        title_count = Counter(title_filtered)
        title_stemmer = PorterStemmer()
        title_stemmed = stem_tokens(title_filtered, title_stemmer)
        title_count = Counter(title_stemmed)
        # print(title_count)

        # pst.stem(w)
        abstract = get_tokens(col[2])
        abstract_filtered = [w for w in abstract if not w in stopwords.words('english')]
        abstract_count = Counter(abstract_filtered)
        abstract_stemmer = PorterStemmer()
        abstract_stemmed = stem_tokens(abstract_filtered, abstract_stemmer)
        abstract_count = Counter(abstract_stemmed)
        # print(abstract_count)

        cate = col[4].split('/')

        '''
        if n>testing*2:
            break
    
        for t in task:
            if t!=0:
                TITLE_train.append(title_count)
                ABSTRACT_train.append(abstract_count)
                CATE_train.append(cate)
                TASK_train.append(t)
                combin_train.append(n)
        '''
        if n < testing:
            for t in task:
                if t!=0:
                    TITLE_train.append(title_count)
                    ABSTRACT_train.append(abstract_count)
                    CATE_train.append(cate)
                    TASK_train.append(t)
                    combin_train.append(n)
        else:
            #print(title_count)
            break
            TITLE_test.append(title_count)
            ABSTRACT_test.append(abstract_count)
            CATE_test.append(cate)
            #TASK_test.append(t)
            ANSWER.append([TTT(task[0]),TTT(task[1]),TTT(task[2]),TTT(task[3])])
        
        
print("----------------------------------讀檔 test-----------------")

with open("task2_public_testset.csv", "r", encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile, delimiter=',')

    n = 0
    for col in reader:
        n += 1

        if n == 1:
            continue
        '''
        if n>testing:
            break
        '''
        ALL_TITLE.append(col[1])
        ALL_ABSTRACT.append(col[2])

        title = get_tokens(col[1])
        title_filtered = [w for w in title if not w in stopwords.words('english')]
        title_count = Counter(title_filtered)
        title_stemmer = PorterStemmer()
        title_stemmed = stem_tokens(title_filtered, title_stemmer)
        title_count = Counter(title_stemmed)
        # print(title_count)

        # pst.stem(w)
        abstract = get_tokens(col[2])
        abstract_filtered = [w for w in abstract if not w in stopwords.words('english')]
        abstract_count = Counter(abstract_filtered)
        abstract_stemmer = PorterStemmer()
        abstract_stemmed = stem_tokens(abstract_filtered, abstract_stemmer)
        abstract_count = Counter(abstract_stemmed)
        # print(abstract_count)

        cate = col[4].split('/')

        #print(title_count)
        TITLE_test.append(title_count)
        ABSTRACT_test.append(abstract_count)
        CATE_test.append(cate)
        #combin_test.append(n)


def cate_all(CATEs):
    temp=[]
    for c in CATEs:
        for cc in c:
            issame=0
            for t in temp:
                if t==cc:
                    issame=1
                    break
            if issame==0:
                temp.append(cc)
    return temp

def TFIDF_score_train(train,c):
    out=[]
    key=[]
    for i, count in enumerate(train):
        out.append([])
        #print("Top words in document {}".format(i+1))
        scores = {word: tfidf(word, count, train) for word in count}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            if add_key(word, key) and round(score, 5) >= catch[c]:
                key.append(word)
            out[i].append([word, round(score,5)])
    return out, key

def TFIDF_score_test(test):
    out=[]
    for i, count in enumerate(test):
        out.append([])
        scores = {word: tfidf(word, count, test) for word in count}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words:
            #print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
            out[i].append([word, round(score,5)])
    return out

def KEYWORDS_vector(key, words):
    T = []
    for i in range(len(words)):
        temp = []
        num = 0
        for k in key:
            have = 0
            for w in words[i]:
                if w[0] == k:
                    have = 1
                    temp.append(w[1])
                    break
            if have == 0:
                num += 1
                temp.append(0)
        if num == len(key):
            print("error in keyword vector ")
        # print(temp)
        #print(" ")
        T.append(temp)
    return T

def KEYWORDS_vector_for_CATE(key, words):
    T = []
    for w in words:
        temp=[]
        for k in key:
            have=0
            for ww in w:
                if ww==k:
                    temp.append(1)
                    have=1
                    break
            if have==0:
                temp.append(0)
        T.append(temp)
    
    return T
                
def RandomForest(X_train,Y_train,X_test):
    forest = ensemble.RandomForestClassifier(n_estimators=10)
    forest.fit(X_train, Y_train)
    test_predicted = forest.predict(X_test)
    return test_predicted
   
def run(train,test,task_train,c):
    keyWord=[]
    print("--寫入---------------------------------")
    out_train, keyWord = TFIDF_score_train(train,c)
    out_test = TFIDF_score_test(test)
    print("--keyword vector train-----------------")
    out_train = KEYWORDS_vector(keyWord, out_train)
    print("--keyword vector test-----------------")
    out_test = KEYWORDS_vector(keyWord, out_test)
    print("--RandomForestClassifier --------------")
    test_predicted=RandomForest(out_train,task_train,out_test)
    return test_predicted


print("----------------------------------TITLE-----------------")
test_T_predicted=run(TITLE_train,TITLE_test,TASK_train,0)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print("----------------------------------ABSTRACT-----------------")
test_A_predicted=run(ABSTRACT_train,ABSTRACT_test,TASK_train,1)
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print("----------------------------------寫入 CATE-----------------")
keyword_C=cate_all(CATE_train)
print("----------------------------------keyword vector  C train-----------------")
out_C_train=KEYWORDS_vector_for_CATE(keyword_C,CATE_train)
print("----------------------------------keyword vector  C test-----------------")
out_C_test=KEYWORDS_vector_for_CATE(keyword_C,CATE_test)
print("----------------------------------RandomForestClassifier  CATE------")
test_C_predicted=RandomForest(out_C_train,TASK_train,out_C_test)

#-------------------------------------------

print("--------------compare-------")
all_answer=[]

for a in range(len(test_A_predicted)):
    T=[0,0,0,0]
    T[test_A_predicted[a]-1]+=1
    T[test_T_predicted[a]-1]+=1
    T[test_C_predicted[a]-1]+=1
    
    
    all_answer.append([TTT(T[0]),TTT(T[1]),TTT(T[2]),TTT(T[3])])



print("--------------------------------------------output------------")


with open('1_1__out_answer_public_test.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    for do in all_answer:
        writer.writerow(do)
'''
with open('1_1_ANSWER.csv', 'w', newline='') as csvFile:
    writer = csv.writer(csvFile)
    for do in ANSWER:
        writer.writerow(do)
'''