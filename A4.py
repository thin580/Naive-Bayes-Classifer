import pandas as pd
import numpy as np
import math as math
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))

# Load in the training set .csv
training_set = pd.read_csv("trg.csv")
training_set.head()

abstracts = training_set["abstract"].str.split(" ")
new_ab = []
new_ts = []

for i in abstracts:
    for word in i:
        if word not in en_stops:
            new_ts.append(word)
    new_ab.append(np.unique(new_ts))
    new_ts = []

classes = training_set["class"]
A = 0
B = 0
E = 0
V = 0

for i in range( 0, training_set["class"].size):
    if classes[i] == "A":
        A += 1
    elif classes[i] == "B":
        B += 1
    elif classes[i] == "E":
        E += 1
    elif classes[i] == "V":
        V += 1

prior_a = A / classes.size
prior_b = B / classes.size
prior_e = E / classes.size
prior_v = V / classes.size

a_df = pd.DataFrame({" ": [0]})
b_df = pd.DataFrame({" ": [0]})
e_df = pd.DataFrame({" ": [0]})
v_df = pd.DataFrame({" ": [0]})


for j in range(0, training_set["class"].size):
    for word in new_ab[j]:
        if classes[j] == "A":
            if word not in a_df:
                a_df[word] = 1
            else:
                a_df[word] += 1
        elif classes[j] == "B":
            if word not in b_df:
                b_df[word] = 1
            else:
                b_df[word] += 1
        elif classes[j] == "E":
            if word not in e_df:
                e_df[word] = 1
            else:
                e_df[word] += 1
        elif classes[j] == "V":
            if word not in v_df:
                v_df[word] = 1
            else:
                v_df[word] += 1
    #print(j)

a_df.to_csv("FreqA.csv", index=False)
b_df.to_csv("FreqB.csv", index=False)
e_df.to_csv("FreqE.csv", index=False)
v_df.to_csv("FreqV.csv", index=False)

freq_a = pd.read_csv("freqA.csv")
freq_a.head(10)
freq_b = pd.read_csv("freqB.csv")
freq_b.head()
freq_e = pd.read_csv("freqE.csv")
freq_e.head()
freq_v = pd.read_csv("freqV.csv")
freq_v.head()

for word in freq_a:
    if freq_a.loc[0,word] == 1:
        freq_a.drop(word, axis=1, inplace = True)

for word in freq_b:
    if freq_b.loc[0,word] == 1:
        freq_b.drop(word, axis=1, inplace = True)

for word in freq_e:
    if freq_e.loc[0,word] == 1:
        freq_e.drop(word, axis=1, inplace = True) #axis=1 refers to column name i.e. word

for word in freq_v:
    if freq_v.loc[0,word] == 1:
        freq_v.drop(word, axis=1, inplace = True)

freq_a.to_csv("newA.csv", index=False)
freq_b.to_csv("newB.csv", index=False)
freq_e.to_csv("newE.csv", index=False)
freq_v.to_csv("newV.csv", index=False)

freq_a = pd.read_csv("newA.csv")
freq_a.head()
freq_b = pd.read_csv("newB.csv")
freq_b.head()
freq_e = pd.read_csv("newE.csv")
freq_e.head()
freq_v = pd.read_csv("newV.csv")
freq_v.head()

freq_a.drop(" ", axis=1, inplace=True) 
freq_b.drop(" ", axis=1, inplace=True) #drops initialized 'empty' column name and key
freq_e.drop(" ", axis=1, inplace=True)
freq_v.drop(" ", axis=1, inplace=True)

unq_words = [] #find total unique words over all abstracts

for i in range(0, training_set["class"].size):
    for word in new_ab[i]:
        if (word not in unq_words) and (word in freq_a or word in freq_b or word in freq_e or word in freq_v):
            unq_words.append(word)

denom_a = freq_a.values.sum() + len(unq_words)
denom_b = freq_b.values.sum() + len(unq_words)
denom_e = freq_e.values.sum() + len(unq_words)
denom_v = freq_v.values.sum() + len(unq_words)

for i in range(0, training_set["class"].size):
    for word in new_ab[i]:
        if word in freq_a:
            if freq_a.at[0, word] > 1: #correct for conditionals that only occur in other frequency dataframes
                cond_a = (float(freq_a.at[0,word]) + float(1)) / float(denom_a) #Multinomial equation
                freq_a[word] = cond_a
        elif word in freq_b or word in freq_e or word in freq_v:
            cond_a = (float(0) + float(1)) / float(denom_a) #multinomial equation if doesn't exist in related abstract
            freq_a[word] = cond_a
    for word in new_ab[i]:
        if word in freq_b:
            if freq_b.at[0, word] > 1:
                cond_b = (float(freq_b.at[0,word]) + float(1)) / float(denom_b)
                freq_b[word] = cond_b
        elif word in freq_a or word in freq_e or word in freq_v:
            cond_b = (float(0) + float(1)) / float(denom_b)
            freq_b[word] = cond_b
    for word in new_ab[i]:
        if word in freq_e:
            if freq_e.at[0, word] > 1:
                cond_e = (float(freq_e.at[0,word]) + float(1)) / float(denom_e)
                freq_e[word] = cond_e
        elif word in freq_a or word in freq_b or word in freq_v:
            cond_e = (float(0) + float(1)) / float(denom_e)
            freq_e[word] = cond_e
    for word in new_ab[i]:
        if word in freq_v:
            if freq_v.at[0, word] > 1:
                cond_v = (float(freq_v.at[0,word]) + float(1)) / float(denom_v)
                freq_v[word] = cond_v
        elif word in freq_a or word in freq_b or word in freq_e:
            cond_v = (float(0) + float(1)) / float(denom_v)
            freq_v[word] = cond_v
    

freq_a.to_csv("condA.csv", index=False)
freq_b.to_csv("condB.csv", index=False)    
freq_e.to_csv("condE.csv", index=False)    
freq_v.to_csv("condV.csv", index=False)    

cond_a = pd.read_csv("condA.csv")
cond_a.head()
cond_b = pd.read_csv("condB.csv")
cond_b.head()
cond_e = pd.read_csv("condE.csv")
cond_e.head()
cond_v = pd.read_csv("condV.csv")
cond_v.head()



# Remove duplicate words and sort them alphabetically
#words = list(np.unique(np.sort(words)))


# Process the text, find a 'good model' with cross-validation
print("Text processing...")


# Train the NBC with this data (your own NBC code)
print("Training the NBC...")


# Use this 'good model' to generate classifications. 
def classify(abstracts):
    
    # Text processing, cleaning, outlier removal, attribute selection etc. 
    # This function must be deterministic 
    # eg. if you select the 100 most frequent words, it must be the 100 most frequent words in the TRAINING set not
    # in the 'abstracts' parsed
    print("Processing the test abstracts...")
    
    test_ab = abstracts.str.split(" ")

    classifyA = math.log(prior_a)
    classifyB = math.log(prior_b) 
    classifyE = math.log(prior_e) 
    classifyV = math.log(prior_v)

    prediction = []
    
    for i in range(0, len(test_ab)):
        for word in test_ab[i]:
            if word in cond_a:
                classifyA += math.log(cond_a.loc[0, word]) #naive bayes classifier
        for word in test_ab[i]:
            if word in cond_b:
                classifyB += math.log(cond_b.loc[0, word])
        for word in test_ab[i]:
            if word in cond_e:
                classifyE += math.log(cond_e.loc[0, word])
        for word in test_ab[i]:
            if word in cond_v:
                classifyV += math.log(cond_v.loc[0, word])

        #finds highest correlated classifier
        
        if (classifyA == max(classifyA, classifyB, classifyE, classifyV)): 
            prediction.append("A")
        elif (classifyB == max(classifyA, classifyB, classifyE, classifyV)):
            prediction.append("B")
        elif (classifyE == max(classifyA, classifyB, classifyE, classifyV)):
            prediction.append("E")
        elif (classifyV == max(classifyA, classifyB, classifyE, classifyV)):
            prediction.append("V")
        #resets the classifier per iteration
        classifyA = math.log(prior_a)
        classifyB = math.log(prior_b) 
        classifyE = math.log(prior_e) 
        classifyV = math.log(prior_v)
        
          
    
    # Run processed abstracts through the pre-trained naive bayes classifier
    print("Classifying the test abstracts...")

        
    # Temporary: use the null model. Assign everything to "E"
    return prediction
    
    
# Load in the test set .csv
test_set = pd.read_csv("tst.csv")

# Apply the model to the test set
test_set_class_predictions = classify(test_set["abstract"])
test_set["class"] = test_set_class_predictions


# Write the test set classifications to a .csv so it can be submitted to Kaggle
test_set.drop(["abstract"], axis = 1).to_csv("tst_kaggle.csv", index=False)
test_set.head()
