#UnComment if you dont want to use GPU
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

#Data Preprocessing
import pandas as pd
dataset1 = pd.read_csv('articles1.csv')
dataset2 = pd.read_csv('articles2.csv')
dataset3 = pd.read_csv('articles3.csv')
datasetTrain = pd.concat([dataset1, dataset2, dataset3], ignore_index=True)
df = datasetTrain.drop(['id', 'title', 'author', 'date', 'year', 'month', 'url'], 1)
df = df.dropna()

#drop the rows where content exceeds 100000 words
df = df.drop(df[df['content'].map(len) > 100000].index)

df['content'] = df['content'].str.lower().str.split()
from nltk.corpus import stopwords
stop = stopwords.words('english')
df['content'] = df['content'].apply(lambda x: [item for item in x if item not in stop])
df['content'] = df['content'].apply(' '.join)
# df.drop(indexNames , inplace=True)

print(df.groupby('publication').count())

#Sampling
Atlantic = df[df['publication'] == "Atlantic"]
Breitbart = df[df['publication'] == "Breitbart"]
BusinessInsider  = df[df['publication'] == "Business Insider"]
BuzzfeedNews = df[df['publication'] == "Buzzfeed News"]
CNN = df[df['publication'] == "CNN"]
FoxNews = df[df['publication'] == "Fox News"]
Guardian = df[df['publication'] == "Guardian"]
NPR = df[df['publication'] == "NPR"]
NationalReview = df[df['publication'] == "National Review"]
NewYorkPost  = df[df['publication'] == "New York Post"]
NewYorkTimes  = df[df['publication'] == "New York Times"]
Reuters  = df[df['publication'] == "Reuters"]
TalkingPointsMemo   = df[df['publication'] == "Talking Points Memo"]
Vox  = df[df['publication'] == "Vox"]
WashingtonPost  = df[df['publication'] == "Washington Post"]


AtlanticO = Atlantic.sample(8000, replace=True)
BreitbartU = Breitbart.sample(10000)
BusinessInsiderO = BusinessInsider.sample(8000, replace=True)
BuzzfeedNewsO = BuzzfeedNews.sample(7500, replace=True)
FoxNewsO = FoxNews.sample(7500, replace=True)
GuardianO = Guardian.sample(9000, replace=True)
NPRU = NPR.sample(11000)
NationalReviewO = NationalReview.sample(9000, replace=True)
NewYorkPostU = NewYorkPost.sample(11000)
NewYorkTimesO = NewYorkTimes.sample(9000, replace=True)
TalkingPointsMemoO = TalkingPointsMemo.sample(8000, replace=True)
VoxO = Vox.sample(8000, replace=True)

df1 = pd.concat([AtlanticO, BreitbartU, BusinessInsiderO, BuzzfeedNewsO,
FoxNewsO,GuardianO,NationalReviewO,NewYorkPostU,NewYorkTimesO,TalkingPointsMemoO,VoxO,
                          CNN,NPRU,Reuters,WashingtonPost], axis=0)


print(df1.groupby('publication').count())


# print(df.shape)
X = df1.iloc[:, 2].values
Y = df1.iloc[:, 1].values


print(  (max([len(x) for x in X]))    )

from sklearn.feature_extraction.text import TfidfVectorizer  # to count the occurence of a word
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import pickle

count_vect = TfidfVectorizer(max_df = 0.5,min_df = 0.08).fit(X)
X = count_vect.transform(X)
with open('countVect.pickle', 'wb') as handle:
    pickle.dump(count_vect, handle, protocol=pickle.HIGHEST_PROTOCOL)

tfidf_transformer = TfidfTransformer().fit(X)
with open('transformer.pickle', 'wb') as handle:
    pickle.dump(tfidf_transformer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X = tfidf_transformer.transform(X)

X = X.toarray()


"""Label Encoding"""
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
with open('labelencodery.pickle', 'wb') as handle:
    pickle.dump(labelencoder_Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""One Hot Encoding"""
y = Y.reshape(-1, 1)  # Because Y has only one column

from sklearn.preprocessing import LabelBinarizer,OneHotEncoder
lb = OneHotEncoder(categories='auto')
Y = lb.fit_transform(y).toarray()

with open('labelbinizer.pickle', 'wb') as handle:
    pickle.dump(lb, handle, protocol=pickle.HIGHEST_PROTOCOL)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

input = X.shape[1]
output = Y.shape[1]
print(X.shape)
print(Y.shape)
from keras.models import Sequential
from keras.layers import Dense,Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=650, kernel_initializer='uniform', activation='relu', input_dim=input))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=450, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=350, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.1))
classifier.add(Dense(units=output, kernel_initializer='uniform', activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Reduces learning rate when there is no improvement in learning rate
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
# Fitting the ANN to the Training set.
classifier.fit(X_train, y_train, batch_size=500, epochs=100
               ,callbacks=[reduce_lr])
classifier.save('NewsClassifier.h5')




#Prediction
# from keras.models import load_model
# classifier = load_model('classifierU.h5')


y_pred = classifier.predict(X_test)
print("PREDICTED VALUES \n")
inv = lb.inverse_transform(y_pred)
print(inv)
result = labelencoder_Y.inverse_transform(inv)
print(result)

print("ACTUAL VALUES \n")
inv = lb.inverse_transform(y_test)
result1 = labelencoder_Y.inverse_transform(inv)
print(result1)

from sklearn.metrics import accuracy_score
accuracyScore = accuracy_score(result1, result)
print("accuracy percentage=", accuracyScore * 100, "%")

import csv
lines = [(result1[i-1], result[i-1]) for i in range(1, len(result)+1)]
header = ['Actual','Predicted']
with open("Submission.csv", "w", newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(header) # write the header
    # write the actual content line by line
    for l in lines:
        writer.writerow(l)

