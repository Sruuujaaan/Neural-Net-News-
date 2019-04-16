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
# df.drop(indexNames , inplace=True)



print(df.shape)
X = df.iloc[:, 2].values
Y = df.iloc[:, 1].values


print(  (max([len(x) for x in X]))    )

from sklearn.feature_extraction.text import TfidfVectorizer  # to count the occurence of a word
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder
import pickle

count_vect = TfidfVectorizer(max_df = 0.9, min_df = 0.05).fit(X)
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

# #Random Shuffle (Manually coded)
# from sklearn.utils import shuffle
# df = shuffle(df)
# print(df)
#
# #Random Selection from shuffled data
# def randomSelection(matrix, target, test_proportion):
#     ratio = int(matrix.shape[0]/test_proportion)
#     X_train = matrix[ratio:,:]
#     X_test =  matrix[:ratio,:]
#     Y_train = target[ratio:,:]
#     Y_test =  target[:ratio,:]
#     return X_train, X_test, Y_train, Y_test
#
# X_train, X_test, y_train, y_test = randomSelection(X, Y, 8)

input = X.shape[1]
output = Y.shape[1]

from keras.models import Sequential
from keras.layers import Dense,Dropout


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=900, kernel_initializer='uniform', activation='relu', input_dim=input))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=650, kernel_initializer='uniform', activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units=500, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=output, kernel_initializer='uniform', activation='softmax'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Reduces learning rate when there is no improvement in learning rate
from keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, min_lr=0.001)
# Fitting the ANN to the Training set.
classifier.fit(X_train, y_train, batch_size=1000, epochs=100,callbacks=[reduce_lr])
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

