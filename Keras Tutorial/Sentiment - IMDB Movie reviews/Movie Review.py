## This program is use to understand the sentiments of the movie reviews using imbd data set 
# imdb data set consists of 25,000 records for training and 25,000 for testing. 
# this data set is available in Keras as well as in Tensorflow Libraries 


import numpy
from keras.datasets import imdb
from matplotlib import pyplot

## load the imdb data set 
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# contains reviews 
X = numpy.concatenate((X_train, X_test), axis=0)
#print(X[1])
y = numpy.concatenate((y_train,y_test), axis=0)
#print(y[1])

# summary of data size
print("Training data")
print(X.shape)
print(y.shape)

# Label data class and its levels
print("Class of Labeled data")
print(numpy.unique(y))

print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))

# summarize reivew lenght
print("review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))
# plot review length
pyplot.boxplot(result)
pyplot.show()

# MLP for the IMDB problem
