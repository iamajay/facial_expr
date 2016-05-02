# facialexpr

#FACIAL EXPRESSION RECOGNITION SYSTEM USING DEEP LEARNING (Machine Learning)
#--> Machine Learning program is developed using deep convolution neural network. We have used theano,PIL
#and scipy libraries of Python for implementation of this ML program where seven different expressions of face are
#recognized based on trained data set.



#Libraries Used :- 1.Theano.
#                  2.PIL
#                 3.Numpy
#                  4.cPickle
#                  5.Gzip
#                  6.sklearn
#Here two models are there-: 1. Convolution layer(deep learning)
#                            2. SVM
#Dataset Used- jaffer(213 images) -converted into pickle format using Converter.py and processed using crop.py
#net3.py has been used in the deep learning algorithm for buiding the desired convolution layer and which uses the concept of feature #mapaping 
#Here codeFACIAL is the main code which builds the main trained model of deep learning using 6 convolution layer and 2 fully connected #layers
#SVM model is build by loading the same jaffe data which has been built using Converter.py in pickle format and then fitting the #training model and
#label to predict the accuracy of the code .
#Accuracy is better if size of the dataset is large and it has been preprocessed by any feature extraction model on the dataset .We #had not appliedthe feature extraction on the code.If you are familiar with the same, you can append the logic of same in this model.
