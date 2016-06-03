# facialexpr

FACIAL EXPRESSION RECOGNITION SYSTEM USING DEEP LEARNING (Machine Learning)<br />
--> Machine Learning program is developed using deep convolution neural network. We have used theano,PIL<br />
and scipy libraries of Python for implementation of this ML program where seven different expressions of face are<br />
recognized based on trained data set.<br />
<br />
<br />
Libraries Used :-<br /> 1.Theano.<br />
                  2.PIL<br />
                 3.Numpy<br />
                  4.cPickle<br />
                  5.Gzip<br />
                  6.sklearn<br />
                  <br />
                  <br />
Here two models are there-: <br />1. Convolution layer(deep learning)<br />
                            2. SVM<br /><br />
                            <br /><br />
Dataset Used- jaffer(213 images) -converted into pickle format using Converter.py and processed using crop.py<br />
net3.py has been used in the deep learning algorithm for buiding the desired convolution layer and which uses the concept of feature mapaping <br />
Here codeFACIAL is the main code which builds the main trained model of deep learning using 6 convolution layer and 2 fully connected layers<br />
SVM model is build by loading the same jaffe data which has been built using Converter.py in pickle format and then fitting the training<br /> model and label to predict the accuracy of the code .<br />
Accuracy is better if size of the dataset is large and it has been preprocessed by any feature extraction model on the dataset .<br />
We had not applied the feature extraction on the code. If you are familiar with the same, you can append the logic of same in this model.<br />

<br />
