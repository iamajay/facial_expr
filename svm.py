

#### Libraries
# My libraries
import  loader 

# Third-party libraries
from sklearn import svm
import numpy as np
def svm_baseline():
    training_data=[]
    validation_data=[]
    test_data=[]
    training_data, validation_data, test_data = loader.load_data()
    # train
    
    clf = svm.SVC()
    #print training_data[1]
    #print training_data[0]
    #t=np.reshape(training_data,(-1,1))
    t_x=np.asmatrix(training_data[0])
    t_y=np.asmatrix(training_data[1])
    te_x=np.asmatrix(test_data[0])
    
    te_x=np.asmatrix(test_data[1])
    clf.fit(training_data[0], training_data[1])
    # test
    predictions = [int(a) for a in clf.predict(test_data[0])]
    num_correct = sum(int(a == y) for a, y in zip(predictions, test_data[1]))
    print "Baseline classifier using an SVM."
    print "%s of %s values correct." % (num_correct, len(test_data[1]))
    a=num_correct
    b=len(test_data[1])
    c=a/(b*1.0)*100
    print "accuracy - %f"%c
if __name__ == "__main__":
    svm_baseline()
    
