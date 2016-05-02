from PIL import Image
from numpy import genfromtxt
import gzip, cPickle
from glob import glob
import numpy as np
import pandas as pd
def dir_to_dataset(glob_files, loc_train_labels=""):    
    dataset = []
    for file_count, file_name in enumerate( sorted(glob(glob_files),key=len) ):
        image = Image.open(file_name)
        img = Image.open(file_name).convert('LA')
        pixels = [f[0] for f in list(img.getdata())]

        dataset.append(pixels)
        if file_count % 1 == 0:
            print("\t %s files processed"%file_count)


    if len(loc_train_labels) > 0:
        df = pd.read_csv(loc_train_labels)
        print df
        return np.array(dataset), np.array(df)
    else:
        return np.array(dataset)
    
if __name__ == '__main__':
    
    Data= dir_to_dataset("jaffe//*.tiff")
    y=[4,4,4,5,5,5,6,6,6,6,1,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,5,5,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,6,6,6,1,1,1,1,0,0,0,2,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,
       1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,
       5,5,5,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,6,1,1,1,0,0,0,2,2,2,3,3,3,]
    y=np.array(y)

train_set_x = Data[:150]
val_set_x = Data[151:160]
test_set_x = Data[161:212]
train_set_y = y[:150]
val_set_y = y[151:160]
test_set_y = y[161:212]

#print train_set_x.shape
#print train_set_y.shape
train_set = train_set_x, train_set_y
val_set = val_set_x, val_set_y
test_set = test_set_x, test_set_y
#print test_set_y.shape;
#print test_set_x.shape;
#print val_set_x.shape;
#print val_set_y.shape;
dataset = [train_set, val_set, test_set]
f = gzip.open('face256.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()


    

