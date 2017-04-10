import tensorflow as tf
import numpy as np
from CIFAR.cifarDataLoader import maybe_download_and_extract,load_class_names,load_training_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class model():
    def __init__():
        pass

class catsanddogs():
    def __init__(self):
        maybe_download_and_extract()
        imgs,cls = load_training_data()
        cats = []
        dogs = []
        self.data=[]
        for i in range(0,len(cls)):
            if cls[i] ==  3:
                cats.append(imgs[i])
                self.data.append(imgs[i])
            elif cls[i] == 5:
                dogs.append(imgs[i])
                self.data.append(imgs[i])
            else :
                pass

    def getdata(self):
        return self.data        
        

def main():
    maybe_download_and_extract()
    data = catsanddogs()
    candd = data.getdata()
    print('The number of samples are', len(candd))
if __name__ == "__main__":
    # execute only if run as a script
    main()
