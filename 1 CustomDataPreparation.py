import os
import cv2
import numpy as np
from tqdm import tqdm
 

# Data preprocessing. Set to True if need to change something in training data.
REBUILD_DATA = True

class DataBulder():
    IMG_SIZE = 400
    CANCER = "D://Projects//Yura//data//OLD//_Raked 12.05.20//Cancer1050//"
    HEALTHY = "D://Projects//Yura//data//empty//"
    LABELS = {CANCER: 1, HEALTHY: 0}
    training_data = []
    
    cancercount = 0
    healthycount = 0
    
    
    def make_training_data(self):
        for label in self.LABELS:
            #print(label)
            for f in tqdm(os.listdir(label)):
                if "jpg" in f:
                    try:
                        path = os.path.join(label, f)
                        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (self.IMG_SIZE, self.IMG_SIZE))
                        self.training_data.append([np.array(img),np.eye(2)[self.LABELS[label]]])

                        if label == self.CANCER:
                            self.cancercount += 1
                        elif label == self.HEALTHY:
                            self.healthycount += 1
                    except Exception as e:
                        pass
                
                
        np.random.shuffle(self.training_data)
        np.save("D://Projects//Yura//OnlyCancer1050.npy", self.training_data)
        print("Cancer: ", self.cancercount)
        print("Healthy: ", self.healthycount)

        
        
if REBUILD_DATA:
    databuild = DataBulder()
    databuild.make_training_data()
    
    
training_data = np.load("D://Projects//Yura//OnlyCancer1050.npy", allow_pickle=True)
print("Number of samples: ", len(training_data))

print("DATA PREPROCESSED")