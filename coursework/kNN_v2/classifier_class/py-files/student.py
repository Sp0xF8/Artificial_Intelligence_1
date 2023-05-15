import numpy as np
import math

class simple_KNN:

    def __init__(self, K, verbose = False):
        #Distance from two random items
        self.distance= euclidean_distance
        #print(f"distance{self.distance}")=140483187706176
        self.K=K
        #print(f"k{self.K}")=[1,3,9]
        self.verbose= verbose

    def fit(self,X,y):
        """1-NN fit function"""
        #Row number in training data (shape[0]) is number of example
        self.numTrainingItems = X.shape[0]
        #print(f"numTrainingItems{self.numTrainingItems}")=100
        
        #Column number in training data (shape[1]) is number of features
        self.numFeatures = X.shape[1]
        #print(f"numFeatures{self.numFeatures}")=4
        
        #ModelX of the training data with shape of number of example and features   
        self.modelX = X
        #print(f'modelX{self.modelX}')
        
        #ModelY with shape of number of example 
        self.modelY = y
        #print(f'modelY{self.modelY}') = lenght is 100
        
        #List all different label values 
        self.labelsPresent = np.unique(self.modelY)
        #print(f"labelsPresent{self.labelsPresent}")= [0,1,2] for K=1
        
        if (self.verbose):
            print(f"There are {self.numTrainingItems} training examples, each described by values for {self.numFeatures} features")
            print(f"So self.modelX is a 2D array of shape {self.modelX.shape}")
            print(f"self.modelY is a list with {len(self.modelY)} entries, each being one of these labels {self.labelsPresent}")


    def predict(self,newItems):
        """ 1NN Prediction function to find out predictions for every new item"""
        
        #Row number is the number of predictions
        numToPredict = newItems.shape[0]
        #print(f"numToPredict{numToPredict}")=50
        
        #Empty array for predictions with number of predictions
        predictions =  np.zeros(numToPredict,dtype=int)
        #print(f"PREDICTIONS{predictions}")
        
        #For loop  from 0 to numToPredict-1
        for item in range(numToPredict):
            #fill the predictions of items from prediction new item function
            thisPrediction = self.predict_new_item(newItems[item])
            predictions[item]=thisPrediction
            #print(f"PREDICTIONS[item]{predictions[item]}")#
                  
        return predictions  
 

    def predict_new_item(self,newItem):
        """Prediction function for every item"""
        
        #Step 1:1D distance array from new item to each training set item --- same with 1-NN  
        distFromNewItem = np.zeros((self.numTrainingItems)) 
        #print(f"distFromNewItem{distFromNewItem}") = all zeros
        
        #For loop for each example from o to numExamplars-1
        for stored_example in range (self.numTrainingItems):
            #Every distance from new item goes to stored example  
            distFromNewItem[stored_example] = self.distance(newItem,self.modelX[stored_example])
            #print(f"distFromNewItem[stored_example]{distFromNewItem[stored_example]}")
        #print(f"distFromNewItem{distFromNewItem}") 
        
        #Step 2: Indexes of the k nearest neighbours for our new item
        #Closest K is get_ids_of_k_closest(distFromNewItem,self.K) function -- array of all K elements
        closestK = get_ids_of_k_closest(distFromNewItem,self.K)
        #print(f"closestK in predict_new_item(function{closestK}")
        
 
        #Step 3: Calculate most popular of the m possible labels 
        #1D array with 0 of lenght of labelsPresent for labelcounts----numpy.zeros
        m = len(self.labelsPresent)
        labelcounts=np.zeros(m,dtype=int)
        #print(f"labelcounts{labelcounts}")
    
        #For loop from o to K-1
        for k in range(self.K):
            
            #Index is closestK
            thisindex = closestK[k]
            
            #Label is index of y_train
            thislabel = y_train[thisindex]
            #print(f"thislabel{thislabel}")#
            
            #Increment label of labelcounts
            labelcounts[thislabel]+=1
            #print(f"labelcounts[thislabel]+1{labelcounts[thislabel]}")#
            
        #Prediction is highest labelcount
        thisPrediction = np.argmax(labelcounts)
        print(f"thisPrediction{thisPrediction}")#
        
        print(f"labelcounts{labelcounts}")
        
        #print(f"thisPrediction outside loop {thisPrediction}")#
        return thisPrediction
                   
                
def get_ids_of_k_closest(distFromNewItem, K):
    """Array of K closest items' index"""
    
    #ClosestK array ------np.zeros()
    closestK = np.zeros(K,dtype = int)
    #closestK = np.empty(K, dtype = int)
    #print(f"closestK{closestK}")
    
    #Array size is lenght of distance from new item which is a numpy array so distance = shape[0] attribute
    arraySize = distFromNewItem.shape[0]
    #print(f"arraySize{arraySize}")=100
    
    #For loop from o to K-1
    for k in range(K):
        #Let's say closest is 0 for starting
        thisClosest=0    
        
        #For loop from o to arraysize-1
        for exemplar in range (arraySize):
            
            #If this exemplar's distance smaller than this closest's distance, our closestK will be exemplar 
            if ( distFromNewItem[exemplar] < distFromNewItem[thisClosest]):
                
                #This closest will be our exemplar
                thisClosest = exemplar
                #print(f"thisClosest{thisClosest}")
                
        #ClosestK's kth index will be thisClosest
        closestK[k] = thisClosest
        #print(f"closestK[k]{closestK[k]}")
                
        #We will set the distance to 100000 so program won't pick in next loop
        distFromNewItem[thisClosest] = 100000
        
    #print(f"closestK{closestK}")
    return closestK

def euclidean_distance(a,b):
    ''' Euclidean (straight line) distance between two items '''
    
    #a's number of dimension(features) == b's number of dimension(features)
    assert a.shape[0] == b.shape[0]
    
    distance=0.0
    
    for feature in range( a.shape[0]):
        difference = a[feature] - b[feature]
        distance= distance + difference*difference
        
    return math.sqrt(distance) 
