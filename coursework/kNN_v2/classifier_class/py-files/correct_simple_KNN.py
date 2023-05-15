import math
import numpy as np


# your KNN class code here
def euclidean_distance(a,b):
    ''' calculates the Euclidean (straight line) distance between two items a and b'''
    ''' this is just Pythagoras' theorem in N-dimensions'''
    #a and b must have same number of dimensions/feastures
    assert a.shape[0] == b.shape[0]
    distance=0.0
    for feature in range( a.shape[0]):
        difference = a[feature] - b[feature]
        distance= distance + difference*difference
    return math.sqrt(distance)      


class simple_KNN:

    def __init__(self, K, verbose = False):
        """init function, Needs adapting to take an argument K with default 1"""
        
        ## SPECIFY function to calculate distance metric d(i,j) for any two items *i* and *j*
        self.distance= euclidean_distance
        ## SET value of K
        #===> change line below to take K from an argument to this init() method <====
        self.K=K
        
        #just affects prints to screen
        self.verbose= verbose     


    def fit(self,X,y):
        """stores the dataset values X and labels y. Same code as 1-NN"""
        
        ##SET numExemplars = READ(number of rows in training data)  
        self.numTrainingItems = X.shape[0]
        
        ##SET numFeatures = READ(number of columns in training data) 
        self.numFeatures = X.shape[1]
        
        # Just store a local copy of the training data as two arrays:*   
        ## CREATE_AND_FILL(X_train of shape (numExemplars , numFeatures)).     
        self.modelX = X
        ## CREATE_AND_FILL(y_train of shape( numExemplars))
        self.modelY = y
        
        #additional reporting -  not part of algorithm
        self.labelsPresent = np.unique(self.modelY) # list the unique values found in the labels provided
        if (self.verbose):
            print(f"There are {self.numTrainingItems} training examples"
                  f"   each described by values for {self.numFeatures} features"
                 )
            print(f"So self.modelX is a 2D array of shape {self.modelX.shape}")
            print(f"self.modelY is a list with {len(self.modelY)} entries,"
                  f"each being one of these labels {self.labelsPresent}"
                 )


  
    def predict(self,newItems):
        """ make a prediction for each new item - same code as 1-NN"""
        
        ## SET numToPredict = READ(number of rows in newItems) 
        numToPredict = newItems.shape[0]
        
        ## SET predictions = CREATE_EMPTYARRAY( numToPredict)
        predictions = np.empty(numToPredict)
        
        ##FOREACH item in (0...,numToPredict-1) 
        for item in range(numToPredict):
        
            ##...SET predictions[item] = predictNewItem ( newItems[item]) 
            thisPrediction = self.predict_new_item ( newItems[item])
            predictions[item] = thisPrediction
            
            
        ## RETURN predictions    
        return predictions  
 

 
    def predict_new_item(self,newItem):
        """make prediction for single item. Step 1 is same as 1-NN steps 2 and 3 need writing"""

        ## Step 1:   
        ## Make 1D array distances from newItem to each training set item*   
        distFromNewItem = np.zeros((self.numTrainingItems)) 

        ## FOREACH exemplar in (0,...,numExemplars -1  
        for stored_example in range (self.numTrainingItems):
            ## ...SET distFromNewItem [exemplar] = d (newItem , X_train[exemplar] )   
            distFromNewItem[stored_example] = self.distance(newItem,  self.modelX[stored_example])
        

        ## Step 2: Get indexes of the k nearest neighbours for our new item    
    
        ## SET closestK = GET_IDS_OF_K_CLOSEST(K,distFromNewItem)
        #closestK is array with K elements  
        #===> add one line of  code  to call the new function <===       
        closestK= get_ids_of_k_closest(distFromNewItem, self.K)
        #print(closestK)
        
        ## Step 3: Calculate most popular of the m possible labels* 
    
        ## SET labelcounts = CREATE(1D array with m zero values)  
        #==> add one line of code using numpy.zeros to do this.  <===
        #remember that in fit() we created self.labelsPresent
        # so m = len(self.labelsPresent) 
        labelcounts = np.zeros(len(self.labelsPresent))
      
        ##    FOREACH  k in (0,...K-1)  
        #== add line of code putting in a for() loop here <===
        for k in range (self.K):
    
            ##... SET thisindex = closestK[k] 
            #==> add line of code to do this
            thisindex = closestK[k]
            
            ##... SET thislabel = y_train[thisindex]  
            #==> add line of code to do this
            thislabel= self.modelY[thisindex]
            
            ##... INCREMENT labelcounts[thislabel] 
            #==> add line of code to do this
            labelcounts[thislabel] +=1

         ##SET thisPrediction = READ(index of labelcounts with highest value)    
         #==> add one or two lines of code to do this
         # suggest you google "python highest value in numpy array" 
        thisPrediction = np.argmax(labelcounts)
        
        ##RETURN thisPrediction   
        return thisPrediction
    
    
  
                
                
def get_ids_of_k_closest(distFromNewItem, K):
    """new function that returns array containing indexes of K closest items"""
    
    # Several way of doing this.  
    #This one just does K iterations of the loop from 1-NN that found the sigble closest 

    ## SET closestK= CREATE(1D array with K values) 
    #==> add line of code to do this using np.empty(k,dtype=int)  <==
    closestK=np.empty(K,dtype=int)

    ##SET arraySize = len(distFromNewItem)  
    #==> add line of code to do this, 
    #distFromNewItem is a numpy array so you use its .shape[0] attribute <===
    arraySize = distFromNewItem.shape[0]
    
    ## FOR k in (0,...,K-1)  
    #==> add line of code to do this
    for k in range(K):
        
        # look at 1-NN predict_new_item() for inspiration for the contents of this loop

        ##... SET thisClosest=0
        #==> add line of code to do this
        thisClosest=0 
        ##... FOR exemplar in (0,...,arraySize -1) 
        #==> add line of code to do this
        for exemplar in range (arraySize):
            ##......IF ( distFromNewItem[exemplar] < distFromNewItem[thisClosest]  )  
            #==> add line of code to do this
            if (distFromNewItem[exemplar] < distFromNewItem[thisClosest]):
                ##......... SET thisClosest = exemplar
                #==> add line of code to do this
                thisClosest=exemplar
        ##... SET closestK[k] = thisClosest  
        #===> add line of code to do this
        closestK[k] = thisClosest
        ##... SET distFromNewItem[thisClosest] = BigNumber 
        # so we don't pick it again in next loop
        #==>add line of code to do this, you could use 100000 for bignum
        distFromNewItem[thisClosest]=1000000
    ##RETURN closestK
    #==> add line of code to do this
    return closestK            
                