'''
Template submission for coursework task implementing greedy rule induction algorithm
 @Jim Smith: james.smith@uwe.ac.uk 2023
 Students should implement the pseudocode provided 
 to complete the two funtions fit() and predict_one() as  indicated below
 
 
'''
from learned_rule_model import LearnedRuleModel
import numpy as np

#define some constants for where things are in a rule
FEATURE=0
OPERATOR=1
THRESHOLD=2
LABEL = 3



class GreedyRuleInductionModel(LearnedRuleModel):
    '''
    sub-class that uses greedy constrcutive search to find a set of rules
    that classify a dataset
    '''
    def __init__(self,max_rules=10, increments=25):
        '''constructor 
        calls the init function for the super class
        and inherit all the other methods
        '''
        super().__init__(max_rules=max_rules, increments=increments)


        
    def fit( self,train_X:np.ndarray,train_y:list):
        ''' Learns a set of rules that fit the training data
             calls then extends the superclass method.
            Makes repeated use of the method __get_examples_covered_by()
            
            Parameters
            ----------
            train_X - 2D numpy array of instance feature values
                      shape (num_examples, num_features)
            train_y - 1D numpy array of labels, shape(num_examples,0)
        '''

        #  superclass method preprocesses the training set  
        super().fit(train_X,train_y)     
        
        ###== YOUR CODE HERE====####
            ## I suggest you copy in the pseudocode for the function function GreedyRuleInduction
            ##from the lecture then code to that
            ## some of the lines in that pseudocode have been covered above
            ##
            ## HINT 1: you probably want to use the superclass method _get_examples_covered_by()
            ##
            ## HINT 2: smaller_array = numpy.delete(my_array, my_set , axis=0) creates a smaller_array
            ## by removing all the rows from my_array with indexes in the list my_set
            ##
            ## HINT 3: during development it might help to put in varous print() statements
            ## for example, each time around the main loop
            ##
            ##  HINT 4: you can add a 1D array called 'best_new_rule' to the rule set using
            ##.  self.rule_set= np.row_stack((self.rule_set, best_new_rule)).astype(int)


        # print("printing Rule Set: ")
        # print(self.rule_set)
        # print(self.rule_set.shape)
        # print(type(self.rule_set))

        

        # print()

        # print("printing thresholds:")
        # print(self.thresholds)

        # beta = 0
        # for group in self.thresholds:
        #     for threshold in group:
        #         print()
        #         print()
        #         print(f"beta: {beta} | Feature: {train_X[beta]} threshold: {threshold} Label: {train_y[beta]} ")
        #         print(f"{type(train_X[beta][0])}    {type(threshold)}    {type(train_y[beta])}")
        #         beta += 1
        # print()
        # print()

        # for i in range(len(train_X)):

        #     print(f" ID) {i} | Feature: {train_X[i]} Label: {train_y[i]}")
        #     print()


        # print() #  - Debugging
        # print()  #  - Debugging
        # print("------------------Debug------------------") #  - Debugging
        # print()  #  - Debugging

        # print(self.labels) #  - Debugging

        not_covered = train_X.copy()
        labels_not_covered = train_y.copy()

        self.default_prediction = np.argmax(np.bincount(train_y))

        # print(f"printing default prediction: {self.default_prediction}")  #  - Debugging
        # print(f"printing default prediction type: {type(self.default_prediction)}") #  - Debugging

        improved = True
        while (len(not_covered) > 0 and len(self.rule_set) < self.max_rules and improved == True):
            improved = False
            best_new_rule = []
            best_covered = []

            for i in range(len(not_covered[0])):
                for j in range(len(self.operator_set)):
                    jk = 0
                    for k in self.thresholds[jk]:

                        for l in self.labels:
                            print(f"feature: {not_covered[i][jk]} | operator: {j} | threshold: {k} | label: {l}")
                            new_rule = np.array([i, j, k, l])
                            covered = super()._get_examples_covered_by(new_rule, not_covered, labels_not_covered)

                            if len(covered) > len(best_covered):
                                best_covered = covered
                                best_new_rule = new_rule
                                improved = True
                    jk += 1
                
            
            if improved == True:    

                print(f"printing best new rule: {best_new_rule}")


                not_covered = np.delete(not_covered, best_covered, axis=0)
                labels_not_covered = np.delete(labels_not_covered, best_covered, axis=0)

                self.rule_set = np.row_stack((self.rule_set, best_new_rule)).astype(int)

                print(f"printing not_covered: \n {not_covered}")

                

        print(self.rule_set)
        

        
    
    def predict_one(self, example:np.ndarray)->int:
        '''
        Method that overrides the naive code in the superclass
        function GreedyRuleInduction.
        
        Parameters
         ---------
        example: numpy array of feature values that represent one exanple
    
        Returns: valid label in form of int index into set of values found in the training set
        '''
        prediction=999


        
  
        ###== YOUR CODE HERE====####
        ### Start copy-pasting in the pseudocode for 
        ### function       makePrediction(example,ruleset)
        ###
        ### Hint 1: you should have set self.default_prediction in your fit() method
        ###
        ### Hint 2 : You may well want to make use of the supporting method _meets_conditions() 
        ###
        ### Hint 3: make sure your code changes what is held in the variable prediction!


        prediction = self.default_prediction

        for rule in self.rule_set:
            if super()._meets_conditions(example, rule):
                prediction = rule[LABEL]
                break

        return prediction
    

