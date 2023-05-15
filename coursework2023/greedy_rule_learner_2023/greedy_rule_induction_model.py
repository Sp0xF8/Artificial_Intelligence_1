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
        print("Preprocessing training set")
        print(f"train_X: \n {train_X} \n train_y: \n {train_y}")
        print(f"train_X shape {train_X.shape} train_y shape {train_y.shape}")

        super().fit(train_X,train_y)     

        
        print("Finished Greedy Rule Induction")


    # def Get_Examples_Covered_By(self, rule:np.ndarray, not_covered:np.ndarray)->np.ndarray:
    #     errors = False
    #     covered = np.empty(shape=(0,4))

    #     # check that the rule is valid
    #     for example in not_covered:
    #         if self.Meets_Conditions(example, rule):
    #             if example[LABEL] != rule[LABEL]:
    #                 errors = True
    #                 break
    #             else:
    #                 covered = np.vstack((covered, example))

    #     if errors:
    #         print("Rule is invalid")
    #         return np.empty(shape=(0,4))
    #     else:
    #         return covered


    # def Meets_Conditions(self, example:np.ndarray, rule:np.ndarray)->bool:
    #     matches = False
    #     feature = rule[FEATURE]
    #     operator = rule[OPERATOR]
    #     threshold = int(rule[THRESHOLD])

    #     print(f"example: {example}")

    #     example_value = example[FEATURE]
    #     print(f"example value: {example_value}")
    #     print(f"threshold: {threshold}")
    #     print(f"operator: {operator}")

    #     if operator == "==":
    #         if example_value == threshold:
    #             matches = True
    #     elif operator == "<":
    #         if example_value < threshold:
    #             matches = True
    #     elif operator == ">":
    #         if example_value > threshold:
    #             matches = True





    def predict_one(self, example:np.ndarray)->int:
        '''
        Method that overrides the naive code in the superclass
        function GreedyRuleInduction.

        Parameters
            ---------
        example: numpy array of feature values that represent one exanple

        Returns: valid label in form of int index into set of values found in the training set
        '''
        prediction = self.default_prediction
        return prediction
    

