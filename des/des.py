import math 
import numpy as np 
import pandas as pd 
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split  

from sklearn.feature_selection import RFE   
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

def get_value(arr):
    if np.ndim(arr) == 0:  # Scalar case
        return arr.item()
    elif np.ndim(arr) == 1 and len(arr) == 1:  # Single element array
        return arr[0]
    elif np.ndim(arr) == 2 and arr.shape[0] == 1 and arr.shape[1] == 1:  # Nested array with single element
        return arr[0, 0]
    else:
        raise ValueError("Array shape not supported")


def select(features, supports): 
    selected_features = [] 
    
    for i in range(len(features)): 
        if supports[i]: 
            selected_features.append(features[i]) 
    
    return selected_features


def count_model_occurrences(list_of_lists):
    # Initialize a dictionary to store counts of each digit
    digit_counts = {}

    # Iterate through each sublist
    for sublist in list_of_lists:
        # Iterate through each digit in the sublist
        for digit in sublist:
            # Increment the count of the digit in the dictionary
            if digit in digit_counts:
                digit_counts[digit] += 1
            else:
                digit_counts[digit] = 1

    return digit_counts
    

class BaseDES: 
    def __init__(self, pool_regressors=None, k=7, knn_metric='minkowski', metrics='mse', threshold=0.2, feature_selection=False): 
        self.pool_regressors   = pool_regressors 
        self.k                 = k 
        self.knn_metric        = knn_metric 
        self.threshold         = threshold 
        self.feature_selection = feature_selection

        if metrics == 'mse': 
            self.eval_metric = mean_squared_error 
        
        if metrics == 'mape': 
            self.eval_metric = mean_absolute_percentage_error 


    def get_region_of_competence(self, query): 
        nbrs    = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric).fit(self.X_dsel[self.selected_features])         
        indices = nbrs.kneighbors(query[self.selected_features].values.reshape(1, -1), return_distance=False) 

        self.roc = self.X_dsel.iloc[indices[0]] 
        self.roc_labels = self.y_dsel.iloc[indices[0]] 
        

    def estimate_competence(self):
        competence_list = [] 
        
        for regressor in self.pool_regressors: 
            preds = regressor.predict(self.roc[self.selected_features]) 
            competence = self.eval_metric(self.roc_labels, preds) 
            competence_list.append(competence) 

        return competence_list 



class DES(BaseDES): 
    def __init__(self, pool_regressors=None, k=7, knn_metric='minkowski', metrics='mse', threshold=0.2, feature_selection=False):
        super(DES, self).__init__(pool_regressors=pool_regressors, k=k, 
                                  knn_metric=knn_metric, metrics=metrics, 
                                  threshold=threshold, feature_selection=feature_selection) 
        

    def fit(self, X_train=None, y_train=None, split=0.2, random_state=42):
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)  

        features = X_train.columns.tolist() 

        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=split, random_state=random_state) 

        self.X_dsel = X_dsel 
        self.y_dsel = y_dsel 

        if self.feature_selection: 
            # feature importance 
            model_rf = LGBMRegressor(verbose=-1, random_state=45)  
            selector = RFE(model_rf, step=1)
            selector = selector.fit(X_train, y_train) 
    
            self.selected_features = select(features, selector.support_) 
        else: 
            self.selected_features = features 

        # Train models 
        for model in self.pool_regressors: 
            model.fit(X_train[self.selected_features], y_train) 


    
    def select(self, competences):
        criteria = self.threshold 
        selected_models_indices = []  

        while(len(selected_models_indices) <= 0):  
            for i in range(len(competences)):  
                if competences[i] <= criteria: 
                    selected_models_indices.append(i) 
            
            if len(selected_models_indices) == 0: 
                criteria = criteria + 0.05 
            
        
        self.selected_models_indices = selected_models_indices   

    
    def predict_xai(self, query): 
        # 1) define region of competence 
        self.get_region_of_competence(query) 

        # 2) estimate competence  
        competences = self.estimate_competence() 

        # 3) select models 
        self.select(competences) 

        final_prediction = 0 
        weight_total = 0 

        model_names = [] 
        individual_predictions = [] 
        weights = [] 
        selected_competences = [] 

        
        for i in self.selected_models_indices: 
            pred = get_value(self.pool_regressors[i].predict(query))  
            final_prediction += pred * (1/(competences[i] + 1e-10)) 
            weight_total += (1/(competences[i] + 1e-10)) 

            model_names.append(self.pool_regressors[i].__class__.__name__)
            individual_predictions.append(round(pred, 3)) 
            weights.append(round(1/competences[i], 3))
            selected_competences.append(round(competences[i], 3))

        final_prediction = final_prediction/weight_total

        contribution_dict = {"selected_models": model_names, 
                             "predictions": individual_predictions, 
                             "competences": selected_competences, 
                             "weights": weights}

        contribution_df = pd.DataFrame.from_dict(contribution_dict) 

        self.roc['target'] = self.roc_labels 
        neighbors_df = self.roc

        print(self.roc_labels)
        
        return final_prediction, contribution_df, neighbors_df



    def predict_single_sample(self, query): 
        # 1) define region of competence 
        self.get_region_of_competence(query) 

        # 2) estimate competence  
        competences = self.estimate_competence() 

        # 3) select models 
        self.select(competences) 

        # 4) predict 
        selected_models_competence = [] 
    

        final_prediction = 0 
        weight_total = 0 
        
        for i in self.selected_models_indices: 
            pred = get_value(self.pool_regressors[i].predict(query[self.selected_features]))  
            final_prediction += pred * (1/(competences[i] + 1e-10)) 
            weight_total += (1/(competences[i] + 1e-10)) 

        final_prediction = final_prediction/weight_total
        
        return final_prediction
            

    def predict(self, X, external_xai: bool = False):
        preds = []  
        external_contribution = [] 
        
        for i in range(X.shape[0]):
            query = X.iloc[[i]] 

            pred = self.predict_single_sample(query[self.selected_features])
            preds.append(pred) 
            external_contribution.append(self.selected_models_indices)

        if external_xai: 
            return preds, count_model_occurrences(external_contribution)
            
        return preds
    
    
    def score(self, X, y): 
        preds = self.predict(X) 
        
        return mean_squared_error(y, preds)



class DRS(BaseDES): 
    def __init__(self, pool_regressors=None, k=7, knn_metric='minkowski', metrics='mse', threshold=0.2, feature_selection=False):
        super(DRS, self).__init__(pool_regressors=pool_regressors, k=k, knn_metric=knn_metric, metrics=metrics, 
                                  threshold=threshold, feature_selection=feature_selection) 
        

    def fit(self, X_train=None, y_train=None, split=0.2, random_state=42):
        X_train = X_train.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)  

        features = X_train.columns.tolist() 

        X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=split, random_state=random_state) 

        self.X_dsel = X_dsel 
        self.y_dsel = y_dsel 

        if self.feature_selection: 
            # feature importance 
            model_rf = RandomForestRegressor(random_state=45)  
            selector = RFE(model_rf, step=1)
            selector = selector.fit(X_train, y_train) 
    
            self.selected_features = select(features, selector.support_) 
        else: 
            self.selected_features = features 

        # Train models 
        for model in self.pool_regressors: 
            model.fit(X_train[self.selected_features], y_train) 

    
    def select(self, competences):  
        index_min = np.argmin(competences)  
        
        self.selected_models_indices = [index_min]   


    def predict_single_sample(self, query): 
        # 1) define region of competence 
        self.get_region_of_competence(query) 

        # 2) estimate competence  
        competences = self.estimate_competence()


        # 3) select models 
        self.select(competences) 

        # 4) predict 

        final_prediction = 0 
        for i in self.selected_models_indices: 
            pred = get_value(self.pool_regressors[i].predict(query[self.selected_features]))  
            final_prediction += pred  

        final_prediction = final_prediction/len(self.selected_models_indices)
        
        return final_prediction
            

    def predict(self, X):
        preds = []  

        for i in range(X.shape[0]):
            query = X.iloc[[i]] 

            pred = self.predict_single_sample(query[self.selected_features])
            preds.append(pred) 
        
        return preds 
    

    def score(self, X, y): 
        preds = self.predict(X) 
        
        return mean_squared_error(y, preds)
