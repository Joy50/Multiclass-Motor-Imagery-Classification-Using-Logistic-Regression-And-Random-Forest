import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score

from cspA import generate_projectionA,generate_eyeA,extract_featureA
from cspB import generate_projectionB,generate_eyeB,extract_featureB
from get_dataA import get_dataA
from get_dataB import get_dataB
from filtersA import load_filterbankA
from filtersB import load_filterbankB

class CSP_ModelA():
    def __init__(self):
        self.crossvalidation = False
        self.data_path 	= 'ADatasets/'
        self.useCSP = True
        self.NO_splits = 5 # number of folds in cross validation
        self.fs = 250. # sampling frequency
        self.NO_channels = 22 # number of EEG channels
        self.NO_subjects = 9
        self.NO_csp = 24 # Total number of CSP feature per band and timewindow
        self.bw = np.array([2,4,8,16,32]) # bandwidth of filtered signals
        self.ftype = 'butter' # 'fir', 'butter'
        self.forder= 2 # 4
        self.filter_bank = load_filterbankA(self.bw,self.fs,order=self.forder,max_freq=40,ftype = self.ftype) # get filterbank coeffs
        time_windows_flt = np.array([[2.5,3.5],[3,4],[3.5,4.5],[4,5],[4.5,5.5],[5,6],[2.5,4.5],[3,5],[3.5,5.5],[4,6],[2.5,6]])*self.fs # time windows in [s] x fs for using as a feature
        self.time_windows = time_windows_flt.astype(int)
        self.NO_bands = self.filter_bank.shape[0]
        self.NO_time_windows = int(self.time_windows.size/2)
        self.NO_features = self.NO_csp*self.NO_bands*self.NO_time_windows
        self.train_time = 0
        self.train_trials = 0
        self.eval_time = 0
        self.eval_trials = 0
    def run_csp(self):
        ################################ Training ############################################################################
        start_train = time.time()
        # 1. Apply CSP to bands to get spatial filter
        if self.useCSP:
            w = generate_projectionA(self.train_data,self.train_label, self.NO_csp,self.filter_bank,self.time_windows)
        else:
            w = generate_eyeA(self.train_data,self.train_label,self.filter_bank,self.time_windows)
        # 2. Extract features for training
        resulted_feature_matrix = extract_featureA(self.train_data,w,self.filter_bank,self.time_windows)

        clf = LogisticRegression()
        clf.fit(resulted_feature_matrix,self.train_label)

        #kappa_for_logistic = cohen_kappa_score(self.eval_data,self.eval_label)
        model = RandomForestClassifier()
        model.fit(resulted_feature_matrix,self.train_label)
        end_train = time.time()
        self.train_time += end_train-start_train
        self.train_trials += len(self.train_label)
        ################################# Evaluation ###################################################
        start_eval = time.time()
        eval_feature_mat = extract_featureA(self.eval_data,w,self.filter_bank,self.time_windows)
        success_rate = clf.score(eval_feature_mat,self.eval_label)
        Success_rate_Random = model.score(eval_feature_mat,self.eval_label)
        end_eval = time.time()
        self.eval_time += end_eval-start_eval
        self.eval_trials += len(self.eval_label)

        return success_rate,Success_rate_Random
    def load_data(self):
        if self.crossvalidation:
            data,label = get_dataA(self.subject,True,self.data_path)
            kf = KFold(n_splits=self.NO_splits)
            split = 0
            for train_index, test_index in kf.split(data):
                if self.split == split:
                    self.train_data = data[train_index]
                    self.train_label = label[train_index]
                    self.eval_data = data[test_index]
                    self.eval_label = label[test_index]
                    split += 1
        else:
            self.train_data,self.train_label = get_dataA(self.subject,True,self.data_path)
            self.eval_data,self.eval_label = get_dataA(self.subject,False,self.data_path)
class CSP_ModelB():
    def __init__(self):
        self.crossvalidation = False
        self.data_path 	= 'BDatasets/'
        self.useCSP = True
        self.NO_splits = 5 # number of folds in cross validation
        self.fs = 250. # sampling frequency
        self.NO_channels = 3 # number of EEG channels
        self.NO_subjects = 9
        self.NO_csp = 24 # Total number of CSP feature per band and timewindow
        self.bw = np.array([2,4,8,16,32]) # bandwidth of filtered signals
        self.ftype = 'butter' # 'fir', 'butter'
        self.forder= 2 # 4
        self.filter_bank = load_filterbankB(self.bw,self.fs,order=self.forder,max_freq=40,ftype = self.ftype) # get filterbank coeffs
        time_windows_flt = np.array([[2.5,3.5],[3,4],[3.5,4.5],[4,5],[4.5,5.5],[5,6],[2.5,4.5],[3,5],[3.5,5.5],[4,6],[2.5,6]])*self.fs # time windows in [s] x fs for using as a feature
        self.time_windows = time_windows_flt.astype(int)
        self.NO_bands = self.filter_bank.shape[0]
        self.NO_time_windows = int(self.time_windows.size/2)
        self.NO_features = self.NO_csp*self.NO_bands*self.NO_time_windows
        self.train_time = 0
        self.train_trials = 0
        self.eval_time = 0
        self.eval_trials = 0
    def run_csp(self):
        ################################ Training ############################################################################
        start_train = time.time()
        # 1. Apply CSP to bands to get spatial filter
        if self.useCSP:
            w = generate_projectionB(self.train_data,self.train_label, self.NO_csp,self.filter_bank,self.time_windows)
        else:
            w = generate_eyeB(self.train_data,self.train_label,self.filter_bank,self.time_windows)
        # 2. Extract features for training
        resulted_feature_matrix = extract_featureB(self.train_data,w,self.filter_bank,self.time_windows)

        clf = LogisticRegression(solver='liblinear')
        clf.fit(resulted_feature_matrix,self.train_label)

        #kappa_for_logistic = cohen_kappa_score(self.eval_data,self.eval_label)
        model = RandomForestClassifier()
        model.fit(resulted_feature_matrix,self.train_label)
        end_train = time.time()
        self.train_time += end_train-start_train
        self.train_trials += len(self.train_label)
        ################################# Evaluation ###################################################
        start_eval = time.time()
        eval_feature_mat = extract_featureB(self.eval_data,w,self.filter_bank,self.time_windows)
        success_rate = clf.score(eval_feature_mat,self.eval_label)
        Success_rate_Random = model.score(eval_feature_mat,self.eval_label)
        end_eval = time.time()
        self.eval_time += end_eval-start_eval
        self.eval_trials += len(self.eval_label)

        return success_rate,Success_rate_Random
    def load_dataB(self):
        if self.crossvalidation:
            data,label = get_dataB(self.subject,True,self.data_path)
            kf = KFold(n_splits=self.NO_splits)
            split = 0
            for train_index, test_index in kf.split(data):
                if self.split == split:
                    self.train_data = data[train_index]
                    self.train_label = label[train_index]
                    self.eval_data = data[test_index]
                    self.eval_label = label[test_index]
                    split += 1
        else:
            self.train_data,self.train_label = get_dataB(self.subject,True,self.data_path)
            self.eval_data,self.eval_label = get_dataB(self.subject,False,self.data_path)

def mainA():
    model = CSP_ModelA()
    print("Number of used features: "+ str(model.NO_features))
    # success rate sum over all subjects
    success_tot_sum = 0
    success_tot_sum_rand = 0
    if model.crossvalidation:
        print("Cross validation run")
    else:
        print("Test data set")

    start = time.time()

    for model.subject in range(1,model.NO_subjects+1):
        if model.crossvalidation:
            success_sub_sum = 0
            success_sub_sum_rand = 0
            for model.split in range(model.NO_splits):
                model.load_data()
                logistic_Success,random_success = model.run_csp()
                success_sub_sum += logistic_Success
                success_sub_sum_rand += random_success
                print('for Logistic Regression :',success_sub_sum/(model.split+1))
                print('for Random Forest classifier:',success_sub_sum_rand/(model.split+1))
            # average over all splits
            success_rate = success_sub_sum/model.NO_splits
            success_rate_rand = success_sub_sum_rand/model.NO_splits
        else:
            model.load_data()
            success_rate,success_rate_rand = model.run_csp()
        print('For Logistic regression:',success_rate)
        print('For Random Forest classifier',success_rate_rand)
        success_tot_sum += success_rate
        success_tot_sum_rand += success_rate_rand
    # Average success rate over all subjects
    print(model.subject," 's Average success rate for Logistic regression" + str(success_tot_sum/model.NO_subjects))
    print(model.subject," 's Average Success rate for random Forest"+str(success_tot_sum_rand/model.NO_subjects))

    print("Training average time: " +  str(model.train_time/model.NO_subjects))
    print("Evaluation average time: " +  str(model.eval_time/model.NO_subjects))
    end = time.time()
    print("Time elapsed [s] " + str(end - start))
def mainB():
    model = CSP_ModelB()
    print("Number of used features: "+ str(model.NO_features))
    # success rate sum over all subjects
    success_tot_sum = 0
    success_tot_sum_rand = 0
    if model.crossvalidation:
        print("Cross validation run")
    else:
        print("Test data set")

    start = time.time()

    for model.subject in range(1,model.NO_subjects+1):
        if model.crossvalidation:
            success_sub_sum = 0
            success_sub_sum_rand = 0
            for model.split in range(model.NO_splits):
                model.load_dataB()
                logistic_Success,random_success = model.run_csp()
                success_sub_sum += logistic_Success
                success_sub_sum_rand += random_success
                print('for Logistic Regression :',success_sub_sum/(model.split+1))
                print('for Random Forest classifier:',success_sub_sum_rand/(model.split+1))
            # average over all splits
            success_rate = success_sub_sum/model.NO_splits
            success_rate_rand = success_sub_sum_rand/model.NO_splits
        else:
            model.load_dataB()
            success_rate,success_rate_rand = model.run_csp()
        print('For Logistic regression:',success_rate)
        print('For Random Forest classifier',success_rate_rand)
        success_tot_sum += success_rate
        success_tot_sum_rand += success_rate_rand
    # Average success rate over all subjects
    print(model.subject," 's Average success rate for Logistic regression" + str(success_tot_sum/model.NO_subjects))
    print(model.subject," 's Average Success rate for random Forest"+str(success_tot_sum_rand/model.NO_subjects))

    print("Training average time: " +  str(model.train_time/model.NO_subjects))
    print("Evaluation average time: " +  str(model.eval_time/model.NO_subjects))
    end = time.time()
    print("Time elapsed [s] " + str(end - start))
def datasetA():
    mainA()
def datasetB():
    mainB()
if __name__ == '__main__':
    datasetA()
    #datasetB()
