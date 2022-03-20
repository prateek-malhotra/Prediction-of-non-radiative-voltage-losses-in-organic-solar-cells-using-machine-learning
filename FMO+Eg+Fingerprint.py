import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from scipy.stats import pearsonr
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneOut 
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")

# Importing .csv files
# Choose any one file and commentline all others
a = pd.read_csv('FMO+Eg+MACCS.csv')
# a = pd.read_csv('FMO+Eg+Extended FP.csv')
# a = pd.read_csv('FMO+Eg+Pubchem.csv')
# a = pd.read_csv('FMO+Eg+Morgan.csv')

#Creating and transforming descriptors with their median values for each distinct donor and acceptor
a["Dh"] = a.groupby('Donor')["Dhomo"].transform('median')           # Creating new descriptors
a["Dl"] = a.groupby('Donor')["Dlumo"].transform('median')
a["Ah"] = a.groupby('Acceptor')["Ahomo"].transform('median')
a["Al"] = a.groupby('Acceptor')["Alumo"].transform('median')
a["Eg"] = a.groupby('Small Eg')["EgOpt"].transform('median')
a['Hoffset']=a['Ah']-a['Dh']
a['Loffset']=a['Al']-a['Dl']
a['Toffset']=a['Hoffset']+a['Loffset']
a['Ei']=a['Dh']-a['Al']
a['Eg-Ei']=a['Eg']-a['Ei']
a['Loss_per'] = a['L3']/a['Eg']*100              # Defining target variable 
a.head()

#For Reported use only 154 otherwise continue with extended data
a=a[:154]  
a

# shuffling the dataset
from sklearn.utils import shuffle
a = shuffle(a, random_state=42)

# Feature Engineering
ab=a.drop(['Donor', 'Acceptor', 'Donor Smiles', 'Acceptor Smiles', 'Dhomo','Dlumo', 'Ahomo', 'Alumo', 
           'Small Eg', 'EgOpt','L3','Dh','Dl','Ah','Al','Eg','Hoffset','Loffset','Toffset','Jsc','Voc','FF','PCE','Ei','Eg-Ei','Loss_per'],axis=1)
print('Total features :',len(ab.columns))
ab=ab.select_dtypes(exclude=object)
ab= ab.replace([np.inf, -np.inf], np.nan)
b=ab.dropna()
c = b.replace(np.nan,0)
print('Features with zero standard deviation :',len(c.std()[c.std() == 0]))
d=c.drop(c.std()[c.std() == 0].index.values, axis=1)   #To remove columns with std deviation equals to zero

from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
a1=sel.fit_transform(d)
sel.get_support()
features = d.columns[sel.get_support()]
print('Features selected by using variance threshold of 0.8 :',len(features))
f = features
a2=d[f]

a3=a[['Dh','Dl','Ah','Al','Eg','Hoffset','Loffset','Ei','Eg-Ei','Loss_per']]  # For FMO+Eg+Fingerprints use this code and coomentline line 67

# a3=a[['Loss_per']]                 # For only fingerprint use this line of code and commentline line 68

#Preparing dataset for LOOCV

frames =[a2, a3]
loocv_df = pd.concat(frames, axis =1)

Xr = loocv_df.drop(['Loss_per'],axis=1)
yr = loocv_df['Loss_per'].copy()
features = len(Xr.columns)
print('Features left :',features)

# Creating a pipeline for feature scaling

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([('std_scaler', StandardScaler())])
Xr1 = my_pipeline.fit_transform(Xr)

X_array = np.array(Xr1)
y_array = np.array(yr)

loocv_df

# Defining Metrics
def R2Score(X,Y):
    r2=metrics.r2_score(X,Y)
    return r2
def PearsonCoefficient(X, Y):
    corr, _ = pearsonr(X,Y)
    return corr
def MSE(X,Y):
    mse=mean_squared_error(X,Y)
    return mse
def RMSE(X,Y):
    rmse=np.sqrt(mean_squared_error(X,Y))
    return rmse
def MAPE(X,Y):
    mape=np.average(abs(np.array(X)-np.array(Y))/np.array(Y))*100
    return mape

#Tuning SVM using GridSearchCV
from sklearn.model_selection import GridSearchCV
my_cv = LeaveOneOut()
param_grid = {'C': [10,20,30,40,50,60,70,80,90,100], 
'epsilon':[0.1,0.01,0.001]}

nn = model1 = SVR()
tune_SVM = GridSearchCV(estimator=nn, param_grid=param_grid, scoring='neg_mean_squared_error',cv=my_cv)
tune_SVM.fit(X_array,y_array)
# print clf.best_score_
tune_SVM.best_params_

# Results
loo = LeaveOneOut()
ytests = []
y_pred_list_RF = []
y_pred_list_GB = []
y_pred_list_SVM = []
y_pred_list_ANN = []
for train_idx, test_idx in loo.split(Xr):
    X_train_loocv, X_test_loocv = X_array[train_idx], X_array[test_idx] #requires arrays
    y_train_loocv, y_test_loocv = y_array[train_idx], y_array[test_idx]
    
    model_RF = RandomForestRegressor(random_state=42).fit(X_train_loocv, y_train_loocv) 
    y_pred_RF = model_RF.predict(X_test_loocv)
    # there is only one y-test and y-pred per iteration over the loo.split, 
    # so to get a proper graph, we append them to respective lists.
    ytests += list(y_test_loocv)
    y_pred_list_RF += list(y_pred_RF)
    model_GB = GradientBoostingRegressor(random_state=42).fit(X_train_loocv, y_train_loocv) 
    y_pred_GB = model_GB.predict(X_test_loocv)
    y_pred_list_GB += list(y_pred_GB)
    model_SVM = SVR(**tune_SVM.best_params_).fit(X_train_loocv, y_train_loocv) 
    y_pred_SVM = model_SVM.predict(X_test_loocv)
    y_pred_list_SVM += list(y_pred_SVM)
    model_ANN = MLPRegressor(solver='lbfgs',random_state=42,max_iter=200,tol=1e-10,hidden_layer_sizes=[features],
                          activation='relu',alpha=10).fit(X_train_loocv, y_train_loocv)
    y_pred_ANN = model_ANN.predict(X_test_loocv)
    y_pred_list_ANN += list(y_pred_ANN)
    
print('RF Results')    

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print('R2Score',R2Score(ytests,y_pred_list_RF))
print('PearsonCoefficient',PearsonCoefficient(ytests,y_pred_list_RF))
print('MSE',MSE(ytests,y_pred_list_RF))
print('RMSE',RMSE(ytests,y_pred_list_RF))
print('MAPE',MAPE(ytests,y_pred_list_RF))
# sns.jointplot(x=ytests, y=y_pred_list_RF, data=a, kind='reg', stat_func=pearsonr)
# sns.jointplot(x=ytests, y=y_pred_list_RF, data=a, kind='reg', stat_func=r2_score)

print('\n\nGB Results')

print('R2Score',R2Score(ytests,y_pred_list_GB))
print('PearsonCoefficient',PearsonCoefficient(ytests,y_pred_list_GB))
print('MSE',MSE(ytests,y_pred_list_GB))
print('RMSE',RMSE(ytests,y_pred_list_GB))
print('MAPE',MAPE(ytests,y_pred_list_GB))
# sns.jointplot(x=ytests, y=y_pred_list_GB, data=a, kind='reg', stat_func=pearsonr)
# sns.jointplot(x=ytests, y=yp_red_list_GB, data=a, kind='reg', stat_func=r2_score) 

print('\n\nSVM Results')

print('R2Score',R2Score(ytests,y_pred_list_SVM))
print('PearsonCoefficient',PearsonCoefficient(ytests,y_pred_list_SVM))
print('MSE',MSE(ytests,y_pred_list_SVM))
print('RMSE',RMSE(ytests,y_pred_list_SVM))
print('MAPE',MAPE(ytests,y_pred_list_SVM))
# sns.jointplot(x=ytests, y=y_pred_list_SVM, data=a, kind='reg', stat_func=pearsonr)
# sns.jointplot(x=ytests, y=yp_red_list_SVM, data=a, kind='reg', stat_func=r2_score) 

print('\n\nANN Results')

print('R2Score',R2Score(ytests,y_pred_list_ANN))
print('PearsonCoefficient',PearsonCoefficient(ytests,y_pred_list_ANN))
print('MSE',MSE(ytests,y_pred_list_ANN))
print('RMSE',RMSE(ytests,y_pred_list_ANN))
print('MAPE',MAPE(ytests,y_pred_list_ANN))
# sns.jointplot(x=ytests, y=y_pred_list_ANN, data=a, kind='reg', stat_func=pearsonr)
# sns.jointplot(x=ytests, y=yp_red_list_ANN, data=a, kind='reg', stat_func=r2_score) 

# Result Dataframe
print('\n\nResults Dataframe\n')
result_data = [{'R2Score': R2Score(ytests,y_pred_list_RF),'PearsonCoefficient': PearsonCoefficient(ytests,y_pred_list_RF),
                'MSE': MSE(ytests,y_pred_list_RF),'RMSE': RMSE(ytests,y_pred_list_RF),'MAPE': MAPE(ytests,y_pred_list_RF)},
                {'R2Score': R2Score(ytests,y_pred_list_GB),'PearsonCoefficient': PearsonCoefficient(ytests,y_pred_list_GB),
                 'MSE': MSE(ytests,y_pred_list_GB),'RMSE': RMSE(ytests,y_pred_list_GB),'MAPE': MAPE(ytests,y_pred_list_GB)},
                {'R2Score': R2Score(ytests,y_pred_list_SVM),'PearsonCoefficient': PearsonCoefficient(ytests,y_pred_list_SVM),
                 'MSE': MSE(ytests,y_pred_list_SVM),'RMSE': RMSE(ytests,y_pred_list_SVM),'MAPE': MAPE(ytests,y_pred_list_SVM)},
                {'R2Score': R2Score(ytests,y_pred_list_ANN),'PearsonCoefficient': PearsonCoefficient(ytests,y_pred_list_ANN),
                 'MSE': MSE(ytests,y_pred_list_ANN),'RMSE': RMSE(ytests,y_pred_list_ANN),'MAPE': MAPE(ytests,y_pred_list_ANN)}] 
result_df = pd.DataFrame(result_data, index=['RF','GB','SVM','ANN'])
print(result_df)



