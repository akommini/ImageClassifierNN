# Import python modules
import numpy as np
import kaggle
import csv
from sklearn.metrics import accuracy_score
#import time
from sklearn import neighbors
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsClassifier

# Read in train and test data
def read_image_data():
	print('Reading image data ...')
	train_x = np.load('../../Data/data_train.npy')
	train_y = np.load('../../Data/train_labels.npy')
	test_x = np.load('../../Data/data_test.npy')

	return (train_x, train_y, test_x)

# Decision TRee Question 1
def DecisionTree_reg(train_x, train_y, test_x,tree_depth):
    # Question 1 decision tree
    # Creating kfold crossvalidation indices using KFold
    # 5 different nearest neighbors regressors using the following number of {3, 5, 10, 20, 25}
    #tree_depth=[3, 6, 9, 12, 15, 20, 25, 30, 35, 40]
    #tim_arr=[]
    # mean error to model select
    mean_err=np.ones((len(tree_depth),1))
    #mean_err_CV=np.ones((len(tree_depth),1))
    for i in range(len(tree_depth)):
        dec_Rig =  DecisionTreeClassifier(max_depth=tree_depth[i],criterion="entropy")    
        # intializing error array to calculate the average
        error=[];
        #t = time.process_time();
        #y_pred_CV = cross_validate(dec_Rig,train_x, train_y, cv=5,return_train_score=True)
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            ytest_pre = dec_Rig.fit(X_train, y_train).predict(X_test)
            error.append(accuracy_score(ytest_pre, y_test));
        #end = time.process_time()-t;
        mean_err[i]=np.abs(error).mean()
        #mean_err_CV[i]=y_pred_CV['test_score'].mean()
        #tim_arr.append(end)
    index_min = np.argmax(mean_err)
    #index_min_CV = np.argmin(mean_err_CV)
    
    print('Decision tree depth with minimum error for 5-fold cross validation is:',tree_depth[index_min])
    #print('Decision tree depth with minimum error for 5-fold cross validation is:',tree_depth[index_min_CV])
    # Training with the total data set with optimal no of neighbors
    dec_Rig =  DecisionTreeClassifier(max_depth=tree_depth[index_min],criterion="entropy")
    ytrain_full = dec_Rig.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=accuracy_score(ytrain_full, train_y)
    print('Test accuracy of Decision tree=%0.4f' % err_trainfull_knn)
    y_predict=dec_Rig.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)

def knn_regression(train_x, train_y, test_x, no_neighbors):
    # Question 3 Nearest Neighbors analysis

    # mean error to model select
    mean_err=np.ones((len(no_neighbors),1))
    for i in range(len(no_neighbors)):
        # knn defined using scikit learn
        knn = KNeighborsClassifier(no_neighbors[i], weights='uniform',n_jobs=18)    
        # intializing error array to calculate the average
        error=[];
        #mean_err_CV=np.ones((len(tree_depth),1))
        #t = time.process_time();
        # cross validation to train data
        #y_pred_CV = cross_validate(knn,train_x, train_y, cv=5,return_train_score=True)
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            # prediction for the test data
            ytest_pre = knn.fit(X_train, y_train).predict(X_test)
            error.append(accuracy_score(ytest_pre, y_test));
        #end = time.process_time()-t;
        #mean eror for all the cross validation sets
        mean_err[i]=np.abs(error).mean()
        #mean_err_CV[i]=y_pred_CV['test_score'].mean()
        #tim_arr.append(end)
    # minimum mean error to select the best model
    index_min = np.argmax(mean_err)
    #index_min_CV = np.argmin(mean_err_CV)
    print('Number of neighbors with minimum error for 5-fold cross validation is:',no_neighbors[index_min])
    #print('Number of neighbors with minimum error for 5-fold cross validation is:',no_neighbors[index_min_CV])
    # Training with the total data set with optimal no of neighbors
    knn = KNeighborsClassifier(no_neighbors[index_min], weights='uniform',n_jobs=18)
    ytrain_full = knn.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=accuracy_score(ytrain_full, train_y)
    print('Test accuracy of Decision tree=%0.4f' % err_trainfull_knn)
    # Prediction for the final test data
    y_predict=knn.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)

def linearClass_hinge(train_x, train_y, test_x, alpha_arr):
    # Question 3 Hinge regression
    #alpha_arr=[10**-6,10**-4,10**-2,1,10]
    # mean error to model select
    #tim_arr=[]
    mean_err=np.ones((len(alpha_arr),1))
    
    for i in range(len(alpha_arr)):
        # Lasso regression defined using scikit learn
        clf = linear_model.SGDClassifier(max_iter=3000, tol=1e-3, alpha=alpha_arr[i],n_jobs=18)   
        # intializing error array to calculate the average
        error=[];
        #mean_err_CV=np.ones((len(alpha_arr),1))
        #t = time.process_time();
        # cross validation to train data
        #y_pred_CV = cross_validate(clf,train_x, train_y, cv=5,return_train_score=True)
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            ytest_pre = clf.fit(X_train, y_train).predict(X_test)
            # prediction for the test data
            error.append(accuracy_score(ytest_pre, y_test));
            #end = time.process_time()-t;
            #mean eror for all the cross validation sets
        mean_err[i]=np.abs(error).mean()
        #mean_err_CV[i]=y_pred_CV['test_score'].mean()
        #tim_arr.append(end)
     # minimum mean error to select the best model     
    index_min = np.argmax(mean_err)
    print('Alpha for Ridge with minimum error for 5-fold cross validation is:',alpha_arr[index_min])
    # Training with the total data set with optimal alpha
    clf = linear_model.SGDClassifier(max_iter=3000, tol=1e-3,alpha=alpha_arr[index_min],n_jobs=18)
    ytrain_full = clf.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=accuracy_score(ytrain_full, train_y)
    print('Test accuracy of linear model with hinge loss=%0.4f' % err_trainfull_knn)
    # Prediction for the final test data
    y_predict=clf.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)

def linearClass_logLoss(train_x, train_y, test_x, alpha_arr):
    # Question 3 Log loss linear classifier
    #alpha_arr=[10**-6,10**-4,10**-2,1,10]
    # mean error to model select
    #tim_arr=[]
    mean_err=np.ones((len(alpha_arr),1))
    
    for i in range(len(alpha_arr)):
        # Lasso regression defined using scikit learn
        clf = linear_model.SGDClassifier(max_iter=3000, tol=1e-3, alpha=alpha_arr[i],n_jobs=18,loss='log')   
        # intializing error array to calculate the average
        error=[];
        #mean_err_CV=np.ones((len(alpha_arr),1))
        #t = time.process_time();
        # cross validation to train data
        #y_pred_CV = cross_validate(clf,train_x, train_y, cv=5,return_train_score=True)
        for train_index, test_index in kf.split(train_x):
            #print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = train_x[train_index], train_x[test_index]
            y_train, y_test = train_y[train_index], train_y[test_index]
            ytest_pre = clf.fit(X_train, y_train).predict(X_test)
            # prediction for the test data
            error.append(accuracy_score(ytest_pre, y_test));
            #end = time.process_time()-t;
            #mean eror for all the cross validation sets
        mean_err[i]=np.abs(error).mean()
        #mean_err_CV[i]=y_pred_CV['test_score'].mean()
        #tim_arr.append(end)
     # minimum mean error to select the best model     
    index_min = np.argmax(mean_err)
    print('Alpha for Ridge with minimum error for 5-fold cross validation is:',alpha_arr[index_min])
    # Training with the total data set with optimal alpha
    clf = linear_model.SGDClassifier(max_iter=3000, tol=1e-3,alpha=alpha_arr[index_min],n_jobs=18,loss='log')
    ytrain_full = clf.fit(train_x, train_y).predict(train_x)
    err_trainfull_knn=accuracy_score(ytrain_full, train_y)
    print('Test accuracy of Linear model with log loss=%0.4f' % err_trainfull_knn)
    # Prediction for the final test data
    y_predict=clf.predict(test_x)
    return (err_trainfull_knn, y_predict,mean_err,index_min)

############################################################################
# Creating kfold crossvalidation indices using KFold
kf = KFold(n_splits=5)


train_x, train_y, test_x = read_image_data()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Decision tree
tree_depth=[3, 6, 9, 12, 15]
# 5 different nearest neighbors regressors using the following number of {3, 5, 10, 20, 25}
no_neighbors=[3, 5, 7, 9, 11]
# regularization constants used
alpha_arr=[10**-6,10**-4,10**-2,1,10]
#dec_Rig =  DecisionTreeClassifier(max_depth=3)
#y_pred_pre = cross_val_predict(dec_Rig,train_x, train_y, cv=3)
err_trainfull_decR_airfoil, y_predict_decR_airfoil,mean_err_decR_airfoil,index_min_decR_airfoil=DecisionTree_reg(train_x, train_y, test_x, tree_depth);
err_trainfull_knn_airquality, y_predict_knn_airquality,mean_err_airquality,index_min_airquality=knn_regression(train_x, train_y, test_x);
err_trainfull_linear_hinge, y_predict_linear_hinge,mean_err_linear_hinge,index_min_linear_hinge=linearClass_hinge(train_x, train_y, test_x,alpha_arr);
err_trainfull_linear_logLoss, y_predict_linear_logLoss,mean_err_linear_logLoss,index_min_linear_logLoss=linearClass_logLoss(train_x, train_y, test_x,alpha_arr);


# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(y_predict_knn_airquality, file_name)
