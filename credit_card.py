import numpy as np
from sklearn import preprocessing,metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import time

np.set_printoptions(threshold=np.nan,suppress=True)
pd.set_option('display.height', 10000)
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)

#define some basic classifiers
#logistic regression classifier
def logistic_regression_classifier(data_train, data_test, labels_train, labels_test):
    in_time = time.clock()
    print("Logistic Regression Classifier results : ")
    logit_model = LogisticRegression()
    #fit a model to the data
    logit_model.fit(data_train,labels_train)
    print(logit_model)
    # make predictions
    logit_predicted = logit_model.predict(data_test)
    #summarize the fit of the model
    print(metrics.classification_report(labels_test, logit_predicted))
    out_time = time.clock()
    print("Logistic Regression runtime: " + str(out_time-in_time))

#naive Bayes classifier
def naive_bayes_classifier(data_train, data_test, labels_train, labels_test):
    in_time = time.clock()
    print("Naive Bayes Classifier results : ")
    gaussian_model = GaussianNB()
    # fit a model to the data
    gaussian_model.fit(data_train,labels_train)
    print(gaussian_model)
    # make predictions
    gaussian_predicted = gaussian_model.predict(data_test)
    # summarize the fit of the model
    print(metrics.classification_report(labels_test, gaussian_predicted))
    out_time = time.clock()
    print("Naive Bayes runtime: " + str(out_time - in_time))


#Use a K-nn classifier
def knn_classifier(data_train, data_test, labels_train, labels_test):
    in_time = time.clock()
    print("K-nn Classifier results : ")
    knn_model = KNeighborsClassifier(n_neighbors=7)
    # fit a model to the data
    knn_model.fit(data_train,labels_train)
    print(knn_model)
    # make predictions
    knn_predicted = knn_model.predict(data_test)
    # summarize the fit of the model
    print(metrics.classification_report(labels_test, knn_predicted))
    out_time = time.clock()
    print("K-nn runtime: " + str(out_time - in_time))


#Use an SVM classifier
def svm_classifier(data_train, data_test, labels_train, labels_test):
    in_time = time.clock()
    print("SVM Classifier results : ")
    SVM_model = SVC(kernel='rbf')
    # fit a model to the data
    SVM_model.fit(data_train,labels_train)
    print(SVM_model)
    # make predictions
    svm_predicted = SVM_model.predict(data_test)
    # summarize the fit of the model
    print(metrics.classification_report(labels_test, svm_predicted))
    out_time = time.clock()
    print("SVM runtime: " + str(out_time - in_time))


if __name__ == '__main__':

    #read data as a CSV from the .DATA file provided
    df = pd.read_csv('crx.DATA', sep=',', header = None, skiprows=0)

    #rename the default columns from numbers to A1,A2...
    df.rename(index=str,columns={0: 'A1', 1: 'A2',2: 'A3',3: 'A4',4: 'A5',5: 'A6',6: 'A7',7: 'A8',8: 'A9',9: 'A10',
                                 10: 'A11',11: 'A12',12: 'A13',13: 'A14',14: 'A15',15 : 'A16'},inplace=True)

    #define columns which have categorical data and continous data
    category_columns = ['A1','A4','A5','A6','A7','A9','A10','A12','A13']
    continous_columns = ['A2','A3','A8','A11','A14','A15']

    #Observation - some of the attributes were found to have the character '?' instead of the class. All these records have been removed below.
    #observation - Categorical data is not present for all classes that have been mentioned in the document.
    #observation - No Data for 't' class in A4 attribute

    #Count of the rows in the raw data
    before_process = df.shape[0]
    #rename dataframe index
    df.index = range(df.shape[0])

    #remove all records having a '?'
    for col_index, col in df.iteritems():
        row_index = 0
        for element in col:
            if(str(element) == '?'):
                df.drop(row_index, axis=0, inplace=True)
            row_index += 1
        df.index = range(df.shape[0])

    after_process = df.shape[0]
    print("number of record removed due to incorrect data :" + str(before_process - after_process))

    #format the data for the next step
    df[category_columns] = df[category_columns].astype(str)
    df[continous_columns] = df[continous_columns].astype(float)
    df.A16 = df.A16.astype(str)


    #drop outliers for A11
    df.drop([116,44], axis =0,inplace=True)

    # drop outliers for A14
    df.drop(388, axis=0, inplace=True)

    # drop outliers for A15
    df.drop([6,29,304,65,127], axis=0, inplace=True)

    #shuffle the dataframe rows
    df.reindex(np.random.permutation(df.index))

    #remove the class labels from the data frame and store it a new data frame
    class_labels = df['A16']
    df.drop('A16',axis=1,inplace=True)

    #apply get_dummies to get the one hot encoding for all categorical data
    feature_df = pd.get_dummies(df,columns=category_columns)

    #observation - there are 41 classes in total from the 9 categorical attributes.
    #Since there is no data for class 't' of attribute 4, the one hot encoding of the categorical data results in replacing
    #the 9 attributes by 40 columns of features in stead of 41.
    #in total there are 46 features i.e. 41 from the categorical data and 6 from the continuous data

    #create a numpy array from the dataframe
    feature_np_array = feature_df.values
    features_rows,features_cols = feature_np_array.shape
    print("feature matrix size : " + str(feature_np_array.shape))

    #calcualte some basic stats about the unnormalized data
    A2 = feature_np_array[:,0:1]
    A3 = feature_np_array[:,1:2]
    A8 = feature_np_array[:,2:3]
    A11 = feature_np_array[:,3:4]
    A14 = feature_np_array[:,4:5]
    A15 = feature_np_array[:,5:6]

    A2_max = np.max(A2)
    A2_min = np.min(A2)
    A2_mean = np.mean(A2)

    A3_max = np.max(A3)
    A3_min = np.min(A3)
    A3_mean = np.mean(A3)

    A8_max = np.max(A8)
    A8_min = np.min(A8)
    A8_mean = np.mean(A8)

    A11_max = np.max(A11)
    A11_min = np.min(A11)
    A11_mean = np.mean(A11)

    A14_max = np.max(A14)
    A14_min = np.min(A14)
    A14_mean = np.mean(A14)

    A15_max = np.max(A15)
    A15_min = np.min(A15)
    A15_mean = np.mean(A15)

    #Plots
    plt.figure(1)
    plt.plot(range(features_rows),A2)
    plt.ylabel('A2')

    plt.figure(2)
    plt.plot(range(features_rows),A3)
    plt.ylabel('A3')

    plt.figure(3)
    plt.plot(range(features_rows),A8)
    plt.ylabel('A8')

    plt.figure(4)
    plt.plot(range(features_rows),A11)
    plt.ylabel('A11')

    plt.figure(5)
    plt.plot(range(features_rows), A14)
    plt.ylabel('A14')

    plt.figure(6)
    plt.plot(range(features_rows), A15)
    plt.ylabel('A15')
    #plt.show()

    #Stats about the continous data
    print("A2_max : " + str(A2_max))
    print("A2_min : " + str(A2_min))
    print("A2_mean : " + str(A2_mean))

    print("A3_max : " + str(A3_max))
    print("A3_min : " + str(A3_min))
    print("A3_mean : " + str(A3_mean))

    print("A8_max : " + str(A8_max))
    print("A8_min : " + str(A8_min))
    print("A8_mean : " + str(A8_mean))

    print("A11_max : " + str(A11_max))
    print("A11_min : " + str(A11_min))
    print("A11_mean : " + str(A11_mean))

    print("A14_max : " + str(A14_max))
    print("A14_min : " + str(A14_min))
    print("A14_mean : " + str(A14_mean))

    print("A15_max : " + str(A15_max))
    print("A15_min : " + str(A15_min))
    print("A15_mean : " + str(A15_mean))

    #apply various data normalization techniques to represent the continuous data features efficiently
    #the categorical data need not be normalized since there are in 0's and 1's

    # apply robust scaler
    robust_scaler = preprocessing.RobustScaler()
    feature_np_array_robust = robust_scaler.fit_transform(feature_np_array)

    # apply standard scaler
    standard_scaler = preprocessing.StandardScaler()
    feature_np_array_standard = standard_scaler.fit_transform(feature_np_array)

    # Normalize the data to a value between 0 and 1
    min_max_scaler = preprocessing.MinMaxScaler()
    feature_np_array_minmax = min_max_scaler.fit_transform(feature_np_array_robust)

    #apply the L2 norm and generate a new feature set
    feature_np_array_normalized_l2 = preprocessing.normalize(feature_np_array, norm='l2')

    #apply the L1 norm and generate a new feature set
    feature_np_array_normalized_l1 = preprocessing.normalize(feature_np_array, norm='l1')

    #split the processed feature set into training and test set
    data_train, data_test, labels_train, labels_test = train_test_split(feature_np_array_minmax, class_labels, test_size=0.20, random_state=42)


    logistic_regression_classifier(data_train, data_test, labels_train, labels_test)
    naive_bayes_classifier(data_train, data_test, labels_train, labels_test)
    knn_classifier(data_train, data_test, labels_train, labels_test)
    svm_classifier(data_train, data_test, labels_train, labels_test)


#################################################################################################################################################################