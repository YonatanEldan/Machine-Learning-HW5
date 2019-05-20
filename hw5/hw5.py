from numpy import count_nonzero, logical_and, logical_or, concatenate, mean, array_split, poly1d, polyfit, array
from numpy.random import permutation
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt


SVM_DEFAULT_DEGREE = 3
SVM_DEFAULT_GAMMA = 'auto'
SVM_DEFAULT_C = 1.0
ALPHA = 1.5


def prepare_data(data, labels, max_count=None, train_ratio=0.8):
    """
    :param data: a numpy array with the features dataset
    :param labels:  a numpy array with the labels
    :param max_count: max amout of samples to work on. can be used for testing
    :param train_ratio: ratio of samples used for train
    :return: train_data: a numpy array with the features dataset - for train
             train_labels: a numpy array with the labels - for train
             test_data: a numpy array with the features dataset - for test
             test_labels: a numpy array with the features dataset - for test
    """
    
    if max_count:
        data = data[:max_count]
        labels = labels[:max_count]

    # combining labels and features in order to shuffle the dataset
    combined_data = concatenate((data, array([labels]).T), axis=1)
    combined_data = permutation(combined_data)

    # spliiting dataset to train and test data based on train_ratio parameter
    length = (len(data[:,0]))
    splitIndex = int(train_ratio*length)
    training, test = combined_data[:splitIndex,:], combined_data[splitIndex:,:]

    # splitting to training and testins features and labels
    train_data = training[:, :-1]
    train_labels = training[:, -1]
    test_data = test[:, :-1]
    test_labels = test[:, -1]

    return train_data, train_labels, test_data, test_labels


def get_stats(prediction, labels):
    """
    :param prediction: a numpy array with the prediction of the model
    :param labels: a numpy array with the target values (labels)
    :return: tpr: true positive rate
             fpr: false positive rate
             accuracy: accuracy of the model given the predictions
    """
    length = len(prediction)
    
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0
    for i in (range(length)):
        if(prediction[i]==1):
            if(labels[i]==1):
                tpr+=1 
                accuracy+=1
            else: 
              fpr+=1
        else:
            if(labels[i]==0):
                accuracy+=1
    tpr /= count_nonzero(labels)
    fpr /= length - count_nonzero(labels)
    accuracy /= length

    return tpr, fpr, accuracy


def get_k_fold_stats(folds_array, labels_array, clf):
    """
    :param folds_array: a k-folds arrays based on a dataset with M features and N samples
    :param labels_array: a k-folds labels array based on the same dataset
    :param clf: the configured SVC learner
    :return: mean(tpr), mean(fpr), mean(accuracy) - means across all folds
    """

    tpr = []
    fpr = []
    accuracy = []



    for i in (range(len(folds_array))):
        testData = folds_array.pop(0)
        labels = labels_array.pop(0)
        X , y= concatenate((folds_array)),concatenate((labels_array))
        clf.fit(X, y)
        prediction = clf.predict(testData)
        tempTpr, tempFpr, tempAccuracy = get_stats(prediction, labels)

        folds_array.append(testData)
        labels_array.append(labels)
        
        #append the results to the arrays
        tpr.append(tempTpr)
        fpr.append(tempFpr)
        accuracy.append(tempAccuracy)


    return mean(tpr), mean(fpr), mean(accuracy)


def compare_svms(data_array,
                 labels_array,
                 folds_count,
                 kernels_list=('poly', 'poly', 'poly', 'rbf', 'rbf', 'rbf',),
                 kernel_params=({'degree': 2}, {'degree': 3}, {'degree': 4}, {'gamma': 0.005}, {'gamma': 0.05}, {'gamma': 0.5},)):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :param kernels_list: a list of strings defining the SVM kernels
    :param kernel_params: a dictionary with kernel parameters - degree, gamma, c
    :return: svm_df: a dataframe containing the results as described below
    """
    svm_df = pd.DataFrame()
    svm_df['kernel'] = kernels_list
    svm_df['kernel_params'] = kernel_params
    svm_df['tpr'] = None
    svm_df['fpr'] = None
    svm_df['accuracy'] = None

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    tpr = []
    fpr = []
    accuracy = []
    
    
    # split the data and the labels arrays into k subarrays based on the folds_count param
    folds_array, labels_folds_array = array_split(data_array, folds_count), array_split(labels_array, folds_count)

    # generates a SVM based in the default parameters
    clf = SVC(C = SVM_DEFAULT_C, gamma = SVM_DEFAULT_GAMMA, degree = SVM_DEFAULT_DEGREE)
    for i in range(len(kernels_list)):
        
        #reset to the defult parameters
        clf.set_params(**{'C' : SVM_DEFAULT_C, 'gamma' : SVM_DEFAULT_GAMMA, 'degree' : SVM_DEFAULT_DEGREE, 'kernel' :kernels_list[i]})

        # change the specific parameter based on kernel_params 
        clf.set_params(**kernel_params[i])

        #get the results using get_k_fold_stats function and append to the arrays
        tempTpr, tempFpr, tempAccuracy = get_k_fold_stats(folds_array, labels_folds_array, clf)
        tpr.append(tempTpr)
        fpr.append(tempFpr)
        accuracy.append(tempAccuracy)

    svm_df['tpr'] = tpr
    svm_df['fpr'] = fpr
    svm_df['accuracy'] = accuracy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm_df


def get_most_accurate_kernel():
    """
    :return: integer representing the row number of the most accurate kernel
    """
    best_kernel = 0
    return best_kernel


def get_kernel_with_highest_score():
    """
    :return: integer representing the row number of the kernel with the highest score
    """
    best_kernel = 0
    return best_kernel


def plot_roc_curve_with_score(df, alpha_slope=1.5):
    """
    :param df: a dataframe containing the results of compare_svms
    :param alpha_slope: alpha parameter for plotting the linear score line
    :return:
    """
    x = df.fpr.tolist()
    y = df.tpr.tolist()

    b = -1 * (alpha_slope*(x[get_kernel_with_highest_score()])) + y[get_kernel_with_highest_score()]
    linearP = poly1d([alpha_slope,b])
    #curveP = poly1d(polyfit(x,y,3))
    plt.title('ROC plot')
    plt.plot([0,1],linearP([0,1]),'-b')
    #plt.plot(x,curveP(x),'--b')
    plt.plot(x, y, 'ro')
    plt.xlim([0, 1.1])
    plt.ylim([0, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def evaluate_c_param(data_array, labels_array, folds_count):
    """
    :param data_array: a numpy array with the features dataset
    :param labels_array: a numpy array with the labels
    :param folds_count: number of cross-validation folds
    :return: res: a dataframe containing the results for the different c values. columns similar to `compare_svms`
    """

    res = pd.DataFrame()
    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    kernels_list = ['poly']*18
    cValues = []
    # creates a list of 18 dictionaries of type : 'C' : valueOfC
    for i in [1,0,-1,-1,-3,-4]:
        for j in [1,2,3]:
            cTemp = ((j/3)*(10 ** i))
            cValues.append({'C' : cTemp})
    print(cValues)
    res = compare_svms(data_array, labels_array, folds_count, kernels_list, cValues)


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return res


def get_test_set_performance(train_data, train_labels, test_data, test_labels):
    """
    :param train_data: a numpy array with the features dataset - train
    :param train_labels: a numpy array with the labels - train

    :param test_data: a numpy array with the features dataset - test
    :param test_labels: a numpy array with the labels - test
    :return: kernel_type: the chosen kernel type (either 'poly' or 'rbf')
             kernel_params: a dictionary with the chosen kernel's parameters - c value, gamma or degree
             clf: the SVM leaner that was built from the parameters
             tpr: tpr on the test dataset
             fpr: fpr on the test dataset
             accuracy: accuracy of the model on the test dataset
    """
    kernel_type = 'poly'
    kernel_params = {'class_weight' : 'balanced' ,'C' : SVM_DEFAULT_C, 'gamma' : SVM_DEFAULT_GAMMA, 'degree' : 2, 'kernel' : kernel_type}
    clf = SVC(gamma = 'auto')
    clf.set_params(**kernel_params)
  # TODO: set the right kernel
    tpr = 0.0
    fpr = 0.0
    accuracy = 0.0

    ###########################################################################
    # TODO: Implement the function                                            #
    ###########################################################################
    clf.fit(train_data, train_labels)
    prediction = clf.predict(test_data)
    tpr, fpr, accuracy = get_stats(prediction, test_labels)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return kernel_type, kernel_params, clf, tpr, fpr, accuracy
