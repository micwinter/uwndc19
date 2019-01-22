"""
Base model class from UWNDC19 challenge
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.metrics import r2_score

class BaseModel:


    def __init__(self):
        self = self

    def fit(self, data, target):
        self.train_data = data
        self.train_target = target
        self.rgb = np.mean(data, (1,2))#average RGB across rows and columns of each image

    def predict(self, test_data, test_target):

        def rmse(x, y):
            return np.sqrt(np.mean((x-y) ** 2))

        test_preds = [] # create list to contain predictions
        trainr2 = [] # list for fit to training data
        testr2 = [] # list for fit to testing data
        testrmse = [] # list for rmse results
        testpredr2 = [] # list for predictive r2 score
        A = np.hstack([self.rgb, np.ones((self.rgb.shape[0], 1))]) #linear predictors rgb, and constant(intercept)
        for i in range(self.train_target.shape[1]):#for each recording
            y = self.train_target[:,i]
            remove_idxs = np.isnan(y)
            y = y[~remove_idxs]
            corsp_train_data = self.train_data[~remove_idxs] # stim corresponding to neural responses that were not nan

            A_train = A[:len(y)]#get features of images we have responses for
            test_rgb = np.mean(test_data, (1,2))
            A_test = np.hstack([test_rgb, np.ones((test_rgb.shape[0], 1))])#get features for test images you are making prediction on
            #train model i.e. regress rgb values onto responses
            coefs = np.linalg.lstsq(A_train, y, rcond=None)[0];
            fit_train_y = np.dot(A_train, coefs)#get the prediction for known responses
            trainr2.append(np.corrcoef(y, fit_train_y)[0,1] ** 2)

            fit_test_y = np.dot(A_test, coefs)#use coefficients found for training data to make prediction on test data

            # Remove nans in test set
            test = test_target[:,i]
            remove_idxs = np.isnan(test)
            test = test[~remove_idxs]
            fit_test_y = fit_test_y[~remove_idxs]

            testrmse.append(rmse(test, fit_test_y))
            testr2.append(np.corrcoef(test, fit_test_y)[0,1] ** 2)
            testpredr2.append(r2_score(test, fit_test_y, multioutput='variance_weighted'))
            test_preds.append(fit_test_y)# append these predictions

        tag = random.randint(1000,9999)

        print('Root mean squared error', str(np.mean(testrmse)))
        print('R2 score', str(np.mean(testpredr2)))


        trainr2= np.array(trainr2)
        plt.plot(trainr2);plt.xlabel('Recording Index');plt.ylabel(r'Train $R^2$');
        plt.xticks(range(len(trainr2)));plt.title('Performance of RGB model');
        plt.savefig(str(tag)+'_trainr2_predscore_varweight')
        plt.close()

        test_preds = np.array(test_preds)#convert predictions to array
        testr2 = np.array(testr2)
        plt.plot(testr2, label='corrcoef^2');plt.xlabel('Recording Index');plt.ylabel(r'Test $R^2$');
        plt.plot(testpredr2, label='pred r2'); # Predictive score
        plt.legend()
        plt.xticks(range(len(testr2)));plt.title('Performance of RGB model');
        plt.savefig(str(tag)+'_testr2_predscore_varweight')
        plt.close()

        #TODO: Print root mean squared error

        # sdf = pd.DataFrame(test_preds.T)
        # sdf.columns = df.columns#replace the columns with the correct cell ids from training data
        # sdf.index.name = 'Id'
        # sdf.to_csv('sub.csv')#save to csv
        # sdf.head()#show top couple rows of submission
