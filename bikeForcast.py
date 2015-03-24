# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:29:59 2015

@author: damienrj
"""
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import math
    
    from sklearn import ensemble
    from sklearn.cross_validation import train_test_split
    from sklearn.metrics import mean_absolute_error
    from sklearn.grid_search import GridSearchCV
    from datetime import datetime
    
    #Load Data with pandas, and parse the first column into datetime
    train = pd.read_csv('train.csv', parse_dates=[0])
    test = pd.read_csv('test.csv', parse_dates=[0])
    
    #Feature engineering
       
    temp = pd.DatetimeIndex(train['datetime'])
    train['year'] = temp.year
    train['month'] = temp.month
    train['hour'] = temp.hour
    train['weekday'] = temp.weekday
    
    temp = pd.DatetimeIndex(test['datetime'])
    test['year'] = temp.year
    test['month'] = temp.month
    test['hour'] = temp.hour
    test['weekday'] = temp.weekday
                
    #the evaluation metric is the RMSE in the log domain, 
    #so we should transform the target columns into log domain as well.
    for col in ['casual', 'registered', 'count']:
        train['log-' + col] = train[col].apply(lambda x: np.log1p(x))
        
    #Define features vector
    features = ['season', 'holiday', 'workingday', 'weather',
            'temp', 'atemp', 'humidity', 'windspeed', 'year',
             'month', 'weekday', 'hour']
    
   

    #Some of the important parameters for the GBM are:
    #    number of trees (n_estimators)
    #    depth of each individual tree (max_depth)
    #    loss function (loss)
    #    learning rate (learning_rate)   
    clf = ensemble.GradientBoostingRegressor(n_estimators=200, max_depth=3)         
    clf.fit(train[features], train['log-count'])
    result = clf.predict(test[features])
    result = np.expm1(result)
    
    df=pd.DataFrame({'datetime':test['datetime'], 'count':result})
    df.to_csv('results1.csv', index = False, columns=['datetime','count'])
    
    #So far, not that great of a result and we are in the bottom 10%.  
    #The first step might be to try to tune the parameters. 
    
    #Hyperparameter tuning
    
    #Split data into training and validation sets
    temp = pd.DatetimeIndex(train['datetime'])
    training = train[temp.day <= 16]
    validation = train[temp.day > 16]
   
    param_grid = {'learning_rate': [0.1, 0.05, 0.01],
                  'max_depth': [10, 15, 20],
                  'min_samples_leaf': [3, 5, 10, 20],
                  }
    
    est = ensemble.GradientBoostingRegressor(n_estimators=500)
    # this may take awhile 
    gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(training[features], training['log-count'])
    
    # best hyperparameter setting
    gs_cv.best_params_
    
    #{'learning_rate': 0.05, 'max_depth': 10, 'min_samples_leaf': 20}
  
  
    error_count = mean_absolute_error(validation['log-count'], gs_cv.predict(validation[features]))
       
    result = gs_cv.predict(test[features])
    result = np.expm1(result)
    df=pd.DataFrame({'datetime':test['datetime'], 'count':result})
    df.to_csv('results2.csv', index = False, columns=['datetime','count'])
    # Our results are now in the top 20%!
    
    #Lets take a look at the number of estimators now
    error_train=[]
    error_validation=[]
    for k in range(10, 501, 10):
        clf = ensemble.GradientBoostingRegressor(
            n_estimators=k, learning_rate = .05, max_depth = 10,
            min_samples_leaf = 20)
        clf.fit(training[features], training['log-count'])
        result = clf.predict(training[features])
        error_train.append(
            mean_absolute_error(result, training['log-count']))
        result = clf.predict(validation[features])
        error_validation.append(
            mean_absolute_error(result, validation['log-count']))        
    
    #Plot the data
    x=range(10,501, 10)
    plt.style.use('ggplot')
    plt.plot(x, error_train, 'k')
    plt.plot(x, error_validation, 'b')
    plt.xlabel('Number of Estimators', fontsize=18)
    plt.ylabel('Error', fontsize=18)
    plt.legend(['Train', 'Validation'], fontsize=18)
    plt.title('Error vs. Number of Estimators', fontsize=20)
    
    # Given that the count is the sum of registered and casual we might
    # get better performance modeling them separately. 
    
    def merge_predict(model1, model2, test_data):
    #    Combine the predictions of two separately  trained models. 
    #    The input models are in the log domain and returns the predictions
    #    in original domain.
        p1 = np.expm1(model1.predict(test_data))
        p2 = np.expm1(model2.predict(test_data))
        p_total = (p1+p2)
        return(p_total)
        
    est_casual = ensemble.GradientBoostingRegressor(n_estimators=80, learning_rate = .05)
    est_registered = ensemble.GradientBoostingRegressor(n_estimators=80, learning_rate = .05)
    param_grid2 = {'max_depth': [10, 15, 20],
                  'min_samples_leaf': [3, 5, 10, 20],
                  }
                  
    gs_casual = GridSearchCV(est_casual, param_grid2, n_jobs=4).fit(training[features], training['log-casual'])  
    gs_registered = GridSearchCV(est_registered, param_grid2, n_jobs=4).fit(training[features], training['log-registered'])      
    
    error_registered = mean_absolute_error(
        validation['log-registered'], gs_registered.predict(validation[features]))
        
    error_casual = mean_absolute_error(
        validation['log-casual'], gs_casual.predict(validation[features]))
        
    print([error_count, error_casual, error_registered])
    #[0.20943850897991825, 0.40021857004963229, 0.1985197355367242]
    #If we take a look, we get the same hyper parameters which shouldn't be a big 
    #suprise            
    
    result3 = merge_predict(gs_casual, gs_registered, test[features])
    df=pd.DataFrame({'datetime':test['datetime'], 'count':result3})
    df.to_csv('results3.csv', index = False, columns=['datetime','count'])
   
    #Looking at the different users seperatly moved me up a few of percent,
    #Lastly, lets train our models on the whole dateset without leaving any out for 
    #looking at our error.
    
    est_casual = ensemble.GradientBoostingRegressor(
        n_estimators=80, learning_rate = .05, max_depth = 10,min_samples_leaf = 20)
    est_registered = ensemble.GradientBoostingRegressor(
        n_estimators=80, learning_rate = .05, max_depth = 10,min_samples_leaf = 20)
        
    est_casual.fit(train[features].values, train['log-casual'].values)
    est_registered.fit(train[features].values, train['log-registered'].values)
    result4 = merge_predict(est_casual, est_registered, test[features])
    
    df=pd.DataFrame({'datetime':test['datetime'], 'count':result4})
    df.to_csv('results4.csv', index = False, columns=['datetime','count'])
    
    #With that I have a result in the top 10%