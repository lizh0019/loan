'''
This prediction model is developed using the most state-of-the-art machine learning toolbox availabe at : https://github.com/AxeldeRomblay/MLBox

The target "MaxOverDueDays" is manually changed to classification labels (0,1) by "=int(Q>90)" in excel.
The training dataset "sample_200_0k_20170120_default_balance.csv" is created by keeping the 50:50 ratio of positive samples and negative samples.
The testing dataset "sample_200_0k_20170120_no_target.csv" is created by removing the target.

The feature importance is reported in "LightGBM_feature_importance.png", most important is monthlyfixedincome
The feature drift is reported in "drifts.txt", most drift is "previous" and "birthplace"
The prediction probabilities and labels are submitted "Default_predictions.csv"
The AUC value of the prediction is roughly 0.70

[1]    What are some notable features of this data set?
67 common features, including 8 categorical features and 59 numerical features
In addition, there are 2 positive samples which are duplicates of other samples, and have been removed, resulting in 31115 positive samples and 31117 negative samples.

[2]    What are some potential drawbacks of your solution?
Basically the approach is ensemble of hundreds of random decision trees. There are many choices of configurations. We try 15 pre-set configurations and find the best one, and it will be used to predict the whole dataset. Computation time is a lot, and adaboosting methods can hardly be parallelized, and no available GPU platform to accelerate. 

[3]    How do you think about whether your algorithm might be overfitting?
It is ensemble method, variance is reduced, and errors or noises will have little effect to the model parameters. Overfitting is unlikely to happen.

[4]    What's the runtime of your algorithm in training and testing? Please provide both:
(a) A theoretical analysis (a rough, hand-wavy analysis would be fine)
The ensemble method used here is Gradient-based Decision Tree Boosting. It is essentially a variant of the traditional adaboosting approach. The main difference is that the loss function is changed from the exponential loss exp(-y*f(x)) to absolute error abs(y-f(x)), and weight udpate is done by loss gradient. A typical adaboosting algorithm has complexity O(N*D*D), where N is number of samples and D is the feature dimensionality. 
(b) The amount of time that your computer actually spent on the dataset
Training process including finding the best configuration needs about 6 hours, while prediction needs in about 20min.

[5]    If you had time to work on this problem for another week, what else would you do beyond what you've done here?
First, expand the positive samples so that the dataset is roughly 17K+17K balanced one. The expand can be done by augmenting the positive samples by introducing Gaussian noises to numerical features.
Second, try a larger set of configurations, say 1000 different combinations, e.g. include more types of ensemble methods such as bagging and adaboosting, and find the best one. Also try removing the least important features and most drifting features.
Third, try other approaches, such as SVM, and try several kernels, such as intersection kernel, Gaussian kernel, heavy-tailed RBF kernel, which are tested to be effective before.
Fourth, ensemble the predictions of GBDT, Bagging, adaboosting, and SVM with different kernels. The ensemble method can be simply arithmetic mean or geometric mean or median.

'''
__author__='lizhen'

## code begins
from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *
from sklearn.metrics import roc_auc_score
sep = ','
#paths = ["train.csv","test.csv"]
paths = ["sample_200_0k_20170120_default.csv","sample_200_0k_20170120_no_target.csv"]

#target_name="Survived"
#target_name = 'MaxOverDueDays'
target_name = 'Default'
#target_name="SalePrice"
#scoring = 'accuracy'
scoring = 'roc_auc'
#scoring = 'mae'
n_folds=3
## wait about 30s to proceed until "(Pdb) 2017-07-10 22:59:06.002 [IPClusterStart] Engines appear to have started successfully"

'''
must wait here!

'''
rd = Reader(sep)
df = rd.train_test_split(paths, target_name)

#df['target'] = map(lambda r: int(r > 90), df['target'] )

dft = Drift_thresholder()
df = dft.fit_transform(df)

# for i in range(len(df['target'])):
#     try:
#         df['target'][i] = int(df['target'][i]>90)
#     except:
#         pass    
#roc_auc_score(y_true, y_pred)
#mape = make_scorer(lambda y_true, y_pred: roc_auc_score(y_true, y_pred), greater_is_better=True, needs_proba=False)
#mape = make_scorer(lambda y_true, y_pred: np.mean(np.abs(int(y_true>90)-int(y_pred>90))), greater_is_better=False, needs_proba=False)
#opt = Optimiser(scoring = mape, n_folds=n_folds)
opt = Optimiser(scoring, n_folds)
opt.evaluate(None, df)


space = {
    'ne__numerical_strategy':{"search":"choice", "space":['mean']},
    'ce__strategy':{"search":"choice", "space":["entity_embedding"]},#"label_encoding",
    'fs__strategy' : {"search":"choice", "space":["l1"]},
    'fs__threshold':{"search":"uniform", "space":[0.15,0.20]},    
    'est__max_depth':{"search":"choice", "space":[4,5]},
    'est__subsample' : {"search":"uniform", "space":[0.7,0.85]},
    'est__learning_rate':{"search":"uniform", "space":[0.03,0.05]},
    'est__n_estimators':{"search":"choice", "space":[300,400]}
}

best = opt.optimise(space, df, 15)
np.save("df.npy",df)
df = np.load("df.npy").item()
prd = Predictor()
prd.fit_predict(best,df)


pass
