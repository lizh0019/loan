'''
This prediction model is developed using the most state-of-the-art machine learning toolbox availabe at : https://github.com/AxeldeRomblay/MLBox


The training dataset "sample_200_0k_20170120_default_balance.csv" is created by keeping the 50:50 ratio of positive samples and negative samples.
The testing dataset "raw_data_30_new_nolabel.csv" is created by removing the target.

The feature importance is reported in "LightGBM_feature_importance.png", most important is feture7
The feature drift is reported in "drifts.txt", most drift is feature is feature47
The prediction probabilities and labels are submitted "save/Bad_label_predictions.csv"
The AUC value of the prediction is roughly 0.63
accuracy is 95.7%

[1] noticeable features of the dataset
number of common features : 100

gathering and crunching for train and test datasets
reindexing for train and test datasets
dropping training duplicates
dropping constant variables on training set

number of categorical features : 43
number of numerical features : 57
number of training samples : 23896
number of test samples : 10240

task : classification
0.0    22892
1.0     1004
Name: Bad_label, dtype: int64
encoding target

computing drifts...
CPU time: 5.21044802666 seconds


[2]    Some potential drawbacks of your solution
Basically the approach is ensemble of hundreds of random decision trees. There are many choices of configurations. We try 15 pre-set configurations and find the best one, and it will be used to predict the whole dataset. Computation time is a lot, and adaboosting methods can hardly be parallelized, and no available GPU platform to accelerate. 

[3]    overfitting?
It is ensemble method, variance is reduced, and errors or noises will have little effect to the model parameters. Overfitting is unlikely to happen.

[4]    runtime of your algorithm in training and testing:
(a) A theoretical analysis (a rough, hand-wavy analysis would be fine)
The ensemble method used here is Gradient-based Decision Tree Boosting. It is essentially a variant of the traditional adaboosting approach. The main difference is that the loss function is changed from the exponential loss exp(-y*f(x)) to absolute error abs(y-f(x)), and weight udpate is done by loss gradient. A typical adaboosting algorithm has complexity O(N*D*D), where N is number of samples and D is the feature dimensionality. 
(b) The amount of time that your computer actually spent on the dataset
Training process including finding the best configuration needs about 2 hours, while prediction needs in about 5 minutes.

[5]    Future work
First, expand the positive samples so that the dataset is roughly 22892+22892 balanced one. The experiments can be done by augmenting the positive samples by introducing Gaussian noises to numerical features.
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
paths = ["raw_data_70_new.csv","raw_data_30_new_nolabel.csv"]

target_name = 'Bad_label'
#scoring = 'accuracy'
scoring = 'roc_auc'
#scoring = 'mae'
n_folds=5
## wait about 30s to proceed until "2017-09-02 20:47:28.999 [IPClusterStart] Engines appear to have started successfully"

'''
must wait here!








'''
rd = Reader(sep)
df = rd.train_test_split(paths, target_name)
dft = Drift_thresholder()
df = dft.fit_transform(df)

#roc_auc_score(y_true, y_pred)
#mape = make_scorer(lambda y_true, y_pred: roc_auc_score(y_true, y_pred), greater_is_better=True, needs_proba=False)
#opt = Optimiser(scoring = mape, n_folds=n_folds)

opt = Optimiser(scoring, n_folds)
opt.evaluate(None, df)


space = {
    'ne__numerical_strategy':{"search":"choice", "space":['mean']},
    'ce__strategy':{"search":"choice", "space":["entity_embedding"]},#
    'fs__strategy' : {"search":"choice", "space":["l1","l2"]},
    'fs__threshold':{"search":"uniform", "space":[0.15,0.17]},    
    'est__max_depth':{"search":"choice", "space":[4,5]},
    'est__subsample' : {"search":"uniform", "space":[0.7,0.75]},
    'est__learning_rate':{"search":"uniform", "space":[0.04,0.05]},
    'est__n_estimators':{"search":"choice", "space":[400,500]}
}

best = opt.optimise(space, df, 15)
np.save("df.npy",df)
df = np.load("df.npy").item()
prd = Predictor()
prd.fit_predict(best,df)


pass
