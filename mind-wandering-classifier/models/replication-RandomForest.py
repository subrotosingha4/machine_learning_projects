#!/usr/bin/env python
"""This script performs a GridSearch over project meta-parameters and
a Random Forest classifier meta-parameters.  The goal is
to replicate (approximately) the configuration and results of the
reference paper for this project.

The GridSearch uses cross validation, and can be used with a
GroupKFold or LeaveOneGroupOut test/train set cross validation
splitter.  Most project meta-parameters are controlled by custom
Pipeline Transformers.  Most of these (in the `mindwandering.data`
module) are wrappers around existing scikit-learn or imblearn pipeline
transformers, so that we can select different types of feature
selection, class balancing procedures, etc.

This script when rerun builds multiple models, performing the
following steps:

1. Load the data
2. Construct transformation pipeline
3. Perform GridSearch using cross validation
4. Summarize results
5. Save resulting dataframe of the search results to a .pkl
   file that can be used for further analysis and
   visualization of the results.

"""
# globally useful imports of standard libraries needed in this notebook
import numpy as np
import os.path

# specific libraries or classes needed for the work in this notebook
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# need to use Pipeline from imblearn to add in a downsample or upsample
# to cross validation training
from imblearn.pipeline import Pipeline

# import project specific modules used in this notebook
import sys
sys.path.append('../src')
from mindwandering.data import FeatureScalerTransformer
from mindwandering.data import WinsorizationOutlierTransformer
from mindwandering.data import VIFThresholdTransformer
from mindwandering.data import FeatureSelectionTransformer
from mindwandering.data import ClassImbalanceTransformer
from mindwandering.data import GridSearchProgressHack
from mindwandering.train import train_models_using_vif_threshold
from mindwandering.train import train_models
from mindwandering.train import save_results


# vif thresholds to apply to data and labels before grid search pipeline
vif_thresholds_orig = [0, 5]
vif_thresholds = [0]

# model / estimator name for progress reporting
model_name = "Random Forest Classifier"

# Estimator transformer Pipeline to use in this search
pipeline = Pipeline(
    [
        ('progress', GridSearchProgressHack(verbose=False)),
        ('scaling',  FeatureScalerTransformer()),
        ('outlier',  WinsorizationOutlierTransformer()),
        ('features', SelectKBest()),
        #('features', SelectFromModel(estimator=LogisticRegression(solver='liblinear', penalty='l1', C=0.1), threshold=-np.inf)),
        ('balance',  ClassImbalanceTransformer()),
        ('model',    RandomForestClassifier(random_state=42)),
    ]
)

# The set of parameters used for grid search on this estimator.
# These determine which parameters are explored in the grid search
# for the logistic regression models trained
parameters = {
    'scaling__type_of_scaling': ['standard'],
    'outlier__outlier_threshold': [0.0, 3.0],
    'features__score_func': [f_classif],
    #'balance__balancer_type': ['random-undersampler', 'allknn', 'nearmiss', 'instance-hardness-threshold'],
    #'balance__balancer_type': ['allknn', 'condensed-nn', 'neighbourhood-cleaning', 'one-sided-selection'],
    #'balance__balancer_type': ['allknn', 'one-sided-selection'],
    'balance__balancer_type': ['allknn', 'smote-enn'],
    'model__n_estimators': [200, 500],
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth': [4, 5, 6, 7, 8],
    #'model__max_iter': [70000],
    'model__criterion': ['gini', 'entropy']
}

# For SelectKBest feature selection, the ratios of best features.
# We use ratios because vif threshold is done first, so we don't
# know exact feature count, so select ratio of remaining features
k_ratios = np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.5])


# The number of cross validation folds to perform for this model grid
# search
n_folds = 5


def main():
    """Main entry point for this model data generation and replication
    """
    # train the results
    results = train_models(vif_thresholds, model_name, pipeline, parameters, k_ratios, n_folds)

    # save the result to pickle file
    result_dir = '.'
    output_file = 'replication-RandomForest.pkl'
    output_file = os.path.join(result_dir, output_file)
    save_results(results, output_file)


if __name__ == "__main__":
    main()
