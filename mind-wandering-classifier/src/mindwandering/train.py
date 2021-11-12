#!/usr/bin/env python
"""
Methods developed for model training.  In particular, for standard ML training
using scikit-learn we use same framework / pipeline to perform a grid search 
of estimator parameters using k-fold cross validation.  The only variation is
which parameters to search through and which estimator / model is being used.
"""
# globally useful imports of standard libraries needed in this library
import numpy as np
import pandas as pd
import os.path
import pickle

# specific libraries or classes needed for the work in this notebook
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, \
    chi2, SelectFpr, f_regression, mutual_info_classif
# from sklearn.pipeline import Pipeline

# need to use Pipeline from imblearn to add in a downsample or upsample
# to cross validation training
from imblearn.pipeline import Pipeline

# import project specific modules used in this notebook
import sys
sys.path.append('../src')
import mindwandering.data
import mindwandering.features
import mindwandering.evaluate
from mindwandering.data import FeatureScalerTransformer
from mindwandering.data import WinsorizationOutlierTransformer
from mindwandering.data import VIFThresholdTransformer
from mindwandering.data import FeatureSelectionTransformer
from mindwandering.data import ClassImbalanceTransformer
from mindwandering.data import GridSearchProgressHack


# rather than pass around, we make the features, labels, participant_ids
# and our traing parameters and features as globals.  Modify these
# to perform grid search over different parameters and features
#df_features = mindwandering.data.get_df_features()
#mind_wandered_label = mindwandering.data.get_mind_wandered_label()
#participant_ids = mindwandering.data.get_participant_ids()

# use a 80/20 train test split of data.  Do cross validation training
# on 80% train split.  Give final evaluation on 20% test split.
# We rely on the sklearn random state to always give same split.
train_size = 0.8   # 80% for training
random_state = 42  # always use same random split
df_features_train, _ = mindwandering.data.get_df_features_train_test_split(train_size, random_state)
mind_wandered_label_train, _ = mindwandering.data.get_mind_wandered_label_train_test_split(train_size, random_state)
participant_ids_train, _ = mindwandering.data.get_participant_ids_train_test_split(train_size, random_state)


def train_models_using_vif_threshold(vif_threshold, pipeline, parameters, k_ratios, n_folds=5):
    """This method performs a GridSearch of a LogisticRegression estimator
    over a set of parameters defined globally above at the top of this script.
    This method is using K-fold cross validation with folds grouped by
    participant_ids, e.g. trials for a participant will not be split between
    multiple training/testing folds.  This method scores estimators using
    aucroc scores and raw accuracy.  Best estimator is destermined by aucroc
    scores.

    Grid search using different numbers of features / data frame columns
    is problematic.  So for each vif threshold, which modifies the features
    to select only features about a vif threshold, we do a separate
    grid search on this vif threshold setting.

    Paramters
    ---------
    vif_threshold - The vif cutoff threshold we are to make a modified
        dataframe of features for training with.
    pipeline - The estimator pipeline.  A sequence of transformers that can take
        the mind wandering dataframe and transform it to train a
        scikit-learn estimator.
    parameters - A dictionary of parameters used in the grid search.  This
        specifies which model and transformer parameter variations are
        explored in the search.
    k_ratios - We commonly use SelectKBest feature selection, but with vif
        thresholding we don't have absolute k to select.  So pass in
        desired ratios of total number of features wanted and we calculate 
        the actual k after vif feature thresholding.
    n_folds - The number of folds to perform for k-fold validation.  Defaults to
        5 folds.


    Returns
    -------
    Returns a tuple with all of the following:

    df_result - A dataframe containing all of the training results.
    best_estimator, best_score, best_params best_index - - The best estimator
       and information about its parameters and evaluation.
    """
    # display progress
    print('')
    print('')
    print('Starting vif meta-parameter condition: vif_threshold: ', vif_threshold)

    # pipeline to apply vif threshold selection for this grid search attempt
    vif_pipeline = Pipeline(
        [
            ('vif', VIFThresholdTransformer(score_threshold=vif_threshold)),
        ]
    )
    df_vif_features = vif_pipeline.transform(df_features_train)
    num_trials, num_features = df_vif_features.shape
    print('   Number of trials: ', num_trials, ' Number of Features: ', num_features)

    # we want to actually select a certain percentage of features of
    # whatever remains after vif thresholding.  So we will calculate
    # these in the loop based on the number of features in data, and
    # add to the parameters dictionary before the grid search set
    # feature selection k correctly
    features__k = k_ratios * num_features
    features__k = features__k.astype(int)
    parameters['features__k'] = features__k.tolist()

    # we assume the pipeline is using the GridSearchProgressHack,
    # reset so we get progress for this training
    GridSearchProgressHack.num_fits = 1

    # Cross Validation Splitter
    cv_group_splitter = GroupKFold(n_splits=n_folds)
    #cv_group_splitter = LeaveOneGroupOut()

    # perform the grid search for this vif selection
    # set up the search
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameters,
        scoring=['roc_auc', 'accuracy'],
        cv=cv_group_splitter,
        refit='roc_auc',
        verbose=1
    )
    search.fit(df_vif_features, mind_wandered_label_train, groups=participant_ids_train)

    # create a dataframe of all results for this vif_threshold and return
    df_result = pd.DataFrame(data=search.cv_results_)
    df_result['param__vif_score_threshold'] = vif_threshold
    return (df_result, search.best_estimator_, search.best_score_,
            search.best_params_, search.best_index_)


def train_models(vif_thresholds, model_name, pipeline, parameters, k_ratios, n_folds=5):
    """Train LogisticRegression models.  This function iterates over
    vif thresholds.  The real work for each vif threshold setting is done
    in the train_models_using_vif_threshold() method.  This method
    simply gathers together individual runs into a resulting data
    frame to be returned and saved.

    Parameters
    ----------
    vif_thresholds - The vif threshold cutoffs to use in search
    model_name - The name of the estimator pipeline we are searching/training.
        For progress reporting.
    pipeline - The estimator pipeline.  A sequence of transformers that can take
        the mind wandering dataframe and transform it to train a
        scikit-learn estimator.
    parameters - A dictionary of parameters used in the grid search.  This
        specifies which model and transformer parameter variations are
        explored in the search.
    k_ratios - We commonly use SelectKBest feature selection, but with vif
        thresholding we don't have absolute k to select.  So pass in
        desired ratios of total number of features wanted and we calculate 
        the actual k after vif feature thresholding.
    n_folds - The number of folds to perform for k-fold validation.  Defaults to
        5 folds.

    Returns
    -------
    Returns a tuple with all of the following:

    df_result - A dataframe containing all of the training results.
    best_estimator, best_score, best_params best_index - The best
       estimator and information about its parameters and evaluation.
    """
    # Empty dataframe to gather grid search results into 1 place
    best_estimator = None
    best_score = 0.0
    best_params = {}
    best_index = 0
    df_result = None

    # search over vif threshold meta parameters, create separate df from
    # vif_threshold for otherwise identical grid searches
    print('')
    print('%s Estimator grid search' % model_name)
    print('=' * 50)
    for vif_threshold in vif_thresholds:

        (vif_df_result, vif_best_estimator, vif_best_score, vif_best_params, vif_best_index) = train_models_using_vif_threshold(vif_threshold, pipeline, parameters, k_ratios, n_folds)

        if best_estimator is None:
            df_result = vif_df_result
            best_estimator = vif_best_estimator
            best_score = vif_best_score
            best_params = vif_best_params
            best_index = vif_best_index
        else:
            df_result = df_result.append(vif_df_result, ignore_index=True)
            if vif_best_score > best_score:
                best_estimator = vif_best_estimator
                best_score = vif_best_score
                best_params = vif_best_params
                best_index = vif_best_index  # todo this is only the index of this search

    print('')
    print('')
    return (df_result, best_estimator, best_score, best_params, best_index)


def save_results(results, output_file):
    """Save the data frame of results for data visualization and analysis

    Parameters
    ----------
    results - A tuple of the results to save.  We just save the tuple,
       no need to know what is in the results, but we expected a
       dataframe of all grid search results, plus specific information
       about the best estimator found.
    output_file - The name of the file to save the model results to.

    """
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)


def load_model_results(input_file):
    """Load the results from a pickle file.  We expect full path name to
    pickle file to be opened.  We also expect that the pickle file was
    saved with the save_results() from the train_models() method, and
    has the expected set of data that is extracted and returned from
    the pickle.

    Parameters
    ----------
    input_file - The full path to the pickle file that contains the
      saved and trained estimator / model we are to load and return.

    Returns
    -------
    Returns a tuple containing all of the saved results from a grid
    search run:

    df_result - A dataframe of all results of runs of multiple models
        for a grid search
    best_estimator - the LogisticRegression estimator with the best
        performance we found
    best_score - the aucroc score of the best performer.  This should
        be the average evaluation over k-folds. TODO: check this
    best_params - the grid search parameters used by the best
        performing estimator
    best_index - The index into the df_results of the best estimator
        that was trained

    """

    # Get the grid search results
    with open(input_file, 'rb') as f:
        df_result, best_estimator, best_score, best_params, best_index = pickle.load(f)

    return df_result, best_estimator, best_score, best_params, best_index


def load_and_combine_model_results(models):
    """Load the results for multiple models from multiple pickle files.
    We expect a list of full path names to pickle files to open and load.
    We combine all df_result dataframes into a single dataframe to return.

    Parameters
    ----------
    models - a list of input model pickle files that we should load
       to generate visualization from.

    Returns
    -------
    Returns a tuple of the following 

    model_names - A list of the extracted model / estimator names
        that were trained in this combined search data.
    df_result - A dataframe of the combined results of all models asked
        for.
    best_estimators - A list of the best estimator found for each 
        estimator type.
    best_scores - A list of the best auc-roc score achieved by each 
        type of estimator loaded.
    best_params - A list of dictionaries of the parameters used to train 
        the best estimator of each type.
    best_indexes - A list of integers indexes into the df_result 
        dataframe of the best estimator. TODO: this will currently be
        index of original dataframe, not of combined dataframe, so need 
        to fix this.
    """
    df_result = None
    model_names = []
    best_estimators = []
    best_scores = []
    best_params = []
    best_indexes = []
    
    # process each model we were asked to load
    for model in models:
        # extract model name from full file name
        model_name = os.path.basename(model)
        model_name = os.path.splitext(model_name)[0]
        model_name = model_name.split('-')[1]
        model_names.append(model_name)
        
        # we expect a full path name to a model pickle file, so open
        # that file to extract
        with open(model, 'rb') as f:
            df_result_model, best_estimator, best_score, best_param, best_index = pickle.load(f)

            # append these results to our return data frame
            if df_result is None:
                df_result = df_result_model
            else:
                df_result = df_result.append(df_result_model, ignore_index=True)
                
            best_estimators.append(best_estimator)
            best_scores.append(best_score)
            best_params.append(best_param)
            best_indexes.append(best_index)

    return model_names, df_result, best_estimators, best_scores, best_params, best_indexes
