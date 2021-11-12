#!/usr/bin/env python
"""Create a figure that compares the auc-roc curves of several
estimator models.  In particular, usually used to compare best 
models achieved in a search using different types of estimators.

"""
import os
import pickle
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
# add mindwandering module into python sys path
sys.path.append('../src')
from mindwandering.data import get_df_features
from mindwandering.data import get_df_features_train_test_split
from mindwandering.data import get_mind_wandered_label
from mindwandering.data import get_mind_wandered_label_train_test_split
from mindwandering.train import load_and_combine_model_results


# rather than pass around, we make the features, labels, participant_ids
# and our traing parameters and features as globals.  Modify these
# to perform grid search over different parameters and features
df_features_train, df_features_test = get_df_features_train_test_split()
df_label_train, df_label_test = get_mind_wandered_label_train_test_split()

# other global constants / locations.  parameterize these if we need
# flexibility to specify them on command line or move them around
figure_dir = '.'
description = """
This script will create a plot of combined auc-roc scores in order
to compare performance of multiple classifiers.  In general we use 
this to compare the best classifier found in a search for different 
types of estimators.
"""


def create_combined_aucroc_curves(model_names, best_estimators, best_scores, output_file):
    """Create and save the auc-roc curve for these
    grid search results.

    Parameters
    ----------
    model_names - The names of the models to be visualized.
    best_estimators - A list of scikit-learn model/estimator, can make predictions
      on df_features_test
    best_scores - A list of the aucroc score achieved by the best estimator we are 
      plotting.
    output_file - A full pathname to the image file to be created with this
      models confusion matrix.  We assume the file type is an image type
      that the savefig will use to determine output image format.
    """
    # start figure, create a baseline random auc-roc curve
    plt.figure(figsize=(12, 8))
    num_trials = df_label_test.shape[0]
    random_probabilities = np.zeros(num_trials)
    random_fpr, random_tpr, _ = roc_curve(df_label_test, random_probabilities)
    plt.plot(random_fpr, random_tpr, label='Baseline performance, aucroc = 0.5')

    # add in each best model one at a time
    for model_name, best_estimator, avg_score in zip(model_names, best_estimators, best_scores):
        predictions = best_estimator.predict(df_features_test)
        probabilities = best_estimator.predict_proba(df_features_test)[:, 1]
        model_fpr, model_tpr, _ = roc_curve(df_label_test, probabilities)
        final_score = roc_auc_score(df_label_test, predictions)
        plt.plot(model_fpr, model_tpr, label='%s, avg = %0.4f, final = %0.4f' % (model_name, avg_score, final_score))

    # add legend labels and title
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("Comparison of Best Estimator AUC-ROC Curves\n", fontsize=16)

    # save resulting figure
    plt.savefig(output_file, transparent=True, dpi=300)


def main():
    """Main entry point for this figure visualizaiton creation
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('models', nargs='+',
                        help='the name of the model files to process and create aucroc comparisons from')
    parser.add_argument('--output', default=None,
                        help='name of output figure, defaults to figure-models-combined-best-aucroc-curves.png')
    args = parser.parse_args()

    # determine output file name if not given explicitly
    model_name = 'standardml'
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'figure-' + model_name + '-combined-best-aucroc-curves' + '.png'
        output_file = os.path.join(figure_dir, output_file)


    # generate and save the figure for the asked for models
    model_names, _, best_estimators, best_scores, _, _ = load_and_combine_model_results(args.models)
    create_combined_aucroc_curves(model_names, best_estimators, best_scores, output_file)


if __name__ == "__main__":
    main()
