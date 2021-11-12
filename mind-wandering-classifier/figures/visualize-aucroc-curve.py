#!/usr/bin/env python
"""Create a standard auc-roc area under the receiver operating curve
figure for an estimator / model.  This script assumes an already
trained model is stored in a pickle file in the results directory.
This script loads the estimator, calculates recall and precision and
plots them, and saves it as a figure suitable for insertion into a
paper.

"""
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
# add mindwandering module into python sys path
sys.path.append('../src')
from mindwandering.data import get_df_features
from mindwandering.data import get_df_features_train_test_split
from mindwandering.data import get_mind_wandered_label
from mindwandering.data import get_mind_wandered_label_train_test_split
from mindwandering.train import load_model_results

# rather than pass around, we make the features, labels, participant_ids
# and our traing parameters and features as globals.  Modify these
# to perform grid search over different parameters and features
df_features_train, df_features_test = get_df_features_train_test_split()
df_label_train, df_label_test = get_mind_wandered_label_train_test_split()

# other global constants / locations.  parameterize these if we need
# flexibility to specify them on command line or move them around
figure_dir = '.'
description = """
This script will create a standard AUC-ROC (area under the curve of
the receive operator characteristics) plot for a model trained on the
mindwandering data suitable for inclusion as a figure.  The name of
the input model file is given and the result saved in the indicated
figure file name.  The file extension for the output figure name will
be used to determine the figure output format.
"""


def create_aucroc_figure(best_estimator, best_score, model_name, output_file):
    """Create and save the auc-roc curve for these
    grid search results.

    Parameters
    ----------
    best_estimator - A scikit-learn model/estimator, can make predictions
      on df_features
    best_score - The aucroc score achieved by the best estimator we are 
      plotting.
    model_name - The name of the model estimator we are visualizing
    output_file - A full pathname to the image file to be created with this
      models confusion matrix.  We assume the file type is an image type
      that the savefig will use to determine output image format.
    """
    num_trials = df_label_test.shape[0]
    random_probabilities = np.zeros(num_trials)
    random_fpr, random_tpr, _ = roc_curve(df_label_test, random_probabilities)
    probabilities = best_estimator.predict_proba(df_features_test)[:, 1]
    model_fpr, model_tpr, _ = roc_curve(df_label_test, probabilities)
    plt.figure(figsize=(12, 8))
    plt.plot(random_fpr, random_tpr, label='Random performance, aucroc = 0.5')
    plt.plot(model_fpr, model_tpr, label='Model performance, aucroc = %0.4f' % best_score)
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('%s Best Estimator AUC-ROC Curve\n' % model_name, fontsize=16)
    plt.savefig(output_file, transparent=True, dpi=300)


def main():
    """Main entry point for this figure visualizaiton creation
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('model',
                        help='the name of the model file to process and create figure for')
    parser.add_argument('--output', default=None,
                        help='name of output figure, defaults to figure-model-aucroc-curve.png')
    args = parser.parse_args()

    # determine output file name if not given explicitly
    model_name = os.path.basename(args.model)
    model_name = os.path.splitext(model_name)[0]
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'figure-' + model_name + '-aucroc-curve' + '.png'
        output_file = os.path.join(figure_dir, output_file)

    # generate and save the figure for the asked for model
    _, best_estimator, best_score, best_params, _ = load_model_results(args.model)
    create_aucroc_figure(best_estimator, best_score, model_name, output_file)


if __name__ == "__main__":
    main()
