#!/usr/bin/env python
"""Create a standard plot of recall vs. precision for an estimator /
model.  This script assumes an already trained model is stored in a
pickle file in the results directory.  This script loads the
estimator, calculates recall and precision and plots them, and saves
it as a figure suitable for insertion into a paper.

"""
import os
import pickle
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
# add mindwandering module into python sys path
sys.path.append('../src')
from mindwandering.data import get_df_features
from mindwandering.data import get_mind_wandered_label
from mindwandering.train import load_model_results


# rather than pass around, we make the features, labels, participant_ids
# and our traing parameters and features as globals.  Modify these
# to perform grid search over different parameters and features
df_features = get_df_features()
df_label = get_mind_wandered_label()

# other global constants / locations.  parameterize these if we need
# flexibility to specify them on command line or move them around
model_dir = '../models'
figure_dir = '.'
description = """
This script will create a standard recall vs. precision plot of
a model trained on the mindwandering data suitable for
inclusion as a figure.  The name of the input model file is
given and the result saved in the indicated figure file name.
The file extension for the output figure name will be used
to determine the figure output format.
"""


def create_recall_precision_figure(best_estimator, model_name, output_file):
    """Create and save the recall vs. precision curve for these
    grid search results.

    Parameters
    ----------
    best_estimator - A scikit-learn model/estimator, can make predictions
      on df_features
    output_file - A full pathname to the image file to be created with this
      models confusion matrix.  We assume the file type is an image type
      that the savefig will use to determine output image format.
    """
    probabilities = best_estimator.predict_proba(df_features)[:, 1]
    precision, recall, thresholds = precision_recall_curve(df_label, probabilities)
    plt.figure(figsize=(12, 8))
    plt.plot(recall[1:], precision[1:], label='Model precision vs. recall')
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('%s Best Estimator Recall vs. Precision\n' % model_name, fontsize=16)
    plt.savefig(output_file, transparent=True, dpi=300)


def main():
    """Main entry point for this figure visualizaiton creation
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('model',
                        help='the name of the model file to process and create figure for')
    parser.add_argument('--output', default=None,
                        help='name of output figure, defaults to figure-model-recall-precision.png')
    args = parser.parse_args()

    # determine output file name if not given explicitly
    model_name = os.path.basename(args.model)
    model_name = os.path.splitext(model_name)[0]
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'figure-' + model_name + '-recall-precision' + '.png'
        output_file = os.path.join(figure_dir, output_file)

    # generate and save the figure for the asked for model
    _, best_estimator, best_score, best_params, _ = load_model_results(args.model)
    create_recall_precision_figure(best_estimator, model_name, output_file)


if __name__ == "__main__":
    main()
