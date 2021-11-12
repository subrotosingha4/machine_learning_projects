#!/usr/bin/env python
"""Create a confusion matrix figure for an estimator / model.  This script assumes
an already trained model is stored in a pickle file in the results directory.
This script loads the estimator, calculates the confusion matrix, and saves it
as a figure suitable for insertion into a paper.

"""
import os
import sys
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
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
This script will create a standard confusion matrix suitable for
inclusion as a figure for a given trained estimator/model.
The name of the model is assumed to match a pickle file .pkl
saved in the standard location for model results for this project.
The resulting confusion matrix figure is generated and saved to
a file by this script.  The file extension of the output file 
name will determine the figure format file is saved as.
"""


def create_confusion_matrix_figure(best_estimator, model_name, output_file):
    """Create and save the confusion matrix of these grid search results.

    Parameters
    ----------
    best_estimator - A scikit-learn model/estimator, can make predictions
      on df_features
    output_file - A full pathname to the image file to be created with this
      models confusion matrix.  We assume the file type is an image type
      that the savefig will use to determine output image format.
    """
    predictions = best_estimator.predict(df_features)
    cm = confusion_matrix(df_label, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure(figsize=(8, 8))
    disp.plot()
    plt.title('      %s Best Estimator Confusion Matrix\n' % model_name, fontsize=14)
    plt.savefig(output_file, transparent=True, dpi=300)


def main():
    """Main entry point for this figure visualizaiton creation
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('model',
                        help='the name of the model file to process and create figure for')
    parser.add_argument('--output', default=None,
                        help='name of output figure, defaults to figure-model-confusion-matrix.png')
    args = parser.parse_args()

    # determine output file name if not given explicitly
    model_name = os.path.basename(args.model)
    model_name = os.path.splitext(model_name)[0]
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'figure-' + model_name + '-confusion-matrix' + '.png'
        output_file = os.path.join(figure_dir, output_file)

    # generate and save the figure for the asked for model
    _, best_estimator, best_score, best_params, _ = load_model_results(args.model)
    create_confusion_matrix_figure(best_estimator, model_name, output_file)


if __name__ == "__main__":
    main()
