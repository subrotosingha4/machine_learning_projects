#!/usr/bin/env python
"""Create histogram of aucroc model performance results for grid
search of models.

"""
import os
import sys
import argparse
import matplotlib.pyplot as plt
# add mindwandering module into python sys path
sys.path.append('../src')
from mindwandering.train import load_and_combine_model_results


# other global constants / locations.  parameterize these if we need
# flexibility to specify them on command line or move them around
model_dir = '../models'
figure_dir = '.'
description = """
This script will visualize a histogram of aucroc performance
from a set of grid searched estimators.  We expect one or more
pickle .pkl files as input.  We load the result data from each
and combine to create a single histogram of all aucroc performance
for grid searched trained models.
"""


def create_aucroc_histogram(df_result, model_name, output_file):
    """Create the actual figure / visualizaiton of the data.  In this
    case we are creating a histogram of multiple grid searched runs
    over parameter space to find good estimators of the mindwandering
    data

    Parameters
    ----------
    df_result - A dataframe containing the results of grid search of
       estimator parameters on the mindwandering data.
    model_name - The name of the model(s) used for the histogram
    output_file - The file name to save the generated histogram figure
       into
    """
    plt.figure(figsize=(12, 8))
    plt.hist(df_result.mean_test_roc_auc,
             bins=20,
             color='lightblue',
             rwidth=0.9)
    num_models, _ = df_result.shape
    plt.xlabel('Area under the receiver operating characteristic (AUC-ROC) score')
    plt.ylabel('Number of models')
    plt.title('Standard ML Estimator AUCROC GridSearch Training Results\n n = %d models searched' % num_models, fontsize=16)
    plt.savefig(output_file, transparent=True, dpi=300)


def main():
    """Main entry point for this figure visualizaiton script
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('models', nargs='+',
                        help='the name of the model files to process and create histogram')
    parser.add_argument('--output', default=None,
                        help='name of output figure, defaults to figure-models-aucroc-histogram.png')
    args = parser.parse_args()

    # determine output file name if not given explicitly 
    model_name = 'standardml'
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'figure-' + model_name + '-aucroc-histogram' + '.png'
        output_file = os.path.join(figure_dir, output_file)


    # generate and save the figure for the asked for models
    model_names, df_result, _, _, _, _ = load_and_combine_model_results(args.models)
    create_aucroc_histogram(df_result, model_names, output_file)


if __name__ == "__main__":
    main()
