#!/usr/bin/env python
"""Generate a LaTeX table comparing the performance of best 
estimators found in the parameter grid searches.

"""
import os
import sys
import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
# add mindwandering module into python sys path
sys.path.append('../src')
from mindwandering.data import get_df_features
from mindwandering.data import get_mind_wandered_label
from mindwandering.train import load_and_combine_model_results


# rather than pass around, we make the features, labels, participant_ids
# and our traing parameters and features as globals.  Modify these
# to perform grid search over different parameters and features
df_features = get_df_features()
df_label = get_mind_wandered_label()

# other global constants / locations.  parameterize these if we need
# flexibility to specify them on command line or move them around
table_dir = '.'
description = """
This script will create a table comparing the auc-roc and accuracy 
performance of the best models achieved in the parameter grid
searches.  The table is output as a LaTeX markup to a file for 
inclusion in reporting documents.
"""


def generate_performance_comparison_df(model_names, best_estimators, best_scores, best_params):
    """Create the table comparing performance of the models.

    Parameters
    ----------
    model_names - The names of the models to be visualized.
    best_estimators - A list of scikit-learn model/estimator, can make predictions
      on df_features
    best_scores - A list of the aucroc score achieved by the best estimator we are 
      plotting.
    output_file - A full pathname to the image file to be created with this
      models confusion matrix.  We assume the file type is an image type
      that the savefig will use to determine output image format.

    Returns
    -------
    performance_df - Returns a pandas dataframe of the gathered performance comparison.
      Pandas dataframes can be converted to LaTeX tables, and other table types, with ease.
    """
    # process each model one by one
    columns = ['Model name', 'k-fold aucroc', 'final aucroc', 'accuracy', 'recall', 'precision']
    performance_df = pd.DataFrame(columns=columns)
    for model_name, best_estimator, avg_score, best_param in zip(model_names, best_estimators, best_scores, best_params):
        predictions = best_estimator.predict(df_features)
        accuracy = accuracy_score(df_label, predictions)
        recall = recall_score(df_label, predictions, zero_division=0)
        precision = precision_score(df_label, predictions, zero_division=0)
        aucroc = roc_auc_score(df_label, predictions)

        # generate the table row for this model
        model_df = {
            'Model name': model_name,
            'k-fold aucroc': avg_score,
            'final aucroc': aucroc,
            'accuracy': accuracy,
            'recall': recall,
            'precision': precision,
        }
        performance_df = performance_df.append(model_df, ignore_index=True)

    return performance_df


def save_table(performance_df, output_file):
    """Create and save a generated LaTeX table of this performance
    information.

    Parameters
    ----------
    performance_df - A dataframe of the estimator performance information
      to save
    output_file - The name of the file to save the table into.
    """
    caption = "Comparison of performance for best standard ML estimator found for each type by parameter grid search."
    label = "table-standardml-estimator-performance-comparison"
    performance_df.to_latex(output_file,
                            index=False,
                            header=True,
                            bold_rows=False,
                            float_format="%0.4f",
                            caption=caption,
                            label=label,
                            longtable=True,
                            )


def main():
    """Main entry point for this figure visualizaiton creation
    """
    # parse command line arguments
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('models', nargs='+',
                        help='the name of the model files to process and create table of performance comparisons')
    parser.add_argument('--output', default=None,
                        help='name of output table, defaults to table-models-estimator-performance-comparison.tex')
    args = parser.parse_args()

    # determine output file name if not given explicitly
    model_name = 'standardml'
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'table-' + model_name + '-estimator-performance-comparison' + '.tex'
        output_file = os.path.join(table_dir, output_file)


    # generate and save the table for the asked for models
    model_names, _, best_estimators, best_scores, best_params, _ = load_and_combine_model_results(args.models)
    performance_df = generate_performance_comparison_df(model_names, best_estimators, best_scores, best_params)
    save_table(performance_df, output_file)


if __name__ == "__main__":
    main()
