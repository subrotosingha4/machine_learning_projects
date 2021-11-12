#!/usr/bin/env python
"""Generate a LaTeX table summarizing all grid search parameter values
used in search for good parameters and estimators.

"""
import os
import re
import sys
import argparse
import pandas as pd
# add mindwandering module into python sys path
sys.path.append('../src')
from mindwandering.train import load_model_results


# other global constants / locations.  parameterize these if we need
# flexibility to specify them on command line or move them around
table_dir = '.'
description = """
This script will create a table summarizing all of the parameter values
used in the grid searches for model estimators.  This script
parses the result dataframes from the grid searches, extacts
the parameters used in the training pipelines, and formats as a LaTeX
table to be inserted into documents as needed.
"""

pipeline_name_map = {
    'balance': 'ClassImbalanceTransformer',
    'features': 'SelectKBest',
    'outlier': 'WinsorizationOutlierTransformer',
    'scaling': 'FeatureScalerTransformer',
    'vif': 'VIFThresholdTransformer'
}

model_name_map = {
    'LogisticRegression': 'LogisticRegression',
    'kNN': 'KNeighborsClassifier',
    'SVM': 'SVC',
    'DecisionTree': 'DecisionTreeClassifier',
    'RandomForest': 'RandomForestClassifier',
    'NaiveBayes': 'GaussianNB',
}


def list_to_str(values):
    """Given a list of objects, which are grid search values, convert
    to a single string that will work fine as a display in a table.
    We convert and format some specific types of values like functions
    and float values.
    """
    result = None
    for value in values:
        # format floats with only 2 decimals 
        if type(value) == float:
            value = "%0.2f" % value
            
        # extract function names for function parameters
        if callable(value):
            value = str(value)
            m = re.search(r'<function (.*?) .*>', value)
            value = m.group(1)
        
        # now build the string to be returned
        if result is None:
            result = str(value)
        else:
            result += ', ' + str(value)
            
    return result


def generate_grid_serach_parameter_summary_df(models):
    """Create a summary table of all grid search parameters used in the
    transformers and estimators pipelines for the search.

    Parameters
    ----------
    models - A list of full path names to the model files to process and
        extract the grid search training parameters from.

    Returns
    -------
    params_df - Returns a dataframe with the constructed summary of the
        grid search parameters.
    """
    df_params = pd.DataFrame(columns=['Model', 'Pipeline', 'Parameter', 'Search Values'])
    
    for model in models:
        # get the model results dataframe
        df_result, best_estimator, best_score, best_params, best_index = load_model_results(model) 
        
        # determine model name from the file name
        model_name = os.path.basename(model)
        model_name = os.path.splitext(model_name)[0]
        model_name = model_name.split('-')[1]
        
        # find parameters and insert into the df_params dataframe
        for param in df_result.columns:
            m = re.search(r'param_+(.*?)_+(.*)', param)
            if m:
                pipeline_name, parameter_name = m.groups()
                values = df_result[param].unique()
                if pipeline_name == 'model':
                    pipeline_name = model_name_map[model_name]
                else:
                    pipeline_name = pipeline_name_map[pipeline_name]
                row = {
                    'Model': model_name,
                    'Pipeline': pipeline_name,
                    'Parameter': parameter_name,
                    'Search Values': list_to_str(values)
                }
                df_params = df_params.append(row, ignore_index=True)
                
    df_params = df_params.set_index(['Model', 'Pipeline', 'Parameter'])            
    return df_params


def save_table(params_df, output_file):
    """Create and save a generated LaTeX table of this grid search parameter
    summary information.

    Parameters
    ----------
    params_df - A dataframe of the estimator performance information
      to save
    output_file - The name of the file to save the table into.
    """
    caption = "Summary of all pipeline transformer and estimator parameters used in grid search to determine best estimator parameters."
    label = "table-standardml-grid-search-parameter-summary"
    params_df.to_latex(output_file,
                       index=True,
                       header=True,
                       bold_rows=False,
                       float_format="%0.2f",
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
                        help='the name of the model files to process and create summary table of grid search parameters')
    parser.add_argument('--output', default=None,
                        help='name of output table, defaults to table-models-grid-search-parameter-summary.tex')
    args = parser.parse_args()

    # determine output file name if not given explicitly
    model_name = 'standardml'
    output_file = args.output
    if output_file is None:
        # make full output file name, assume .png output by default
        output_file = 'table-' + model_name + '-grid-search-parameter-summary' + '.tex'
        output_file = os.path.join(table_dir, output_file)


    # generate and save the table for the asked for models
    params_df = generate_grid_serach_parameter_summary_df(args.models)
    save_table(params_df, output_file)


if __name__ == "__main__":
    main()
