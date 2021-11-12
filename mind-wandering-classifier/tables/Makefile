RESULT_DIR := ../models
TABLE_DIR := .
SRC_DIR := .

# tables to generate
tables = \
	table-standardml-estimator-performance-comparison \
	table-standardml-grid-search-parameter-summary

table_results := $(patsubst %, $(TABLE_DIR)/%.tex, $(tables))


## List of all valid targets in this sub project:
## ----------------------------------------------
## all        : By default run all table generation tasks
##              if the tables are out of date.
##
.PHONY : all
all : $(table_results)



## table-standardml-estimator-performance-comparison.tex
##            : Generate a table comparing the best estimator
##              performance results achieved for search of standard ml
##              trained estimators.
##
standardml_aucroc_models = \
	replication-LogisticRegression \
	replication-kNN \
	replication-SVM \
	replication-DecisionTree \
	replication-RandomForest \
	replication-NaiveBayes
standardml_aucroc_models_results := $(patsubst %, $(RESULT_DIR)/%.pkl, $(standardml_aucroc_models))

$(TABLE_DIR)/table-standardml-estimator-performance-comparison.tex: $(SRC_DIR)/generate-estimator-performance-comparison.py $(standardml_aucroc_models_results)
	python $^ --output $@


## table-standardml-grid-search-parameter-summary.tex
##            : Generate a table summarizing all parameters used
##              in the transformers and estimators for the grid
##              search pipelines.
##
$(TABLE_DIR)/table-standardml-grid-search-parameter-summary.tex: $(SRC_DIR)/generate-grid-search-parameter-summary.py $(standardml_aucroc_models_results)
	python $^ --output $@


## clean      : DANGER: Remove all generated tables.  Table generation is relatively
##              quick, so actually removing and redoing all tables is not that big a deal
##              if needed.
##
.PHONY : clean
clean  :
	rm -rf $(table_results)

## help       : Get all build targets supported by this build.
##
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
