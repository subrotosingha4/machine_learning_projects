RESULT_DIR := .
MODEL_DIR := .

# model results
models = replication-LogisticRegression \
	replication-kNN \
	replication-SVM \
	replication-DecisionTree \
	replication-RandomForest \
	replication-NaiveBayes
model_results := $(patsubst %, $(RESULT_DIR)/%.pkl, $(models))
model_src := $(patsubst %, $(MODEL_DIR)%.py, $(models))


## List of all valid targets in this sub project:
## ----------------------------------------------
## all        : By default run all modeling tasks to generate
##              model results if the results are out of date.
##
.PHONY : all
all : $(model_results)


## model      : Can retrain and create individual model results by
##              doing a make model.pkl, e.g. make replication-01-LogisticRegression.pkl
##
$(RESULT_DIR)/%.pkl : $(MODEL_DIR)/%.py
	python $<

## clean      : DANGER: Remove all modeling result files, can take a while to
##              regenerate all results, so only do this if really want a completely
##              clean replication of all results
##
.PHONY : clean
clean  :
	rm -rf $(model_results)

## help       : Get all build targets supported by this build.
##
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
