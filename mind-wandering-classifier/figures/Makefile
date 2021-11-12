RESULT_DIR := ../models
FIGURE_DIR := .
SRC_DIR := .

# figures to generate
figures = \
	figure-standardml-aucroc-histogram \
	figure-standardml-combined-best-aucroc-curves \
	figure-replication-LogisticRegression-confusion-matrix \
	figure-replication-LogisticRegression-recall-precision \
	figure-replication-LogisticRegression-aucroc-curve \
	figure-replication-kNN-confusion-matrix \
	figure-replication-kNN-recall-precision \
	figure-replication-kNN-aucroc-curve \
	figure-replication-SVM-confusion-matrix \
	figure-replication-SVM-recall-precision \
	figure-replication-SVM-aucroc-curve \
	figure-replication-DecisionTree-confusion-matrix \
	figure-replication-DecisionTree-recall-precision \
	figure-replication-DecisionTree-aucroc-curve \
	figure-replication-RandomForest-confusion-matrix \
	figure-replication-RandomForest-recall-precision \
	figure-replication-RandomForest-aucroc-curve \
	figure-replication-NaiveBayes-confusion-matrix \
	figure-replication-NaiveBayes-recall-precision \
	figure-replication-NaiveBayes-aucroc-curve

figure_results := $(patsubst %, $(FIGURE_DIR)/%.png, $(figures))
figure_src := $(patsubst %, $(SRC_DIR)%.py, $(figures))


## List of all valid targets in this sub project:
## ----------------------------------------------
## all        : By default run all figure generation tasks
##              if the figures are out of date.
##
.PHONY : all
all : $(figure_results)


## figure-standardml-aucroc-histogram.png
##            : Create a combined histogram figure of aucroc
##              performance for standard ML estimators.
##
standardml_aucroc_models = \
	replication-LogisticRegression \
	replication-kNN \
	replication-SVM \
	replication-DecisionTree \
	replication-RandomForest \
	replication-NaiveBayes
standardml_aucroc_models_results := $(patsubst %, $(RESULT_DIR)/%.pkl, $(standardml_aucroc_models))

$(FIGURE_DIR)/figure-standardml-aucroc-histogram.png: $(SRC_DIR)/visualize-aucroc-histogram.py $(standardml_aucroc_models_results)
	python $^ --output $@


## figure-standardml-combined-best-aucroc-curves.png
##            : Create a visualization of the combined aucroc
##              scores for multiple best ml classifiers.
##
$(FIGURE_DIR)/figure-standardml-combined-best-aucroc-curves.png: $(SRC_DIR)/visualize-combined-best-aucroc-curves.py $(standardml_aucroc_models_results)
	python $^ --output $@


## figure-%-confusion-matrix.png
##            : Create confusion matrix figure for a trained estimator model.
##
$(FIGURE_DIR)/figure-%-confusion-matrix.png: $(SRC_DIR)/visualize-confusion-matrix.py $(RESULT_DIR)/%.pkl
	python $^ --output $@


## figure-%-recall-precision.png
##            : Create recall vs. precision figure for a trained estimator model.
##
$(FIGURE_DIR)/figure-%-recall-precision.png: $(SRC_DIR)/visualize-recall-precision.py $(RESULT_DIR)/%.pkl
	python $^ --output $@


## figure-%-aucroc-curve.png
##            : Create aucroc curve figure for a trained estimator model.
##
$(FIGURE_DIR)/figure-%-aucroc-curve.png: $(SRC_DIR)/visualize-aucroc-curve.py $(RESULT_DIR)/%.pkl
	python $^ --output $@


## clean      : DANGER: Remove all generated figures.  Figure generation is relatively
##              quick, so actually removing and redoing all figures is not that big a deal
##              if needed.
##
.PHONY : clean
clean  :
	rm -rf $(figure_results)

## help       : Get all build targets supported by this build.
##
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
