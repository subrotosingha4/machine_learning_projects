PANDOC_FLAGS=
out := _out
cache := $(out)/.cache
build := $(out)/pdf
mkdir = @mkdir -p $(dir $@)

#TEXINPUTS=".:$(cache):" pdflatex -output-directory="$(cache)" -interaction=nonstopmode -file-line-error $< | grep ":[^:]*:" | grep -v "natbib Warning: Citation"
define pdflatex
TEXINPUTS=".:$(cache):" pdflatex -output-directory="$(cache)" -interaction=nonstopmode -file-line-error $< | grep "Output"
endef

define biber
biber  $(basename $< .tex) | grep "Output"
endef

define rubber
rubber -m pdftex --inplace
endef

define make-depend
../src/deps.py $@ < $< > $(cache)/$*.d
endef


## List of all valid targets in this project:
## ------------------------------------------
## all        : by default generate all pdf documents in this repository.
.PHONY : all
all : $(build)/mindwandering-deeplearning-classifier.pdf $(build)/appendix.pdf


## mindwandering-deeplearning-classifier.pdf
##            : Generate mindwandering-deeplearning-classifier.pdf from markdown/tex templates.
##
mindwandering-deeplearning-classifier-docs := mindwandering-deeplearning-classifier.tex
mindwandering-deeplearning-classifier-bibs := mindwandering-deeplearning-classifier.bib
mindwandering-deeplearning-classifier-figures := ../figures/figure-standardml-aucroc-histogram.png ../figures/figure-standardml-combined-best-aucroc-curves.png
mindwandering-deeplearning-classifier-tables := ../tables/table-standardml-estimator-performance-comparison.tex
mindwandering-deeplearning-classifier-src := $(mindwandering-deeplearning-classifier-docs) $(mindwandering-deeplearning-classifier-bibs) $(mindwandering-deeplearning-classifier-figures) $(mindwandering-deeplearning-classifier-tables)
mindwandering-deeplearning-classifier-dest := $(addprefix $(cache)/, $(mindwandering-deeplearning-classifier-src))
$(cache)/mindwandering-deeplearning-classifier.tex : PANDOC_FLAGS+=-s -N
$(build)/mindwandering-deeplearning-classifier.pdf : $(mindwandering-deeplearning-classifier-dest)


## appendix.pdf
##            : Generate appendix.pdf from markdown/tex templates.
##
appendix-docs := appendix.tex
appendix-bibs := mindwandering-deeplearning-classifier.bib
appendix-figures := ../figures/figure-standardml-aucroc-histogram.png \
	../figures/figure-standardml-combined-best-aucroc-curves.png \
	../figures/figure-replication-LogisticRegression-confusion-matrix.png \
	../figures/figure-replication-LogisticRegression-recall-precision.png \
	../figures/figure-replication-LogisticRegression-aucroc-curve.png \
	../figures/figure-replication-kNN-confusion-matrix.png \
	../figures/figure-replication-kNN-recall-precision.png \
	../figures/figure-replication-kNN-aucroc-curve.png \
	../figures/figure-replication-SVM-confusion-matrix.png \
	../figures/figure-replication-SVM-recall-precision.png \
	../figures/figure-replication-SVM-aucroc-curve.png \
	../figures/figure-replication-DecisionTree-confusion-matrix.png \
	../figures/figure-replication-DecisionTree-recall-precision.png \
	../figures/figure-replication-DecisionTree-aucroc-curve.png \
	../figures/figure-replication-RandomForest-confusion-matrix.png \
	../figures/figure-replication-RandomForest-recall-precision.png \
	../figures/figure-replication-RandomForest-aucroc-curve.png \
	../figures/figure-replication-NaiveBayes-confusion-matrix.png \
	../figures/figure-replication-NaiveBayes-recall-precision.png \
	../figures/figure-replication-NaiveBayes-aucroc-curve.png
appendix-tables := ../tables/table-standardml-estimator-performance-comparison.tex \
	../tables/table-standardml-grid-search-parameter-summary.tex
appendix-src := $(appendix-docs) $(appendix-bibs) $(appendix-figures) $(appendix-tables)
appendix-dest := $(addprefix $(cache)/, $(appendix-src))
$(cache)/appendix.tex : PANDOC_FLAGS+=-s -N
$(build)/appendix.pdf : $(appendix-dest)

$(cache)/%.tex : %.pmd
	@echo "\n\n"
	@echo "************ Weaving Markdown: $< **************"
	$(mkdir)
	pweave -f pandoc -o $(cache)/$*.md $<
	pandoc $(PANDOC_FLAGS) -t latex -o $@ $(cache)/$*.md
	$(make-depend)

$(cache)/%.bib : %.bib
	$(mkdir)
	cp $< $@

$(cache)/%.tex : %.tex
	$(mkdir)
	cp $< $@
	$(make-depend)

$(cache)/%.png : %.png
	$(mkdir)
	cp $< $@

$(cache)/%.sty : %.sty
	$(mkdir)
	cp $< $@

$(cache)/%.cls : %.cls
	$(mkdir)
	cp $< $@

$(build)/%.pdf : $(cache)/%.tex
	@echo "\n\n"
	@echo "************ Building PDF: $@ **************"
	$(mkdir)
	#$(rubber) $<
	$(pdflatex)
	$(biber)
	$(pdflatex)
	$(pdflatex)
	cp $(cache)/$*.pdf $@
	cp $(cache)/$*.pdf .



## clean      : Remove auto-generated files for a completely clean rebuild.
.PHONY : clean
clean  :
	rm -f *.aux *.log *.out *.bbl *.blg *.lof *.lot *.toc *.equ *~
	rm -rf $(out)


## help       : Get all build targets supported by this build.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<

