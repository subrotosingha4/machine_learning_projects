## List of all valid targets in this project:
## ----------------------------------------------
## all        : By default the top level make will
##              perform all tasks create train all models,
##              regenerate all results and data, create
##              all visualizations and figures, and create
##              the final paper document.  This works by
##              build tasks in the correct order in the
##              subprojects.
##
.PHONY : all
#all : tests models figures tables paper
all : models figures tables paper


## tests      : Perform all unit tests, should this part of
##              the normal complete rebuild?
##
.PHONY : tests
tests :
	cd src && $(MAKE)


## models     : Train all data and generate all model
##              results needed for paper.
##
.PHONY : models
models :
	cd models && $(MAKE)

## figures    : Generate all figures and visualizations
##              needed for paper.
##
.PHONY : figures
figures :
	cd figures && $(MAKE)

## tables     : Generate all tables needed for paper.
##
.PHONY : tables
tables :
	cd tables && $(MAKE)

## paper      : Create the paper using results and figures
##              from the subprojects.
##
.PHONY : paper
paper :
	cd paper && $(MAKE)


## clean      : DANGER: Remove all generated build products so can
##              rebuild everything from scratch.  It can take time
##              especially to regenerate model data and results, so
##              use this only when really want a complete clean rebuild
##              of all project data and results.
##
.PHONY : clean
clean  :
	#cd models && $(MAKE) clean && cd ../figures && $(MAKE) clean && cd ../tables && $(MAKE) clean && cd ../paper && $(MAKE) clean
	cd ../figures && $(MAKE) clean && cd ../tables && $(MAKE) clean && cd ../paper && $(MAKE) clean

## help       : Get all build targets supported by this build.
##
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
