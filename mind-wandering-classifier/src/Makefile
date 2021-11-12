## List of all valid targets in this project:
## ------------------------------------------
## all        : By default run all src code tasks
.PHONY : all
all : tests


## tests      : Run mindwandering module and other repository tests
##              to check reusable routines for data cleaning, pipelines
##              and other data set manipulations
.PHONY : tests
tests :
	python test.py


## help       : Get all build targets supported by this build.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
