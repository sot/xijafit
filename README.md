# xijafit

This is a tool that can be used to automate the Xija model fitting process via scripting commands. 

A prototype graphical fitting tool is also included for environments that include Jupyter 4.0+, along with ipython widgets (version 5.2.2 tested) and the traitles library (version 4.3.1 tested).

This package is compatible with Python 3.x

Note that this requires either Sherpa 4.7 (Python 2.x only) or Sherpa 4.9 with PR #343 (Python 2.x and 3.x) to avoid a bug that affects fitting with user defined fit statistics.