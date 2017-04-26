.. xijafit documentation master file, created by
   sphinx-quickstart on Mon Apr 24 14:26:53 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Xijafit Documentation
=====================

The Xijafit package provides a suite of methods that can be used to automate the Xija model fitting process.

The included methods enable one to:
 - Load a base model from a file
 - Set or update fit start and stop dates
 - Set initial data for pseudo parameters
 - Freeze, thaw, and set parameters
 - Fit the model using the methods available in Sherpa (e.g. moncar, simplex, etc.)
 - Export new models in JSON format
 - Record and save all relevant parameter and fit information for each fit step

This package does not include commands to build a model from scratch, these commands are already included in the Xija package.


Contents:

.. toctree::
   :maxdepth: 4

   xijafit


Example Fit Process
===================

Load Base Xija Model Spec

  >>> from xijafit import Xijafit
  >>> newmodel = XijaFit('example_data/tcylaft6_model_spec_roll_base.json', set_data_exprs=(u'tcylaft6_0=22.0',), start='2013:300', stop='2016:300', quiet=False, name='tcylaft6')


Zero out all long term solarheat parameters. Thaw only the short term solarheat parameters, along with the solarheat seasonal effect parameter and a pseudo node coupling, then fit. This step and the next step fit for the day to day thermal effects.

  >>> newmodel.zero_solarheat_dp()
  >>> newmodel.freeze_all()
  >>> newmodel.thaw_solarheat_p()
  >>> newmodel.thaw_solarheat_roll()
  >>> newmodel.thaw_param(u'coupling__tcylaft6__tcylaft6_0__tau')
  >>> newmodel.thaw_param(u'solarheat__tcylaft6_0__ampl')
  >>> newmodel.fit(method='moncar')


Thaw only the heatsink parameters and fit.

  >>> newmodel.freeze_all()
  >>> newmodel.thaw_param(u'heatsink__tcylaft6_0__T')
  >>> newmodel.thaw_param(u'heatsink__tcylaft6_0__tau')
  >>> newmodel.fit(method='moncar')


Set the bounds on the long term solarheat parameters to the desired range. Thaw only the long term solarheat parameters, along with the solarheat seasonal effect parameter, then fit. This step fits for the long term heat-up.

  >>> newmodel.freeze_all()
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_45', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_60', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_80', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_90', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_100', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_110', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_120', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_130', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_140', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_150', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_160', -0.5, 0.5)
  >>> newmodel.set_range('solarheat__tcylaft6_0__dP_180', -0.5, 0.5)
  >>> newmodel.thaw_solarheat_dp()
  >>> newmodel.thaw_param(u'solarheat__tcylaft6_0__ampl')
  >>> newmodel.fit(method='moncar')


Thaw only the short term solarheat parameters, along with the pseudo node coupling, then fit. This fine tunes the short term heating now that the long term heat-up is accounted for.

  >>> newmodel.freeze_all()
  >>> newmodel.thaw_solarheat_p()
  >>> newmodel.thaw_solarheat_roll()
  >>> newmodel.thaw_param(u'coupling__tcylaft6__tcylaft6_0__tau')
  >>> newmodel.fit(method='moncar')


Re-fit the long term solarheat parameters again. This further refines the long term heat-up now that the short term heating effects are refined.

  >>> newmodel.freeze_all()
  >>> newmodel.thaw_solarheat_dp()
  >>> newmodel.fit(method='moncar')


Save all relevant information for future use and reference.

  >>> newmodel.write_spec_file()
  >>> newmodel.write_snapshots_file('tcylaft6_fit_snapshots.json')



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

