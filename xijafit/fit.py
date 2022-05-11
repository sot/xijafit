# !/usr/bin/env python

import ast
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import re
import logging
import io
import os
import sys

try:
    from ipywidgets import widgets
    import traitlets
    from IPython.display import display
    HAS_JUPYTER = True
except ImportError:
    HAS_JUPYTER = False

import sherpa.ui as ui
from Chandra.Time import DateTime
import xija

try:
    import Chandra.taco
    # Enable fully-randomized evaluation of ACIS-FP model which is desirable
    # for fitting.
    Chandra.taco.taco.set_random_salt(None)
except ImportError:
    pass

try:
    import plot_cxctime_custom as plot_cxctime
except ImportError:
    from Ska.Matplotlib import plot_cxctime


# Default configurations for fit methods
sherpa_configs = dict(
    simplex=dict(ftol=1e-3,
                 finalsimplex=0,  # converge based only on length of simplex
                 maxfev=1000),)


class CalcModel(object):
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger

    def __call__(self, parvals, x):
        """This is the Sherpa calc_model function, but in this case calc_model does not
        actually calculate anything but instead just stores the desired parameters.  This
        allows for multiprocessing where only the fit statistic gets passed between nodes.
        """
        self.logger.info('Calculating params:')
        for parname, parval, newparval in zip(self.model.parnames, self.model.parvals, parvals):
            if parval != newparval:
                self.logger.info('  {0}: {1}'.format(parname, newparval))
        self.model.parvals = parvals

        return np.ones_like(x)


class CalcStat(object):
    def __init__(self, model, logger):
        self.model = model
        self.cache_fit_stat = {}
        self.min_fit_stat = None
        self.min_par_vals = self.model.parvals
        self.logger = logger

    def __call__(self, _data, _model, staterror=None, syserror=None, weight=None):
        """Calculate fit statistic for the xija model.  The args _data and _model
        are sent by Sherpa but they are fictitious -- the real data and model are
        stored in the xija model self.model.
        """
        parvals_key = tuple('%.4e' % x for x in self.model.parvals)
        try:
            fit_stat = self.cache_fit_stat[parvals_key]
            self.logger.info('nmass_model: Cache hit %s' % str(parvals_key))
        except KeyError:
            fit_stat = self.model.calc_stat()

            self.logger.info('Fit statistic: %.4f' % fit_stat)
        self.cache_fit_stat[parvals_key] = fit_stat

        if self.min_fit_stat is None or fit_stat < self.min_fit_stat:
            self.min_fit_stat = fit_stat
            self.min_parvals = self.model.parvals

        return fit_stat, np.ones(1)


class XijaFit(object):
    def __init__(self, modelobject, days=180, stop=None, start=None, set_data_exprs=None, set_data=None,
                 inherit_from=None, keep_epoch=False, quiet=False, name=None, snapshotfile=None):
        """Initialize XijaFit class.

        :param filename: Full path of file containing parameters to import
        :param days: Number of days of data to use to fit the model
        :param stop: Stop date for model fit duration
        :param set_data_exprs: Iterable of initial data values in the format: '<comp_name>=<value>'
        :param set_data: Dictionary of initialization parameters in one of the following formats: 
            {<comp_name>: <value>} or {<comp_name>: {'times':<times>, 'data':<value>}}
        :param inherit_from: Full path of file containing parameters to inherit
        :param keep_epoch: Maintain epoch in SolarHeat models (default=recenter on fit interval)
        :param quiet: Suppress screen output
        ;param snapshotfile: json file containing fit snapshots
        """

        self.fit_logger = logging.getLogger('fit')
        self.fit_logger.setLevel(logging.INFO)

        self.sherpa_logger = logging.getLogger('sherpa')
        self.sherpa_logger.setLevel(logging.INFO)

        self.log_capture_string = io.StringIO()
        if quiet == True:
            ch = logging.StreamHandler(self.log_capture_string)
        else:
            ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] (%(processName)-10s) %(message)s')
        ch.setFormatter(formatter)
        self.fit_logger.addHandler(ch)

        self.sherpa_log_capture_string = io.StringIO()
        if quiet == True:
            sch = logging.StreamHandler(self.sherpa_log_capture_string)
        else:
            sch = logging.StreamHandler(sys.stdout)
        sch.setLevel(logging.INFO)
        sformatter = logging.Formatter('[%(levelname)s] (%(processName)-10s) %(message)s')
        sch.setFormatter(sformatter)
        self.sherpa_logger.addHandler(sch)

        # Set initial times.
        if stop and not start:
            self.start = DateTime(DateTime(stop).secs - days * 86400).date[:8]
            self.stop = stop
        elif start and not stop:
            self.start = start
            self.stop = DateTime(DateTime(stop).secs + days * 86400).date[:8]
        elif start and stop:
            self.start = start
            self.stop = stop
            self.days = np.floor((DateTime(stop).secs - DateTime(start).secs) / (3600. * 24.))
        else:
            self.start = DateTime(DateTime().secs - 3600*24*192).date
            self.stop = DateTime(DateTime().secs - 3600*24*10).date

        # Initialize Xija model object.
        self.model = xija.XijaModel(name or 'xijamodel', self.start, self.stop, model_spec=modelobject)

        self.model_spec = self.model.model_spec

        if name:
            self.model_spec['name'] = name

        self.set_data_exprs = set_data_exprs
        if self.set_data_exprs:
            self.set_init_data(set_data_exprs)

        if set_data:
            for key, value in set_data.items():
                if isinstance(value, dict):
                    self.model.comp[key].set_data(value['data'], value['times'])
                else:
                    self.model.comp[key].set_data(value)

        # "make" model.
        self.model.make()

        # Load parameter values from inherited model file where parameter names match.
        if inherit_from:
            self.inherit_param_from(inherit_from)

        # Set epoch
        self.keep_epoch = keep_epoch
        if not self.keep_epoch:
            self.set_epoch()

        if snapshotfile:
            self.snapshots = json.load(open(snapshotfile, 'r'))
        else:
            self.snapshots = []
        self.save_snapshot()

    def update_fit_times(self, start=None, stop=None, days=180):
        """Update model fit times to new values.

        :param start: Start date for model fit duration
        :param stop: Stop date for model fit duration
        :param days: Number of days of data to use to fit the model

        Either 'start' and 'days', or 'stop' and 'days', or both 'start' and 'stop' must be defined,
        accounting for the fact that the default value for days is defined above.

        This re-initializes and runs the Xija model object with new times without changing the parameters.
        """
        if (start is None) and (stop is None):
            raise ValueError(
                "Either 'start' and 'days', or 'stop' and 'days', or both 'start' and 'stop' must be defined.")

        if start and stop:
            self.start = DateTime(start).date
            self.stop = DateTime(stop).date
        elif stop is None:
            self.start = DateTime(start).date
            self.stop = DateTime(DateTime(start).secs + days * 24 * 3600.).date[:8]
        else:
            self.start = DateTime(DateTime(start).secs - days * 24 * 3600.).date[:8]
            self.stop = DateTime(stop).date

        self.model = xija.ThermalModel(self.model_spec['name'], self.start, self.stop, model_spec=self.model.model_spec)

        # These solarheat parameters need to be defined to avoid an AttributeError in heat.py. The actual
        # values don't matter as they are deleted when setting the epoch.
        for key, value in list(self.model.comp.items()):
            if 'solarheat' in key:
                if not hasattr(self.model.comp[key], 't_days'):
                    self.model.comp[key].t_days = -1
                if not hasattr(self.model.comp[key], 't_phase'):
                    self.model.comp[key].t_phase = -1

        if self.set_data_exprs:
            self.set_init_data(self.set_data_exprs)

        if not self.keep_epoch:
            self.set_epoch()

        self.model.make()
        self.model.calc()

    def set_init_data(self, set_data_exprs):
        """Set initial data.

        :param set_data_exprs: Iterable of initial data values in the format: '<comp_name>=<value>'

        This is allows for setting initial data via the command line, where passing a dict would not work.

        NOTE: If fitting routines become scripted, this may not be necessary.
        """
        set_data_vals = {}

        for set_data_expr in set_data_exprs:
            set_data_expr = re.sub('\s', '', set_data_expr)  # remove spaces?
            try:
                comp_name, val = set_data_expr.split('=')
            except ValueError:
                raise ValueError("--set_data must be in form '<comp_name>=<value>'")
            # Set data to value.  ast.literal_eval is a safe way to convert any string literal into the
            # corresponding Python object.
            set_data_vals[comp_name] = ast.literal_eval(val)

        for comp_name, val in list(set_data_vals.items()):
            self.model.comp[comp_name].set_data(val)

    def inherit_param_from(self, filename):
        """Read in file from which to inherit paremeter values.

        :param filename: Full path of file containing parameters to inherit

        This provides a way to construct a model which is similar to an existing model but has
        some differences. All the model parameters which are exactly the same will be taking from
        the inherited model specification.
        """
        inherit_spec = json.load(open(filename, 'r'))
        inherit_pars = {par['full_name']: par for par in inherit_spec['pars']}
        for par in self.model.pars:
            if par.full_name in inherit_pars:
                print("Inheriting par {}".format(par.full_name))
                par.val = inherit_pars[par.full_name]['val']
                par.min = inherit_pars[par.full_name]['min']
                par.max = inherit_pars[par.full_name]['max']
                par.frozen = inherit_pars[par.full_name]['frozen']
                par.fmt = inherit_pars[par.full_name]['fmt']

    def set_epoch(self):
        """Set epoch for fitting session.

        """
        new_epoch = np.mean(self.model.times[[0, -1]])
        for comp in list(self.model.comp.values()):
            if isinstance(comp, xija.SolarHeat):
                comp.epoch = new_epoch
                try:
                    comp.epoch = new_epoch
                except AttributeError as err:
                    assert 'can only reset the epoch' in str(err)

    def fit(self, method='simplex'):
        """Initiate a fit of the model using Sherpa.

        :param method: Method to be used to fit the model (e.g. simplex, levmar, or moncar)
        """
        dummy_data = np.zeros(1)
        dummy_times = np.arange(1)
        ui.load_arrays(1, dummy_times, dummy_data)

        ui.set_method(method)
        ui.get_method().config.update(sherpa_configs.get(method, {}))

        ui.load_user_model(CalcModel(self.model, self.fit_logger), 'xijamod')  # sets global xijamod
        ui.add_user_pars('xijamod', self.model.parnames)
        ui.set_model(1, 'xijamod')

        calc_stat = CalcStat(self.model, self.fit_logger)
        ui.load_user_stat('xijastat', calc_stat, lambda x: np.ones_like(x))
        ui.set_stat(xijastat)

        # Set frozen, min, and max attributes for each xijamod parameter
        for par in self.model.pars:
            xijamod_par = getattr(xijamod, par.full_name)
            xijamod_par.val = par.val
            xijamod_par.frozen = par.frozen
            xijamod_par.min = par.min
            xijamod_par.max = par.max

        ui.fit(1)

        self.save_snapshot(fit_stat=calc_stat.min_fit_stat, method=method)

    def save_snapshot(self, fit_stat=None, method=None):
        """Save a snapshot of fit statistics.

        :param fit_stat: Manual way to pass fit statistic (may not be necessary in future)
        :param method: Manual way to pass method (may not be necessary in future)
        """
        if fit_stat is None:
            fit_stat = self.model.calc_stat()

        pattern = 'Final fit statistic\s*=\s*([0-9.e]+\+?\d*)\D+(\d+)'
        found = re.findall(pattern, self.sherpa_log_capture_string.getvalue())
        if found:
            final_eval_num = found[-1][-1]
        else:
            final_eval_num = None

        snapshot = {}
        for pars in self.model.pars:
            snapshot[pars['full_name']] = {k: pars[k] for k in ('frozen', 'min', 'max', 'val')}

        snapshot['fit_stat'] = fit_stat
        snapshot['final_eval_num'] = final_eval_num
        snapshot['tstart'] = DateTime(self.model.tstart).date
        snapshot['tstop'] = DateTime(self.model.tstop).date
        snapshot['method'] = method
        snapshot['date'] = DateTime().date

        self.snapshots.append(snapshot)

    def freeze_all(self):
        """Freeze all parameters.
        """
        for par in self.model.pars:
            par['frozen'] = True

    def thaw_all(self):
        """Thaw all parameters.
        """
        p1 = 'solarheat__[A-Za-z0-9_]+__bias'
        p2 = 'solarheat__[A-Za-z0-9_]+__tau'
        for par in self.model.pars:
            if re.match(p1, par.full_name):
                pass
            elif re.match(p2, par.full_name):
                pass
            else:
                par['frozen'] = Falseballbal

    def thaw_param(self, param):
        """Thaw specific parameter.

        :param param: Specific parameter to thaw
        """
        found = False
        for par in self.model.pars:
            if param in par.full_name:
                par['frozen'] = False
                found = True
        if not found:
            print(('Parameter: {} not found'.format(param)))

    def freeze_param(self, param):
        """Freeze specific parameter.

        :param param: Specific parameter to freeze
        """
        found = False
        for par in self.model.pars:
            if param in par.full_name:
                par['frozen'] = True
                found = True
        if not found:
            print(('Parameter: {} not found'.format(param)))

    def set_param(self, param, value):
        """Set parameter to specific value.

        :param param: Specific parameter to set
        :param value: parameter value
        """
        found = False
        for par in self.model.pars:
            if param in par.full_name:
                par['val'] = value
                if value > par['max']:
                    par['max'] = value
                elif value < par['min']:
                    par['min'] = value
                found = True
        if not found:
            print(('Parameter: {} not found'.format(param)))

    def center_range(self, param, expansion=1.0):
        """Center parameter range around current value.

        :param param: Parameter
        :param expansion: ratio of min-max range, used to expand or contract range
        """
        found = False
        for par in self.model.pars:
            if param in par.full_name:
                vrange = par['max'] - par['min']
                par['min'] = par['val'] - expansion * vrange / 2.0
                par['max'] = par['val'] + expansion * vrange / 2.0
                found = True
        if not found:
            print(('Parameter: {} not found'.format(param)))

    def set_range(self, param, minval, maxval):
        """Center parameter range around current value.

        :param param: Parameter
        :param minval: minimum range value
        :param maxval: maximum range value
        """
        found = False
        for par in self.model.pars:
            if param in par.full_name:
                par['min'] = minval
                par['max'] = maxval
                if par['val'] > par['max']:
                    par['val'] = maxval
                elif par['val'] < par['min']:
                    par['val'] = minval
                found = True
        if not found:
            print(('Parameter: {} not found'.format(param)))

    def zero_solarheat_p(self):
        """Set all short term solarheat "P" parameters zero.
        """
        p1 = 'solarheat__[A-Za-z0-9_]+__P_\d+'
        self.bias = 'solarheat__[A-Za-z0-9_]+__bias'
        p2 = self.bias
        found = False
        for par in self.model.pars:
            if (re.match(p1, par.full_name)) or (re.match(p2, par.full_name)):
                par['val'] = 0.0
                par['min'] = -1.0
                par['max'] = 1.0
                found = True
        if not found:
            print('Solarheat "P" parameters not found')

    def zero_solarheat_dp(self):
        """Set all long term solarheat "dP" parameters zero.
        """
        p = 'solarheat__[A-Za-z0-9_]+__dP_\d+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['val'] = 0.0
                par['min'] = -1.0
                par['max'] = 1.0
                found = True
        if not found:
            print('Solarheat "dP" parameters not found')

    def zero_solarheat_roll(self):
        """Set all solarheat roll parameters zero.
        """
        p = 'solarheat_off_nom_roll.*'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['val'] = 0.0
                par['min'] = -1.0
                par['max'] = 1.0
                found = True
        if not found:
            print('Solarheat "roll" parameters not found')

    def freeze_solarheat_p(self):
        """Freeze all solarheat "P" parameters.
        """
        p = 'solarheat__[A-Za-z0-9_]+__P_\d+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = True
                found = True
        if not found:
            print('Solarheat "P" parameters not found')

    def freeze_solarheat_dp(self):
        """Freeze all solarheat "dP" parameters.
        """
        p = 'solarheat__[A-Za-z0-9_]+__dP_\d+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = True
                found = True
        if not found:
            print('Solarheat "dP" parameters not found')

    def freeze_solarheat_roll(self):
        """Freeze all solarheat roll parameters.
        """
        p = 'solarheat_off_nom_roll.*'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = True
                found = True
        if not found:
            print('Solarheat "roll" parameters not found')

    def thaw_solarheat_p(self):
        """Thaw all solarheat "P" parameters.
        """
        p = 'solarheat__[A-Za-z0-9_]+__P_\d+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = False
                found = True
        if not found:
            print('Solarheat "P" parameters not found')

    def thaw_solarheat_dp(self):
        """Thaw all solarheat "dP" parameters.
        """
        p = 'solarheat__[A-Za-z0-9_]+__dP_\d+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = False
                found = True
        if not found:
            print('Solarheat "dP" parameters not found')

    def thaw_solarheat_roll(self):
        """Thaw all solarheat roll parameters.
        """
        p = 'solarheat_off_nom_roll.*'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = False
                found = True
        if not found:
            print('Solarheat "roll" parameters not found')

    def freeze_dpa_power_bins(self):
        """Freeze all dpa power parameters.
        """
        p = 'dpa_power__pow_[0-9x]+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = True
                found = True
        if not found:
            print('DPA power bin parameters not found')

    def thaw_dpa_power_bins(self):
        """Thaw all dpa power parameters.
        """
        p = 'dpa_power__pow_[0-9x]+'
        found = False
        for par in self.model.pars:
            if re.match(p, par.full_name):
                par['frozen'] = False
                found = True
        if not found:
            print('DPA power bin parameters not found')

    def _backup_current_filename(self, filename):
        """Create a backup of the current filename (if necessary) before it is overwritten.

        :param filename: original intended filename

        This requires that there be a file extension.
        """

        filename_parts = re.findall('(.*)\.(.+)', filename)[0]

        if os.path.isfile(filename):
            files = [f for f in os.listdir('.') if re.search(r'{}\..+-\d+'.format(filename_parts[0]), f)]

            if files:
                # In this case the "dash number" structure has been observed indicating existing backups
                file_extension_structures = [re.findall('.*\.(.+)-(\d+)', f)[0] for f in files]
                max_version = max([int(t[1]) for t in file_extension_structures])
                base_file_extension = file_extension_structures[0][0]
                new_filename = '{}.{}-{}'.format(filename_parts[0], base_file_extension, max_version + 1)
                os.rename(filename, new_filename)

            else:
                # In this case the "dash number" structure has not been used
                new_filename = '{}.{}-{}'.format(filename_parts[0], filename_parts[1], 1)
                os.rename(filename, new_filename)

    def write_spec_file(self, filename=None, overwrite=False):
        """Write model definition to file.

        :param filename: filename to use for model definition file.
        """

        if not filename:
            filename = "{}_model_spec.json".format(self.model.name)

        if not overwrite:
            self._backup_current_filename(filename)

        self.model.write(filename)

    def write_snapshots_file(self, filename=None, overwrite=False):
        """Write fitting snapshots to file.

        :param filename: filename to use for fitting snapshots file
        :param overwrite: flag indicting whether or not it is ok to overwite the filename if it already exists
        """

        if not filename:
            filename = "{}_fit_snapshots.json".format(self.model.name)

        if not overwrite:
            self._backup_current_filename(filename)

        with open(filename, 'w') as outfile:
            json.dump(self.snapshots, outfile, indent=4, sort_keys=True)


def load_parameters_from_snapshot(model, snapshot):
    """Load parameters from a previous model.

    :param model: xija model object
    :param snapshot: previous model fit snapshot

    """
    def setval(pars, param, value):
        for par in pars:
            if param in par.full_name:
                par['val'] = value
                if value > par['max']:
                    par['max'] = value
                elif value < par['min']:
                    par['min'] = value

    for p in list(snapshot.keys()):
        if p in model.parnames:
            setval(model.pars, p, snapshot[p])

    return model.model_spec


class XijaParamGui(object):
    """Class for creating Jupyter based GUI for manually tweaking parameters.

    This is not currently 100% functional. Parameter values don't always set to their new values, and modifying ranges
    can have unexpected effects on values (e.g. resetting them to original values). The option to interactively view the
    impact a modified parameter has on the model should also be included.

    """
    def __init__(self, model, msid, caution_high=None, planning_limit=None):
        if not HAS_JUPYTER:
            raise ImportError('required imports for Jupyter widgets not available')

        self.model = model
        self.msid = msid
        self.caution_high = caution_high
        self.planning_limit = planning_limit

        self.fig = plt.figure(figsize=(15, 8))
        self.ax = self.fig.add_axes(rect=(0.05, 0.1, 0.8, 0.85))

        self.plothelper()

        minlabel = "               Min".replace('', '&nbsp', )
        maxlabel = "                                                           Max".replace('', '&nbsp', )
        vallabel = "                       Value".replace('', '&nbsp', )
        self.label = widgets.HTML(value=vallabel + minlabel + maxlabel)

        params = []
        for p in self.model.model.pars:
            params.append(self.param_object(p))

        display(self.label, *params)

    def param_object(self, updateparam):

        def onvalchange(p):
            # print ("parameter1: {}".format(p))
            updateparam['val'] = valtext.value
            sleep(1)
            self.plothelper()
            # print (self.updateparam)

        def onmaxchange(p):
            sleep(1)
            updateparam['max'] = maxtext.value

        def onminchange(p):
            sleep(1)
            updateparam['min'] = mintext.value

        def onfrozenchange(p):
            # print ("parameter1: {}".format(p))
            updateparam['frozen'] = frozen.value
            # print (self.updateparam)

        minbound = updateparam['min']
        maxbound = updateparam['max']
        initialvalue = updateparam['val']

        label = widgets.HTML(value=updateparam['full_name'], padding='5px')
        slider = widgets.FloatSlider(value=initialvalue,
                                     min=minbound,
                                     max=maxbound,
                                     step=0.001,
                                     description='',
                                     orientation='horizontal',
                                     readout=False,
                                     padding='5px')
        slider.msg_throttle = 0

        mintext = widgets.BoundedFloatText(value=minbound,
                                           min=-1e20,
                                           max=maxbound,
                                           description=' ',
                                           readout_format='.3f',
                                           padding='5px')
        mintext.observe(onminchange, names='value')

        maxtext = widgets.BoundedFloatText(value=maxbound,
                                           min=minbound,
                                           max=+1e20,
                                           description=' ',
                                           readout_format='.3f',
                                           padding='5px')
        maxtext.observe(onmaxchange, names='value')

        valtext = widgets.BoundedFloatText(value=slider.value,
                                           min=slider.min,
                                           max=slider.max,
                                           description=' ',
                                           readout_format='.3f',
                                           padding='5px')
        valtext.observe(onvalchange, names='value')

        frozen = widgets.Checkbox(description=' ', value=updateparam['frozen'], visible=True, padding='5px', height=20,
                                  width=20)
        frozen.observe(onfrozenchange, names='value')

        lmin = traitlets.dlink((mintext, 'value'), (slider, 'min'))
        lmax = traitlets.dlink((maxtext, 'value'), (slider, 'max'))

        vlmin = traitlets.dlink((mintext, 'value'), (valtext, 'min'))
        vlmax = traitlets.dlink((maxtext, 'value'), (valtext, 'max'))

        lval = traitlets.link((slider, 'value'), (valtext, 'value'))

        return widgets.HBox(children=[label, valtext, mintext, slider, maxtext, frozen])


    def plothelper(self):
        self.model.model.calc()
        data = self.model.model.get_comp(self.msid)
        self.ax = plt.gca()
        self.ax.cla()
        _ = plot_cxctime(data.times, data.mvals * 9 / 5. + 32., color='r', fig=self.fig, ax=self.ax)
        _ = plot_cxctime(data.times, data.dvals * 9 / 5. + 32., color='b', fig=self.fig, ax=self.ax)
        _[-1].grid(True)

