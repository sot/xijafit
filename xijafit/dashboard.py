
from hashlib import md5
import json
from urllib.request import urlopen

import numpy as np
import matplotlib
import matplotlib.patheffects as path_effects
from xija.limits import get_limit_color, get_limit_spec
import xija
from Chandra.Time import DateTime

if 'k' not in matplotlib.rcParams['text.color']:
    matplotlib.rcParams['axes.facecolor'] = [.1,.15,.2]

matplotlib.rcParams['xtick.major.pad'] = 5
matplotlib.rcParams['ytick.major.pad'] = 5

plt = matplotlib.pyplot


def c_to_f(temp):
    """ Convert Celsius to Fahrenheit.
    :param temp: Temperature in Celsius
    :type temp: int or float or tuple or list or np.ndarray
    :return: Temperature in Fahrenheit
    :rtype: int or float or list or np.ndarray
    """
    if type(temp) is list or type(temp) is tuple:
        return [c * 1.8 + 32 for c in temp]
    else:
        return temp * 1.8 + 32.0


def getQuantPlotPoints(quantstats, quantile):
    """ Calculate the error quantile line coordinates for the data in each telemetry count value.

    :param quantstats: output from calcquantstats()
    :param quantile: quantile (string - e.g. "q01"), used as a key into quantstats datastructure
    :returns: coordinates for error quantile line

    This is used to calculate the quantile lines plotted on the telemetry vs. error plot (axis 3)
    enclosing the data (i.e. the 1 and 99 percentile lines).

    """

    Tset = [T for (n, T) in quantstats['key']] # Set of temperature values in order
    diffTset = np.diff(Tset) # deltas
    Tpoints = Tset[:-1] + diffTset / 2
    Tpoints = list(np.array([Tpoints, Tpoints]).T.flatten())
    Tpoints.insert(0, Tset[0] - diffTset[0] / 2)
    Tpoints.append(Tset[-1] + diffTset[0] / 2)
    Epoints = [quantstats[num][quantile] for (num, T) in quantstats['key']]
    Epoints = np.array([Epoints, Epoints]).T.flatten()
    return (Epoints, Tpoints)


def calcquantiles(errors):
    """ Calculate the error quantiles.

    :param error: model errors (telemetry - model)
    :returns: datastructure that includes errors (input) and quantile values

    """

    esort = np.sort(errors)
    q99 = esort[int(0.99 * len(esort) - 1)]
    q95 = esort[int(0.95 * len(esort) - 1)]
    q84 = esort[int(0.84 * len(esort) - 1)]
    q50 = np.median(esort)
    q16 = esort[int(0.16 * len(esort) - 1)]
    q05 = esort[int(0.05 * len(esort) - 1)]
    q01 = esort[int(0.01 * len(esort) - 1)]
    stats = {'error': errors, 'q01': q01, 'q05': q05, 'q16': q16, 'q50': q50,
             'q84': q84, 'q95': q95, 'q99': q99}
    return stats


def calcquantstats(T_telem, error, bin_size=None):
    """ Calculate error quantiles for individual telemetry temperatures (each count individually).

    :param T_telem: telemetry values
    :param error: model error (telemetry - model)
    :returns: coordinates for error quantile line

    This is used for the telemetry vs. error plot (axis 3).

    """
    
    def calc_ind(T, T_telem, band=None):
        if band is None:
            return T_telem == T
        else:
            return (T_telem > (T - band)) & (T_telem < (T + band))

    Tquant = {'key': []}
    k = -1

    if bin_size is None:
        Tset = np.sort(list(set(T_telem)))
    else:
        range = max(T_telem) - min(T_telem)
        num = int(np.ceil(range / bin_size))
        Tset = np.linspace(min(T_telem), max(T_telem) + bin_size, int(num))
        
    for T in Tset:
        ind = calc_ind(T, T_telem, band=bin_size)
        if sum(ind) >= 20:
            k = k + 1
            Tquant['key'].append([k, T])
            errvals = error[ind]
            Tquant[k] = calcquantiles(errvals)

    return Tquant


def digitize_data(T_telem, nbins=50):
    """ Digitize telemetry.

    :param T_telem: telemetry values
    :param nbins: number of bins
    :returns: coordinates for error quantile line

    """

    # Bin boundaries
    # Note that the min/max range is expanded to keep all telemetry within the outer boundaries.
    # Also the number of boundaries is 1 more than the number of bins.
    bins = np.linspace(min(T_telem) - 1e-6, max(T_telem) + 1e-6, nbins + 1)
    inds = np.digitize(T_telem, bins) - 1
    means = bins[:-1] + np.diff(bins) / 2

    return np.array([means[i] for i in inds])


def get_local_model(filename):
    """ Load parameters for a single Xija model.

    :param filename: File path or url to local model specification file
    :type filename: str
    :return: Model spec as a dictionary, md5 hash of model spec
    :rtype: tuple
    """

    if 'https://' in filename:
        with urlopen(filename) as url:
            response = url.read()
            f = response.decode('utf-8')
    else:
        with open(filename) as fid:
            f = fid.read()

    return json.loads(f), md5(f.encode('utf-8')).hexdigest()


def run_model(msid, t0, t1, model_spec_file, init={}):
    """ Create and run a Xija model

    This function creates a Xija model object with initial parameters, if any. This function is intended to create a
    streamlined method to creating Xija models that can take both single value data and time defined data
    (e.g. [pitch1, pitch2, pitch3], [time1, time2, time3]), defined in the `init` dictionary.

    :param msid: Primary MSID for model; in this case it can be anything as it is only being used to name the model,
           however keeping the convention to name the model after the primary MSID being predicted reduces confusion
    :type msid: str
    :param t0: Start time for model prediction; this can be any format that cxotime.CxoTime accepts
    :type t0: str or float or int
    :param t1: End time for model prediction; this can be any format that cxotime.CxoTime accepts
    :type t1: str or float or int
    :param model_spec_file: Xija model parameter filename
    :type model_spec_file: str
    :param init: Dictionary of Xija model initialization parameters, can be empty
    :type init: dict
    :rtype: xija.model.XijaModel

    Example::

        model_specs = load_model_specs()
        init = {'1dpamzt': 35., 'dpa0': 35., 'eclipse': False, 'roll': 0, 'vid_board': True, 'pitch':155,
                'clocking': True, 'fep_count': 5, 'ccd_count': 5, 'sim_z': 100000}
        model = run_model('1dpamzt', '2019:001:00:00:00', '2019:010:00:00:00', 'dpa_spec.json', init)

    Notes:

     - Any parameters not specified in `init` will need to be pulled from telemetry

    """

    model_spec, model_spec_md5 = get_local_model(model_spec_file)
    model = xija.ThermalModel(msid, start=t0, stop=t1, model_spec=model_spec)

    for key, value in init.items():
        if isinstance(value, dict):
            model.comp[key].set_data(value['data'], value['times'])
        else:
            model.comp[key].set_data(value)

    model.make()
    model.calc()

    return model, model_spec_md5


def make_dashboard(model_spec_file, t0, t1, init={}, modelname='PSMC', msid='1pdeaat', errorplotlimits=None,
                   yplotlimits=None, bin_size=None, fig=None, savefig=True, legend_loc='best', filter_fcn=None,
                   units='C'):
    """ Generate a watermarked Xija model dashboard

    :param model_spec_file: File location for Xija model definition
    :param t0: Start Time (seconds or HOSC date string)
    :param t1: Stop Time (seconds or HOSC date string)
    :param init: Dictionary of Xija model initialization parameters, can be empty (e.g. {'1dpamzt': 35., 'dpa0': 35.})
    :param modelname: Name of model (e.g. "ACA")
    :param msid: msid name (e.g. "aacccdpt")
    :param errorplotlimits: list or tuple of min and max x axis plot boundaries for both righthand
           plots (optional)
    :param yplotlimits: list or tuple of min and max y axis plot boundaries for both top half
           plots (optional)
    :param bin_size: int or float of desired bin size for 1% and 99% quantile calculations for scatter plot,
           defaults to using telemetry count values if bin_size is left as None (optional)
    :param fig:  Figure object to use, if None, a new figure object is generated (optional)
    :param savefig: Option to automatically save the figure image (optional)
    :param legend_loc: value to be passed to the 'loc' keyword in the  matplotlib pyplot legend
           method, if None, then no legend is displayed (optional)
    :param filter_fcn: User defined function that takes a Xija model object and returns a boolean filtering array
    :param units: String indicating units, used to convert to Fahrenheit if "f" is observed somewhere in the string

    Note:
    The filter_fcn() function must return a boolean numpy array with a length that matches the model and telemetry data
    stored in the model object created by the run_model() function. This boolean array defines which elements to keep
    with a True value, and which elements to ignore with the False value.

    Example filter_fcn:
        def keep_highs(model):
            msiddata = model.get_comp('pm2thv1t')
            keep = msiddata.dvals > 90 # Celsius
            print(f'{sum(keep)} values kept out of {len(keep)}')
            return keep

    """

    model_object, md5_hash = run_model(msid, t0, t1, model_spec_file, init=init)
    msiddata = model_object.get_comp(msid)

    keep = np.zeros_like(msiddata.times) < 1
    if callable(filter_fcn):
        keep = filter_fcn(model_object)

    prediction = msiddata.mvals.astype(np.float64)[keep]
    times = msiddata.times.astype(np.float64)[keep]
    telem = msiddata.dvals.astype(np.float64)[keep]

    if 'f' in units.lower():
        prediction = c_to_f(prediction)
        telem = c_to_f(telem)

    model_limits = {}
    if 'limits' in model_object.model_spec.keys():
        if msid in model_object.model_spec['limits'].keys():
            model_limits = model_object.model_spec['limits'][msid]

    dashboard(prediction, telem, times, model_limits, modelname=modelname, msid=msid, errorplotlimits=errorplotlimits,
              yplotlimits=yplotlimits, bin_size=bin_size, fig=fig, savefig=savefig, legend_loc=legend_loc,
              md5_string=md5_hash)

    return model_object


def dashboard(prediction, tlm, times, limits, modelname='PSMC', msid='1pdeaat', errorplotlimits=None, yplotlimits=None,
              bin_size=None, fig=None, savefig=True, legend_loc='best', md5_string=None):
    """ Plot Xija model dashboard.

    :param prediction: model prediction
    :param tlm: telemetry 
    :param times: model/telemetry time values
    :param limits: model limit dict, (e.g. {"units":"C", "odb.caution.high":-12.0,
                                            "odb.warning.high":-14.0})
    :param modelname: Name of model (e.g. "ACA")
    :param msid: msid name (e.g. "aacccdpt")
    :param errorplotlimits: list or tuple of min and max x axis plot boundaries for both righthand
           plots (optional)
    :param yplotlimits: list or tuple of min and max y axis plot boundaries for both top half
           plots (optional)
    :param bin_size: int or float of desired bin size for 1% and 99% quantile calculations for scatter plot,
           defaults to using telemetry count values if bin_size is left as None (optional)
    :param fig:  Figure object to use, if None, a new figure object is generated (optional)
    :param savefig: Option to automatically save the figure image (optional)
    :param legend_loc: value to be passed to the 'loc' keyword in the  matplotlib pyplot legend
           method, if None, then no legend is displayed (optional)
    :param md5_string: MD5 hash of model file

    Note: prediction, tlm, and times must all have the same number of values.

    """

    # Set some plot characteristic default values
    matplotlib.rc('font', family='sans-serif')
    matplotlib.rc('font', weight='light')

    error = tlm - prediction
    stats = calcquantiles(error)

    # In this case the data is not discretized to a limited number of count values, or has too
    # many possible values to work with calcquantstats(), such as with tlm_fep1_mong.
    if len(np.sort(list(set(tlm)))) > 1000:
        quantized_tlm = digitize_data(tlm)
        quantstats = calcquantstats(quantized_tlm, error, bin_size=bin_size)
    else:
        quantstats = calcquantstats(tlm, error, bin_size=bin_size)

    if 'units' in limits:
        units = limits['units']
    elif 'unit' in limits:
        units = limits['unit']
    else:
        units = "C"

    startsec = DateTime(times[0]).secs
    stopsec = DateTime(times[-1]).secs

    xtick = np.linspace(startsec, stopsec, 10)
    xlab = [lab[:8] for lab in DateTime(xtick).date]

    if not fig:
        # fig = plt.figure(figsize=(16, 10), facecolor=[1, 1, 1])
        fig = plt.figure(figsize=(15, 8))
    else:
        fig.clf()

    if md5_string is not None:
        fig.text(0.01, 0.96, 'MD5: ' + md5_string, fontsize=14, color=[0.5, 0.5, 0.5], horizontalalignment='left')

    # ---------------------------------------------------------------------------------------------
    # Axis 1 - Model and Telemetry vs Time
    #
    # This plot is in the upper lefthand corner and shows predicted temperatures in red and
    # telemetry in blue vs time.
    #

    ax1 = fig.add_axes([0.1, 0.38, 0.44, 0.50], frameon=True)
    pred_line = ax1.plot(times, prediction, color='#d92121', linewidth=1, label='Model')
    telem_line = ax1.plot(times, tlm, color='#386cb0', linewidth=1.5, label='Telemetry')
    ax1.set_title('%s Temperature (%s)' % (modelname.replace('_', ' '), msid.upper()),
                  fontsize=18, y=1.00)
    ax1.set_ylabel('Temperature %s' % units, fontsize=18)
    if yplotlimits is not None:
        ax1.set_ylim(yplotlimits)
    ax1.tick_params(axis='y', labelsize=14)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels('')
    ax1.set_xlim(xtick[0] - 10, times[-1])
    ax1.grid(True)

    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()

    ymin1 = ylim1[0]
    ymax1 = ylim1[1]

    acis_limit_catch = []
    for name, value in limits.items():
        dy = ymax1-ymin1

        # skip over the unit entry and don't plot red limits
        if "unit" in name.lower() or name.startswith("odb.warning"):
            continue

        # if a limit is extreme compared to the available data or no value is entered, skip over it
        if value is None or value < ylim1[0] - 10 or value > ylim1[1] + 10:
            continue

        ax1.axhline(value, color=get_limit_color(name), linewidth=1.5)

        limit = get_limit_spec(name)

        qualifier = str(limit["qualifier"])
        if qualifier in ["acisi", "aciss", "aciss_hot"]:
            # These limits are grouped later for a combined label
            display_name = ""
            acis_limit_catch.append(value)
        elif "acis" in qualifier:
            instr = f"{qualifier[:4]}-{qualifier[4]}".upper()
            hot = " Hot" if qualifier.endswith("hot") else ""
            display_name = f"{instr}{hot} Limit"
        elif "cold_ecs" in qualifier:
            display_name = "Cold ECS Limit"
        elif limit["system"] == "planning":
            display_name = f"Planning {limit['direction'].capitalize()}"
        elif limit["type"] == "caution":
            display_name = f"Caution (Yellow) {limit['direction'].capitalize()}"
        else:
            display_name = name

        if len(display_name) > 0:
            plx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
            if "low" in name.lower():
                ply = value - 0.01 * (ylim1[1] - ylim1[0])
                txt = ax1.text(plx, ply, f'{display_name} = {value:4.1f} {units}',
                               ha="left", va="top", fontsize=12)
            else:
                ply = 0.01 * (ylim1[1] - ylim1[0]) + value
                txt = ax1.text(plx, ply, f'{display_name} = {value:4.1f} {units}',
                               ha="left", va="bottom", fontsize=12)
            txt.set_path_effects(
                [path_effects.Stroke(linewidth=4, foreground='white', alpha=1.0),
                 path_effects.Normal()]
            )
            txt.set_bbox(dict(color='white', alpha=0))

        if value < ymin1:
            ymin1 = value-0.1*dy
        if value > ymax1:
            ymax1 = value+0.1*dy

    # Plot the labels for the ACIS Science Limits, this avoids individual labels that clobber each
    # other.
    if len(acis_limit_catch) > 0:
        acis_lims_str = ", ".join([str(val) for val in acis_limit_catch])
        plx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
        ply = 0.01 * (ylim1[1] - ylim1[0]) + max(acis_limit_catch)
        txt = ax1.text(plx, ply, f'ACIS FP Science Limits ({acis_lims_str}, {units})',
                       ha="left", va="bottom", fontsize=12)
        txt.set_path_effects(
            [path_effects.Stroke(linewidth=4, foreground='white', alpha=1.0),
             path_effects.Normal()]
        )
        txt.set_bbox(dict(color='white', alpha=0))

    ax1.set_ylim(ymin1, ymax1)

    if legend_loc is not None:
        lns = pred_line + telem_line
        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, loc=legend_loc)
    # ---------------------------------------------------------------------------------------------
    # Axis 2 - Model Error vs Time
    #
    # This plot is in the lower lefthand corner and shows model error (telemetry - model) vs. time.
    #

    ax2 = fig.add_axes([0.1, 0.1, 0.44, 0.2], frameon=True)
    ax2.plot(times, error, color='#386cb0', label='Telemetry')
    if errorplotlimits:
        ax2.set_ylim(errorplotlimits)

    ax2.set_title('%s Model Error (Telemetry - Model)' % modelname.replace('_', ' '),
                  fontsize=18, y=1.00)
    ax2.set_ylabel('Error %s' % units, fontsize=18)
    ax2.tick_params(axis='y', labelsize=14)
    # ax2.tick_params(axis='x', which='major', pad=1)
    ax2.set_xticks(xtick)
    ax2.set_xticklabels(xlab, fontsize=14, rotation=30, ha='right')
    ax2.set_xlim(xtick[0] - 10, times[-1])
    ax2.grid(True)

    # ---------------------------------------------------------------------------------------------
    # Axis 3 - Telemetry vs Model Error
    #
    # This plot is in the upper righthand corner of the page and shows telemetry vs. model error.
    # The code to show model temperature vs model error is commented out but can be used in place
    # of the current comparison, although the quantile lines plotted will also need to be removed.
    #

    # Add some noise to the telemetry (or model) data to make it easier to see the data (less
    # pile-up).
    #
    # band is the mean difference between counts in telemetry (resolution).
    band = np.abs(np.diff(tlm))
    band = np.mean(band[band > 0]) / 2
    noise = np.random.uniform(-band, band, len(tlm))

    ax3 = fig.add_axes([0.62, 0.38, 0.36, 0.50], frameon=True)
    ax3.plot(error, tlm + noise, 'o', color='#386cb0', alpha=1, markersize=2,
             markeredgecolor='#386cb0')
    ax3.set_title('%s Telemetry \n vs. Model Error'
                  % modelname.replace('_', ' '), fontsize=18, y=1.00)
    ax3.set_ylabel('Temperature %s' % units, fontsize=18)
    ax3.grid(True)

    # This is an option so that the user can tweak the two righthand plots to use a reasonable
    # set of x axis limits, either because an outlier is expanding the axis range too much, or if
    # more space is needed at the boundaries of the axis.
    if errorplotlimits:
        ax3.set_xlim(errorplotlimits)

    ax3.set_ylim(ax1.get_ylim())
    ax3.set_yticks(ax1.get_yticks())
    ax3.tick_params(axis='y', labelsize=14)

    ax3.tick_params(axis='x', labelsize=14)

    for name, value in limits.items():
        # skip over the unit entry and don't plot red limits
        if "unit" in name.lower() or name.startswith("odb.warning"):
            continue

        # if a limit is extreme compared to the available data or no value is entered, skip over it
        if value is None or value < ylim1[0] - 10 or value > ylim1[1] + 10:
            continue
        ax3.axhline(value, color=get_limit_color(name), linewidth=1.5)

    # Plot quantile lines for each count value
    Epoints01, Tpoints01 = getQuantPlotPoints(quantstats, 'q01')
    Epoints99, Tpoints99 = getQuantPlotPoints(quantstats, 'q99')
    Epoints50, Tpoints50 = getQuantPlotPoints(quantstats, 'q50')
    ax3.plot(Epoints01, Tpoints01, color='k', linewidth=4)
    ax3.plot(Epoints99, Tpoints99, color='k', linewidth=4)
    ax3.plot(Epoints50, Tpoints50, color=[1, 1, 1], linewidth=4)
    ax3.plot(Epoints01, Tpoints01, 'k', linewidth=2)
    ax3.plot(Epoints99, Tpoints99, 'k', linewidth=2)
    ax3.plot(Epoints50, Tpoints50, 'k', linewidth=1.5)


    xlim3 = ax3.get_xlim()
    ylim1 = ax1.get_ylim() # Match axis 1 y scale


    # ---------------------------------------------------------------------------------------------
    # Axis 4 - Error Distribution Histogram
    #
    # This plot is in the lower righthand corner of the page and shows the error distribution
    # histogram.
    #

    ax4 = fig.add_axes([0.62, 0.1, 0.36, 0.2], frameon=True)
    n, bins, patches = ax4.hist(error, 40, range=errorplotlimits, facecolor='#386cb0')
    ax4.set_title('Error Distribution', fontsize=18, y=1.0)
    ytick4 = ax4.get_yticks()

    # This is an option so that the user can tweak the two righthand plots to use a reasonable
    # set of x axis limits, either because an outlier is expanding the axis range too much, or if
    # more space is needed at the boundaries of the axis.
    if errorplotlimits:
        ax4.set_xlim(errorplotlimits)
    else:
        xlim_offset = (np.max(error) - np.min(error))*0.1
        ax4.set_xlim(np.min(error) - xlim_offset, np.max(error) + xlim_offset)

    # Plot lines for statistical boundaries.
    ylim4 = ax4.get_ylim()

    ax4.set_yticks(ytick4)
    ax4.set_yticklabels(['%2.0f%%' % (100 * n / len(prediction)) for n in ytick4], fontsize=14)
    ax4.set_xlabel('Error %s' % units, fontsize=18)

    # Print lines and labels for statistical information.
    ystart = (ylim4[1] + ylim4[0]) * 0.5
    xoffset = -(.2 / 25) * np.abs(np.diff(ax4.get_xlim()))
    unit_char = 'F' if 'f' in units.lower() else 'C'

    ax4.axvline(stats['q50'], color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
    _ = ax4.text(stats['q50'] + xoffset * 1.1, ystart, f'50% {stats["q50"]:4.1f}{unit_char}', ha="right",
                 va="center", rotation=90, size=14)
    _.set_path_effects(
        [path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()]
    )

    if stats["q01"] > ax4.get_xlim()[0]:
        ax4.axvline(stats['q01'], color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
        _ = ax4.text(stats['q01'] + xoffset * 1.1, ystart, f'1% {stats["q01"]:4.1f}{unit_char}', ha="right",
                     va="center", rotation=90, size=14)
        _.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()])

    if stats["q16"] > ax4.get_xlim()[0]:
        ax4.axvline(stats['q16'], color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
        _ = ax4.text(stats['q16'] + xoffset * 1.1, ystart, f'16% {stats["q16"]:4.1f}{unit_char}', ha="right",
                     va="center", rotation=90, size=14)
        _.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()])

    if np.min(error) > (ax4.get_xlim()[0] + 1): # Avoid printing max data at plot boundary
        ax4.axvline(np.min(error), color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
        _ = ax4.text(np.min(error) + xoffset * 1.1, ystart, 'Minimum Error', ha="right",
                     va="center", rotation=90, size=14)
        _.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()])

    if stats['q99'] < ax4.get_xlim()[1]:
        ax4.axvline(stats['q99'], color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
        _ = ax4.text(stats['q99'] - xoffset * 0.9, ystart, f'99% {stats["q99"]:4.1f}{unit_char}', ha="left",
                     va="center", rotation=90, size=14)
        _.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()])

    if stats['q84'] < ax4.get_xlim()[1]:
        ax4.axvline(stats['q84'], color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
        _ = ax4.text(stats['q84'] - xoffset * 0.9, ystart, f'84% {stats["q84"]:4.1f}{unit_char}', ha="left",
                     va="center", rotation=90, size=14)
        _.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()])

    if np.max(error) < (ax4.get_xlim()[1] - 1): # Avoid printing max data at plot boundary
        ax4.axvline(np.max(error), color=[.5, .5, .5], linestyle='--', linewidth=1.5, alpha=1)
        _ = ax4.text(np.max(error) - xoffset * 0.9, ystart, 'Maximum Error', ha="left",
                     va="center", rotation=90, size=14)
        _.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=0.75), path_effects.Normal()])

    xlim4 = ax4.get_xlim()

    xlimright = [min(xlim3[0], xlim4[0]), max(xlim3[1], xlim4[1])]
    ax4.set_ylim(ylim4)
    ax4.set_xlim(xlimright)
    ax4.tick_params(axis='x', labelsize=14)

    # I know the following code looks to be redundant and unnecessary, but for some unholy reason,
    # Matplotlib REALLY does not want the top two axes to share the same Y scale. The following
    # code is used to pound Matplotlib into submission.
    ax3.set_ylim(ax1.get_ylim())
    ax3.set_ybound(ax1.get_ylim())
    ax3.set_yticks(ax1.get_yticks())
    ax3.set_yticklabels(ax1.get_yticks())

    ax1.set_ylim(ax1.get_ylim())
    ax1.set_ybound(ax1.get_ylim())
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels(ax1.get_yticks())

    # ---------------------------------------------------------------------------------------------
    # Force redraw and save.

    plt.draw()

    if savefig:
        fig.savefig(modelname + '_' + msid.upper() + '_Model_Dashboard.png')
