
import numpy as np
import matplotlib
import matplotlib.patheffects as path_effects

from Chandra.Time import DateTime

if 'k' not in matplotlib.rcParams['text.color']:
    matplotlib.rcParams['axes.facecolor'] = [.1,.15,.2]

matplotlib.rcParams['xtick.major.pad'] = 5
matplotlib.rcParams['ytick.major.pad'] = 5



plt = matplotlib.pyplot


def getQuantPlotPoints(quantstats, quantile):
    """ Calculate the error quantile line coordinates for the data in each telemetry count value.

    :param quantstats: output from calcquantstats()
    :param quantile: quantile (string - e.g. "q01"), used as a key into quantstats datastructure
    :returns: coordinates for error quantile line

    This is used to calculate the quantile lines plotted on the telemetry vs. error plot (axis 3)
    enclosing the data (i.e. the 1 and 99 percentile lines).

    """

    Tset = [T for (n, T) in quantstats['key']]
    diffTset = np.diff(Tset)
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


def calcquantstats(Ttelem, error):
    """ Calculate error quantiles for individual telemetry temperatures (each count individually).

    :param Ttelem: telemetry values
    :param error: model error (telemetry - model)
    :returns: coordinates for error quantile line

    This is used for the telemetry vs. error plot (axis 3).

    """

    Tset = np.sort(list(set(Ttelem)))
    Tquant = {'key': []}
    k = -1
    for T in Tset:
        if len(Ttelem[Ttelem == T]) >= 200:
            k = k + 1
            Tquant['key'].append([k, T])
            ind = Ttelem == T
            errvals = error[ind]
            Tquant[k] = calcquantiles(errvals)

    return Tquant


def digitize_data(Ttelem, nbins=50):
    """ Digitize telemetry.

    :param Ttelem: telemetry values
    :param nbins: number of bins
    :returns: coordinates for error quantile line

    """

    # Bin boundaries
    # Note that the min/max range is expanded to keep all telemetry within the outer boundaries.
    # Also the number of boundaries is 1 more than the number of bins.
    bins = np.linspace(min(Ttelem) - 1e-6, max(Ttelem) + 1e-6, nbins + 1)
    inds = np.digitize(Ttelem, bins) - 1
    means = bins[:-1] + np.diff(bins) / 2

    return np.array([means[i] for i in inds])


def dashboard(prediction, tlm, times, limits, modelname='PSMC', msid='1pdeaat',
              errorplotlimits=None, yplotlimits=None, fig=None, savefig=True, legend_loc='best'):
    """ Plot Xija model dashboard.

    :param prediction: model prediction
    :param tlm: telemetry 
    :param times: model/telemetry time values
    :param limits: model limit dict, (e.g. {"units":"C", "caution_high":-12.0,
                                            "planning_limit":-14.0})
    :param modelname: Name of model (e.g. "ACA")
    :param msid: msid name (e.g. "aacccdpt")
    :param errorplotlimits: list or tuple of min and max x axis plot boundaries for both righthand
           plots
    :param yplotlimits: Y axis plot limits to be used where appropriate, if not None
    :param fig:  Figure object to use, if None, a new figure object is generated
    :param savefig: Boolean indicated whether or not to save the plot to an image file
    :param legend_loc: value to be passed to the 'loc' keyword in the  matplotlib pyplot legend
           method, if None, then no legend is displayed

    Note: prediction, tlm, and times must all have the same number of values.

    """

    # Set some plot characteristic default values
    # matplotlib.rcParams['xtick.major.pad'] = 10
    # matplotlib.rcParams['ytick.major.pad'] = 5
    matplotlib.rc('font', family='sans-serif')
    matplotlib.rc('font', weight='light')

    error = tlm - prediction
    stats = calcquantiles(error)

    # In this case the data is not discretized to a limited number of count values, or has too
    # many possible values to work with calcquantstats(), such as with tlm_fep1_mong.
    if len(np.sort(list(set(tlm)))) > 1000:
        quantized_tlm = digitize_data(tlm)
        quantstats = calcquantstats(quantized_tlm, error)
    else:
        quantstats = calcquantstats(tlm, error)

    units = limits['units']
    cautionhigh = limits.get('caution_high', None)
    planninglimit = limits.get('planning_limit', None)
    acisi_limit = limits.get('acisi_limit', None)
    aciss_limit = limits.get('aciss_limit', None)
    fp_sens_limit = limits.get('fp_sens_limit', None)

    startsec = DateTime(times[0]).secs
    stopsec = DateTime(times[-1]).secs

    xtick = np.linspace(startsec, stopsec, 10)
    xlab = [lab[:8] for lab in DateTime(xtick).date]

    if not fig:
        # fig = plt.figure(figsize=(16, 10), facecolor=[1, 1, 1])
        fig = plt.figure(figsize=(15, 8))
    else:
        fig.clf()

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
    ax1.set_ylabel('Temperature deg%s' % units, fontsize=18)
    if yplotlimits is not None:
        ax1.set_ylim(yplotlimits)
    ax1.set_yticklabels(ax1.get_yticks(), fontsize=14)
    ax1.set_xticks(xtick)
    ax1.set_xticklabels('')
    ax1.set_xlim(xtick[0] - 10, times[-1])
    ax1.grid(True)

    if cautionhigh:

        # Draw caution high limit line.
        ylim1 = ax1.get_ylim()
        dy = ylim1[1] - ylim1[0]
        if ylim1[1]-2 <= cautionhigh:
            ax1.set_ylim(ylim1[0], cautionhigh + dy/7.)
            ax1.set_yticklabels(ax1.get_yticks(), fontsize=14)
        ylim1 = ax1.get_ylim()
        yellowlimitline = ax1.plot(ax1.get_xlim(), [cautionhigh, cautionhigh], 'orange',
                                   linewidth=1.5)
        if ylim1[1] <= cautionhigh:
            ax1.set_ylim(ylim1[0], cautionhigh + 1)

        # Print caution high limit value (remember plot is 50% of fig height).
        #chfig = 0.50 * (cautionhigh - ylim1[0]) / (np.diff(ylim1)) + 0.38
        #txt = fig.text(0.11, chfig - 0.000, 'Caution High (Yellow) = {:4.1f} {}'.format(
        #    cautionhigh, units), ha="left", va="bottom", size=18)
        #txt.set_bbox(dict(color='white', alpha=0.8))
        xlim1 = ax1.get_xlim()
        chx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
        chy = 0.01 * (ylim1[1] - ylim1[0]) + cautionhigh
        txt = ax1.text(chx, chy, 'Caution High (Yellow) = {:4.1f} {}'.format(cautionhigh, units),
                ha="left", va="bottom", fontsize=12)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white', alpha=0.7),
                       path_effects.Normal()])

        txt.set_bbox(dict(color='white', alpha=0))

    if planninglimit:

        # Draw planning limit line.
        planninglimitline1 = ax1.plot(ax1.get_xlim(), [planninglimit, planninglimit], 'g--',
                                      linewidth=1.5)
        ylim1 = ax1.get_ylim()

        # Print planning limit value (remember plot is 50% of fig height).
        #if (ylim1[-1] - planninglimit) / np.diff(ylim1) < 0.1:
        #    plfig = 0.50 * (planninglimit - ylim1[0]) / (np.diff(ylim1)) + 0.38
        #else:
        #    # <-- figure coordinates (plot is 50% of fig height)
        #    plfig = 0.50 * (planninglimit - ylim1[0]) / (np.diff(ylim1)) + 0.38 + 0.01
        #
        #if cautionhigh:
        #    if np.abs(planninglimit - cautionhigh)/np.diff(ylim1) < 0.1:
        #        plfig = plfig + 0.02

        #txt = fig.text(0.11, plfig - 0.005, 'Planning Limit = {:4.1f} {}'.format(
        #       planninglimit, units), ha="left", va="top", size=18)
        #txt.set_bbox(dict(color='white', alpha=0.8))

        xlim1 = ax1.get_xlim()
        plx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
        ply = 0.01 * (ylim1[1] - ylim1[0]) + planninglimit
        txt = ax1.text(plx, ply, 'Planning Limit = {:4.1f} {}'.format(planninglimit, units),
                       ha="left", va="bottom", fontsize=12)
        txt.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white', alpha=0.7),
                              path_effects.Normal()])
        # txt.set_bbox(dict(color='white', alpha=0))

    if acisi_limit:

        # Draw ACIS-I limit line.
        acisilimitline = ax1.plot(ax1.get_xlim(), [acisi_limit, acisi_limit], 'b-.',
                                  linewidth=1.5)

        ylim1 = ax1.get_ylim()

        xlim1 = ax1.get_xlim()
        plx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
        ply = 0.01 * (ylim1[1] - ylim1[0]) + acisi_limit
        txt = ax1.text(plx, ply, 'ACIS-I Limit = {:4.1f} {}'.format(acisi_limit, units),
                       ha="left", va="bottom", fontsize=14)
        txt.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=1.0),
                              path_effects.Normal()])

    if aciss_limit:

        # Draw ACIS-S limit line.
        acisslimitline = ax1.plot(ax1.get_xlim(), [aciss_limit, aciss_limit], '-.',
                                  color='purple', linewidth=1.5)

        ylim1 = ax1.get_ylim()

        xlim1 = ax1.get_xlim()
        plx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
        ply = 0.01 * (ylim1[1] - ylim1[0]) + aciss_limit
        txt = ax1.text(plx, ply, 'ACIS-S Limit = {:4.1f} {}'.format(aciss_limit, units),
                       ha="left", va="bottom", fontsize=14)
        txt.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=1.0),
                              path_effects.Normal()])

    if fp_sens_limit:

        # Draw FP SENS limit line.
        fpsenslimitline = ax1.plot(ax1.get_xlim(), [fp_sens_limit, fp_sens_limit], '--',
                                   color='red', linewidth=1.5)

        ylim1 = ax1.get_ylim()

        xlim1 = ax1.get_xlim()
        plx = 0.02 * (xlim1[1] - xlim1[0]) + xlim1[0]
        ply = 0.01 * (ylim1[1] - ylim1[0]) + fp_sens_limit
        txt = ax1.text(plx, ply, 'FP SENS Limit = {:4.1f} {}'.format(fp_sens_limit, units),
                       ha="left", va="bottom", fontsize=14)
        txt.set_path_effects([path_effects.Stroke(linewidth=4, foreground='white', alpha=1.0),
                              path_effects.Normal()])

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
    ax2.set_ylabel('Error deg%s' % units, fontsize=18)
    ax2.set_yticklabels(ax2.get_yticks(), fontsize=14)
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
    ax3.plot(error, tlm + noise, 'o', color='#386cb0', alpha=1, markersize=2, markeredgecolor='#386cb0')
    #ax3.plot(error, prediction + noise, 'b,', alpha = 0.1)
    ax3.set_title('%s Telemetry \n vs. Model Error'
                  % modelname.replace('_', ' '), fontsize=18, y=1.00)
    ax3.set_ylabel('Temperature deg%s' % units, fontsize=18)
    ax3.grid(True)

    # This is an option so that the user can tweak the two righthand plots to use a reasonable
    # set of x axis limits, either because an outlier is expanding the axis range too much, or if
    # more space is needed at the boundaries of the axis.
    if errorplotlimits:
        ax3.set_xlim(errorplotlimits)

    ax3.set_ylim(ax1.get_ylim())
    ax3.set_yticks(ax1.get_yticks())
    ax3.set_yticklabels(ax1.get_yticks(), fontsize=14)
    ylim3 = ax3.get_ylim()

    ax3.set_xticklabels(ax3.get_xticks(), fontsize=14)

    xlim3 = ax3.get_xlim()

    if cautionhigh:

        # Draw caution high limit line.
        dt = 0.05 * np.diff(ax1.get_ylim())
        yellowlimitline3 = ax3.plot(xlim3, [cautionhigh, cautionhigh], 'orange', linewidth=1.5)
        if ylim3[1] <= cautionhigh:
            ax3.set_ylim(ylim3[0], cautionhigh + 1)
            ax3.set_yticklabels(ax3.get_yticks(), fontsize=18)

    if planninglimit:
        # Draw planning limit line.
        planninglimitline3 = ax3.plot(xlim3, [planninglimit, planninglimit], 'g--', linewidth=1.5)

    if acisi_limit:

        # Draw ACIS-I limit line.
        acisilimitline = ax3.plot(xlim3, [acisi_limit, acisi_limit], 'b-.',
                                  linewidth=1.5)

    if aciss_limit:

        # Draw ACIS-S limit line.
        acisslimitline = ax3.plot(xlim3, [aciss_limit, aciss_limit], '-.',
                                  color='purple', linewidth=1.5)

    if fp_sens_limit:

        # Draw FP SENS limit line.
        fpsenslimitline = ax3.plot(xlim3, [fp_sens_limit, fp_sens_limit], '--',
                                   color='red', linewidth=1.5)

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
    ax4.set_yticklabels(['%2.0f%%' % (100 * n / len(prediction)) for n in ytick4], fontsize=14)
    ax4.plot([stats['q01'], stats['q01'] + 1e-6], ylim4, color=[.5, .5, .5], linestyle='--',
             linewidth=1.5, alpha=1)
    ax4.plot([stats['q99'], stats['q99'] + 1e-6], ylim4, color=[.5, .5, .5], linestyle='--',
             linewidth=1.5, alpha=1)
    ax4.plot([np.min(error), np.min(error) + 1e-6], ylim4, color=[.5, .5, .5], linestyle='--',
             linewidth=1.5, alpha=1)
    ax4.plot([np.max(error), np.max(error) + 1e-6], ylim4, color=[.5, .5, .5], linestyle='--',
             linewidth=1.5, alpha=1)
    ax4.set_xlabel('Error deg%s' % units, fontsize=18)

    # Print labels for statistical boundaries.
    ystart = (ylim4[1] + ylim4[0]) * 0.5
    xoffset = -(.2 / 25) * np.abs(np.diff(ax4.get_xlim()))
    ptext4a = ax4.text(stats['q01'] + xoffset * 1.1, ystart, '1% Quantile', ha="right",
                       va="center", rotation=90, size=14)
    
    if np.min(error) > ax4.get_xlim()[0]:
        ptext4b = ax4.text(np.min(error) + xoffset * 1.1, ystart, 'Minimum Error', ha="right",
                           va="center", rotation=90, size=14)
    ptext4c = ax4.text(stats['q99'] - xoffset * 0.9, ystart, '99% Quantile', ha="left",
                       va="center", rotation=90, size=14)
		       
    if np.max(error) < ax4.get_xlim()[1]:
        ptext4d = ax4.text(np.max(error) - xoffset * 0.9, ystart, 'Maximum Error', ha="left",
                           va="center", rotation=90, size=14)

    xlim4 = ax4.get_xlim()

    xlimright = [min(xlim3[0], xlim4[0]), max(xlim3[1], xlim4[1])]
    ax4.set_ylim(ylim4)
    ax4.set_xlim(xlimright)
    ax4.set_xticklabels(ax4.get_xticks(), fontsize=14)

    # Replot axis 3 using widest right hand side xlim
    if cautionhigh:
    #     # Draw caution high limit line.
        yellowlimitline3 = ax3.plot(xlimright, [cautionhigh, cautionhigh], 'orange', linewidth=1.5)

    # Draw planning limit line.
    planninglimitline3 = ax3.plot(xlimright, [planninglimit, planninglimit], 'g--', linewidth=1.5)
    ax3.set_xlim(xlimright)
    ax3.set_xticklabels(ax3.get_xticks())

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
