from __future__ import division

# from pylab import *
from decimal import *
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from itertools import cycle
import os
import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn
from matplotlib.mlab import griddata
import matplotlib.gridspec as gridspec

#need to set up a git hub remote

#need to set up a git hub remote

class Timeseries(object):
    def __init__(self,  dataframe, data, datecol='Date',
        JDcol='JDAY', stype=None,  constituent=None, units=None):

        ## To do:
        # golden ratio for plots


        # Initialize general attributes
        self._stype = stype
        self._constituent = constituent
        self._units = units

        # Initialize data attributes
        self._raw_data = dataframe
        self._data = dataframe[data]
        if datecol is not None:
            self._date = dataframe[datecol]
        else:
            self._date = None
        if JDcol is not None:
            self._JDAY = dataframe[JDcol]
        else:
            self._JDAY = None

        # Initialize stat attributes
        self._N = None
        self._min = None
        self._max = None
        self._mean = None
        self._median = None
        self._std = None
        self._stats = None

        # Initialize plot attributes
        self._plot_properties = None

    ## Set Properties
    # General properties
    @property
    def stype(self):
        return self._stype
    @stype.setter
    def stype(self, value):
        self._stype = value

    @property
    def constituent(self):
        return self._constituent
    @constituent.setter
    def constituent(self, value):
        self._constituent = value

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = value

    # Set stat properties
    @property
    def N(self):
        return self._raw_data.shape[0]

    @property
    def min(self):
        if self._min is None:
            self._min = self._data.min()
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._max = self._data.max()
        return self._max

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.mean(self._data)
        return self._mean

    @property
    def median(self):
        if self._median is None:
            self._median = np.median(self._data)
        return self._median

    @property
    def std(self):
        if self._std is None:
            self._std = np.std(self._data)
        return self._std

    @property
    def stats(self):
        if self._stats is None:
            stats_str =  'N =\t{0.2f}\n Min =\t{1.2f}\n Max =\t{2.2f}\n Mean =\t{3.2f}\n Median =\t{4.2f}\n stdev. =\t{5.2f}\n'
            self._stats = stats_str.format(
                self.N,
                self.min,
                self.max,
                self.mean,
                self.median,
                self.std)
        return self._stats

    # Set plot properties
    @property
    def  plot_properties(self):
        if self._plot_properties is None:
            if self._stype == 'Model':
                self._plot_properties = {
                    'marker': '',
                    'marker_size': 2,
                    'line' : '-',
                    'color' : 'r',
                    }
            elif self._stype == 'Data':
                self._plot_properties = {
                    'marker': 'o',
                    'marker_size': 2,
                    'line' : '',
                    'color' : 'k',
                    }
            else:
                self._plot_properties = {
                    'marker': '',
                    'marker_size': 2,
                    'line' : '--',
                    'color' : 'b',
                    }
        return self._plot_properties


    # Create units for plotting how I like them
    def paramunit(self, usecomma=False):
        '''
        Creates a string representation of the parameter and units

        Input:
            usecomma : optional boot (default is False)
                Toggles the format of the `paramunit` attribute...
                If True:
                    self.paramunit = <parameter>, <unit>
                If False:
                    self.paramunit = <parameter> (<unit>)
        '''
        if usecomma:
            paramunit = '{0}, {1}'
        else:
            paramunit = '{0} ({1})'

        n = self.constituent
        u = self.units

        return paramunit.format(n, u)

    # Timeseries plot
    def plot(self, ax=None, usecomma=False, **mykwargs):
        '''
        Adds a time series plot to a matplotlib figure. xaxis formatted based
        on input data having JDcol, datecol, or both.

        Input:
            ax : Optional matplotlib axes object or None (default).
                Axes on which the boxplot will be drawn. If None, one will
                be created.

            usecomma : optional bool (default=False). Formats the yaxis label.
                If True:
                    self.paramunit = '<parameter>, <unit>'
                If False:
                    self.paramunit = '<parameter> (<unit>)'

            **mykwargs : matplotlib kwargs

        '''
        seaborn.set(style='nogrid',context='paper')

        if self._plot_properties is None:
            self.plot_properties

        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, plt.Axes):
            fig = ax.figure
        else:
            raise ValueError("`ax` must be a matplotlib Axes instance or None")

        if self._JDAY is not None:
            ax.plot(self._JDAY, self._data, label = self._stype,
                linestyle = self._plot_properties['line'],
                color = self._plot_properties['color'],
                marker = self._plot_properties['marker'],
                markersize = self._plot_properties['marker_size'],
                alpha = 0.60,
                **mykwargs)
            # ax.set_xticklabels(self._JDAY, rotation=30  )
            # ax.xaxis.set_major_locator( MaxNLocator(nbins = 9))
            # plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 9))
            ax.set_xlabel('Julian Day')
            plt.setp(plt.xticks()[1], rotation=30)

        elif self._JDAY is None and self._date is not None:
            ax.plot(self._date, self._data, label = self._stype,
                linestyle = self._plot_properties['line'],
                color = self._plot_properties['color'],
                marker = self._plot_properties['marker'],
                markersize = self._plot_properties['marker_size'],
                alpha = 0.60,
                **mykwargs)
            # ax.set_xticklabels(self._date, rotation=30  )
            # ax.xaxis.set_major_locator( MaxNLocator(nbins = 9))
            # plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 9))
            ax.set_xlabel('Date')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(plt.xticks()[1], rotation=30)

        #!!! know bug with ylim changes
        if self._JDAY is not None and self._date is not None:
            ax2 = ax.twiny()
            ax2.plot(self._date, self._data, linestyle='', alpha = 0.60)
            ax2.set_xlabel('Date')
            # ax2.set_xticklabels(self._date, rotation=30  )
            # ax2.xaxis.set_major_locator( MaxNLocator(nbins = 9))
            # plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 9))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(plt.xticks()[1], rotation=30)

        ax.set_ylabel(self.paramunit(usecomma))
        matplotlib.rc('xtick', labelsize=8)
        matplotlib.rc('ytick', labelsize=8)
        # plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 9)) # look at these
        # plt.gca().yaxis.set_major_locator( MaxNLocator(nbins = 18))
        ax.xaxis.set_major_locator( MaxNLocator(nbins = 9)) # look at these

        ax.legend(loc = 'best',  fontsize = 9)


        # figure(figsize=(7, 7/1.6), dpi=300)
        fig.tight_layout()

        if self._JDAY is not None and self._date is not None:
            return fig, ax
        else:
            return fig, ax

class Stats_TS(object):
    def __init__(self, ts1, ts2):

        self.ts1 = ts1
        self.ts2 = ts2

        self._StatData = None

        self._me = None
        self._ame = None
        self._mse = None
        self._rmse = None
        self._stats = None

    @property
    def StatData(self):
        # need to define an N function for stats
        if self._StatData is None:
            df1 = pandas.DataFrame(
                {
                'JDAY':self.ts1._JDAY,
                'DATE':self.ts1._date,
                '{0}_1'.format(self.ts1.stype):self.ts1._data
                })
            df2 = pandas.DataFrame(
                {
                'JDAY':self.ts2._JDAY,
                'DATE':self.ts2._date,
                '{0}_2'.format(self.ts2.stype):self.ts2._data
                })

            # need to research how np.NaN is treated in df subtraction.
            # Might be more appropriate to use and inner join here
            self._StatData = pandas.merge(df1, df2, how='outer')
            # can still play around with the idea of an index here
        return self._StatData


    @property
    def me(self):
        if self._me is None:
            self._me = (self.StatData['{0}_1'.format(self.ts1.stype)] - self.StatData['{0}_2'.format(self.ts2.stype)]).mean()
        return self._me

    @property
    def ame(self):
        if self._ame is None:
            self._ame = (np.absolute(self.StatData['{0}_1'.format(self.ts1.stype)] - self.StatData['{0}_2'.format(self.ts2.stype)])).mean()
        return self._ame

    @property
    def mse(self):
        if self._mse is None:
            self._mse = ((self.StatData['{0}_1'.format(self.ts1.stype)] - self.StatData['{0}_2'.format(self.ts2.stype)])** 2).mean()
        return self._mse

    @property
    def rmse(self):
        if self._rmse is None:
            self._rmse = np.sqrt(((self.StatData['{0}_1'.format(self.ts1.stype)] - self.StatData['{0}_2'.format(self.ts2.stype)])** 2).mean())
        return self._rmse
    @property
    def stats(self):
        if self._stats is None:
            stats_str = "Error Stats ({0}):\nME = {1:.2f}\nAME = {2:.2f}\nMSE = {3:.2f}\nRMSE = {4:.2f}"
            self._stats = stats_str.format(self.ts2.stype, self.me, self.ame, self.mse, self.rmse)
        return self._stats

    def plot(self, stats=False, usecomma=False, **mykwargs):
        #currently plot inherits ts2 constituent and units for yaxis
        fig, ax = self.ts1.plot(usecomma=False, **mykwargs)
        self.ts2.plot(ax=ax, usecomma=False, **mykwargs)

        if stats is True:
            ax.text(0.05, 0.2,s =self.stats, horizontalalignment='left',
            verticalalignment='center',
            transform=ax.transAxes)

        return fig, ax

class Profile(object):
    def __init__(self,  dataframe, data, datecol='Date',
        JDcol='Julian_day', elevationcol='Elevation', stype=None,  constituent=None, units=None):

        # To do:
        # golden ratio

        # Initialize general attributes
        self._stype = stype
        self._constituent = constituent
        self._units = units

        # Initialize data attributes
        self._raw_data = dataframe
        self._data = dataframe[data]
        self._elev = dataframe[elevationcol]
        if datecol is not None:
            self._date = dataframe[datecol]
        else:
            self._date = None
        if JDcol is not None:
            self._JDAY = dataframe[JDcol]
        else:
            self._JDAY = None

        # Initialize stat attributes
        self._N = None
        self._min = None
        self._max = None
        self._mean = None
        self._median = None
        self._std = None
        self._stats = None

        # Initialize plot attributes
        self._plot_properties = None

    # Set Properties
    # General properties
    @property
    def stype(self):
        return self._stype
    @stype.setter
    def stype(self, value):
        self._stype = value

    @property
    def constituent(self):
        return self._constituent
    @constituent.setter
    def constituent(self, value):
        self._constituent = value

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = value

    # Set stat properties
    @property
    def N(self):
        return self._raw_data.shape[0]

    @property
    def min(self):
        if self._min is None:
            self._min = self._data.min()
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._max = self._data.max()
        return self._max

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.mean(self._data)
        return self._mean

    @property
    def median(self):
        if self._median is None:
            self._median = np.median(self._data)
        return self._median

    @property
    def std(self):
        if self._std is None:
            self._std = np.std(self._data)
        return self._std

    @property
    def stats(self):
        if self._stats is None:
            stats_str =  'N =\t{0:.2f}\n Min =\t{1:.2f}\n Max =\t{2:.2f}\n Mean =\t{3:.2f}\n Median =\t{4:.2f}\n stdev. =\t{5:.2f}\n'
            self._stats = stats_str.format(
                self.N,
                self.min,
                self.max,
                self.mean,
                self.median,
                self.std)
        return self._stats

    # Set plot properties
    @property
    def  plot_properties(self):
        if self._plot_properties is None:
            if self._stype == 'Model':
                self._plot_properties = {
                    'marker': '',
                    'marker_size': 2,
                    'line' : '-',
                    'color' : 'r',
                    }
            elif self._stype == 'Data':
                self._plot_properties = {
                    'marker': 'o',
                    'marker_size': 4,
                    'line' : '',
                    'color' : 'k',
                    }
            else:
                self._plot_properties = {
                    'marker': '',
                    'marker_size': 2,
                    'line' : '--',
                    'color' : 'b',
                    }
        return self._plot_properties


    # Create units for plotting how I like them
    def paramunit(self, usecomma=False):
        '''
        Creates a string representation of the parameter and units

        Input:
            usecomma : optional boot (default is False)
                Toggles the format of the `paramunit` attribute...
                If True:
                    self.paramunit = <parameter>, <unit>
                If False:
                    self.paramunit = <parameter> (<unit>)
        '''
        if usecomma:
            paramunit = '{0}, {1}'
        else:
            paramunit = '{0} ({1})'

        n = self.constituent
        u = self.units

        return paramunit.format(n, u)

    # profile plot
    def profile_plot(self, pltday, ax=None, usecomma=False, **mykwargs):
        '''
        Adds a time series plot to a matplotlib figure. xaxis formatted based
        on input data having JDcol, datecol, or both.

        Input:
            ax : Optional matplotlib axes object or None (default).
                Axes on which the boxplot will be drawn. If None, one will
                be created.

            usecomma : optional bool (default=False). Formats the yaxis label.
                If True:
                    self.paramunit = '<parameter>, <unit>'
                If False:
                    self.paramunit = '<parameter> (<unit>)'

            **mykwargs : matplotlib kwargs

        '''
        # seaborn.set_axes_style(style='darkgrid', context='notebook')
        if self._plot_properties is None:
            self.plot_properties

        if ax is None:
            fig, ax = plt.subplots(figsize=((7/1.6), 7), dpi=300)
        elif isinstance(ax, plt.Axes):
            fig = ax.figure
        else:
            raise ValueError("`ax` must be a matplotlib Axes instance or None")

        plot_data = pandas.DataFrame({
            'data' : self._data,
            'elev' : self._elev,
            'JDAY' : self._JDAY
        })

        if self._date is not None:
            plot_data['date'] = self._date
        plot_data = plot_data.set_index('JDAY').sort(columns='elev')

        ax.plot(plot_data.xs(pltday)['data'], plot_data.xs(pltday)['elev'],
            label = self._stype,
            linestyle = self._plot_properties['line'],
            color = self._plot_properties['color'],
            marker = self._plot_properties['marker'],
            markersize = self._plot_properties['marker_size'],
            alpha = 0.60,
            **mykwargs)
        ax.set_ylabel('Elevation (m)')
        ax.set_xlabel(self.paramunit(usecomma))
        # ax.invert_yaxis()

        if self._date is not None and self._JDAY is not None:
            ax.set_title('JDAY {0}\n Date {1}'.format(pltday, plot_data.xs(pltday)['date'].irow(0).date()))
        elif self._JDAY is not None:
            ax.set_title('JDAY{0}'.format(pltday))
        elif self._date is not None:
            ax.set_title('Date {0}'.format(plot_data.xs(pltday)['date'].irow(0).date()))

        matplotlib.rc('xtick', labelsize=12)
        matplotlib.rc('ytick', labelsize=12)
        plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 5)) # look at these
        plt.gca().yaxis.set_major_locator( MaxNLocator(nbins = 7))
        ax.xaxis.set_major_locator( MaxNLocator(nbins = 5)) # look at these
        ax.set_xlim([0, 25])
        # place holder until proxy legend
        ax.legend(loc = 'best',  fontsize = 12)

        # figure(figsize=(7/1.6, 7), dpi=300)
        fig.tight_layout()


        return fig, ax

    def profile_subplots(self, jdays, figshape=(4, 2), startcols=(0,),
                       externaldata=None, fig=None, all_axes=None):

        def formatGSAxes(ax, axtype, col, xticks, ylabel, sublabel=None, labelsize=8):
            # common stuff
            ax.xaxis.tick_bottom()  # this effectively removes ticks from the top

            if sublabel is None:
                sublabel = ''

            # outer axes
            if axtype == 'outer':
                ax.set_ylabel(ylabel, size=labelsize, verticalalignment='bottom')

                # left side of the figure
                if col == 0:
                    #ax.spines['right'].set_color('none')
                    # remove ticks from the right
                    ax.yaxis.tick_left()
                    ax.tick_params(axis='y', right='off', which='both')
                    ax.annotate(sublabel, (0.0, 1.0), xytext=(5, -10),
                                xycoords='axes fraction', textcoords='offset points',
                                fontsize=8, zorder=20)

                # right side of the figure
                if col == 1:
                    #ax.spines['left'].set_color('none')
                    ax.yaxis.tick_left()
                    ax.tick_params(axis='y',left='off', right='off', which='both')

                    # remove ticks from the left
                    ax.yaxis.tick_right()
                    ax.tick_params(axis='y',right='off', left='off', which='both')
                    # ax.yaxis.set_label_position("right")
                    # label = ax.yaxis.get_label()
                    # label.set_rotation(270)
                    # ax.annotate(sublabel, (0.0, 1.0), xytext=(5, -10),
                    #             xycoords='axes fraction', textcoords='offset points',
                    #             fontsize=8, zorder=20)
                if col == 2:
                    #ax.spines['left'].set_color('none')
                    ax.yaxis.tick_right()
                    ax.tick_params(axis='y', left='off', which='both')
                    # remove ticks from the left

                    # ax.yaxis.set_label_position("right")
                    # label = ax.yaxis.get_label()
                    # label.set_rotation(270)
                    ax.annotate(sublabel, (1.0, 1.0), xytext=(-20, -10),
                                xycoords='axes fraction', textcoords='offset points',
                                fontsize=8, zorder=20)

            # inner axes
            elif axtype == 'inner':
                ax.set_yticklabels([])
                ax.tick_params(axis='y', right='off', left='off', which='both')

                # left size
                if col == 0:
                    ax.spines['left'].set_color('none')

                # right side
                if col == 1:
                    ax.spines['left'].set_color('none')
                    ax.spines['right'].set_color('none')

                if col == 2:
                    ax.spines['right'].set_color('none')

            # middle? coming soon?
            else:
                raise ValueError('axtype can only be "inner" or "outer"')

            # clear tick labels if xticks is False or Nones
            if not xticks:
                ax.set_xticklabels([])
                ax.set_xlabel('')


        if fig is None:
            fig = plt.figure(figsize=(6, 10),
                            facecolor='none',
                            edgecolor='none')
        if fig_gs is None:
            fig_gs = gridspec.GridSpec(3, 3)

        letters = list('abcdefghijklmnop')

        if all_axes is None:
            all_axes = []
        for n, wqcomp in enumerate(jdays):
            ax_gs = gridspec.GridSpecFromSubplotSpec(
                1, 4, subplot_spec=fig_gs[n], wspace=0.00
        )

            sublabel = '(%s)' % letters[n]
            col = n % 3

            axes = []
            for colnum in range(len(startcols)):
                if colnum < len(startcols) - 1:
                    cspan_start = startcols[colnum]
                    cspan_stop = startcols[colnum+1]
                    axes.append(fig.add_subplot(ax_gs[cspan_start:cspan_stop]))
                else:
                    axes.append(fig.add_subplot(ax_gs[startcols[colnum]:]))

            self.profile_plot(pltday=wqcomp, ax=axes[0])

            all_axes.append(axes)

            #if row < self.nrows - 1:
            if n >= len(jdays) - 3:
                xticks = True
            else:
                xticks = False

            if len(axes) == 1:
                formatGSAxes(axes[0], 'outer', col, xticks,
                    'Elevation (m)',
                    sublabel=sublabel, labelsize=7
                )

            elif len(axes) == 2:
                formatGSAxes(axes[col], 'inner', col, xticks,
                    'Elevation (m)',
                    sublabel=sublabel, labelsize=7
                )

                formatGSAxes(axes[col-1], 'outer', col, xticks,
                    'Elevation (m)',
                    labelsize=7
                )

            else:
                raise NotImplementedError('no more than 2 axes per gridspec')

        # self._proxy_legend(all_axes[1][-1], externaldata=externaldata)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.3, top=0.96, wspace=0.05)

        # figname = utils.processFilename('%s-megafig-%s-%s-%02d' %
        #                                 (self.siteid, figname, self.sampletype,
        #                                  self.fignum))
        # figpath = os.path.join('cvc', 'output', 'img', figname)
        # fig.savefig(figpath + '.pdf', dpi=300, transparent=True, bbox_inches='tight')
        # fig.savefig(figpath + '.png', dpi=300, transparent=False, bbox_inches='tight')
        return fig, all_axes,

    def depth_date_plot(self, grid_N=100, interpolation=None):
        # http://stackoverflow.com/questions/14120222/matplotlib-imshow-with-irregular-spaced-data-points

        seaborn.set(style='ticks', context='paper')

        zs0 = self._data.values
        ys0 = self._elev.values
        xs0 = self._JDAY.values

        # how do you change to a complex type?
        N = complex(0, grid_N)

        extent = (self._JDAY.min(),
            self._JDAY.max(),
            self._elev.max(),
            self._elev.min())

        xs,ys = np.mgrid[extent[0]:extent[1]:N, extent[3]:extent[2]:N]
        resampled = griddata(xs0, ys0, zs0, xs, ys)

        fig, ax = plt.subplots(figsize=(6, (5/1.6)), dpi=300)
        im = ax.imshow(resampled.T,
            extent=extent,
            interpolation=interpolation,
            cmap='coolwarm',
            aspect="auto")

        ax.invert_yaxis()

        ax.set_xlabel('Julian Day')
        ax.set_ylabel('Elevation (m)')

        ax.grid( 'off' )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label(self.paramunit())
        fig.tight_layout()
        return ax
