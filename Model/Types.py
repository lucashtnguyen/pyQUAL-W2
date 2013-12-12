from __future__ import division
from sys import argv
from pylab import *
from decimal import *
import os
import pandas
import numpy as np
import matplotlib.ticker as mticker
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from itertools import cycle
import matplotlib.dates as mdates



class Timeseries(object):
    def __init__(self,  dataframe, data, datecol='Date', 
        JDcol='JDAY', stype=None,  name=None, units=None):

        self._stype = stype
        self._name = name
        self._units = units

        self._raw_data = dataframe[data]
        self._date = dataframe[datecol]
        self._JDAY = dataframe[JDcol]

        self._min = None
        self._max = None
        self._mean = None
        self._median = None
        self._std = None

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
    def name(self):
        return self._name
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, value):
        self._units = value

    # Set stat properties
    @property
    def min(self):
        if self._min is None:
            self._min = self._raw_data.min()
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._max = self._raw_data.max()
        return self._max

    @property
    def mean(self):
        if self._mean is None:
            self._mean = np.mean(self._raw_data)
        return self._mean

    @property
    def median(self):
        if self._median is None:
            self._median = np.median(self._raw_data)
        return self._median

    @property
    def std(self):
        if self._std is None:
            self._std = np.std(self._raw_data)
        return self._std

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

        n = self.name
        u = self.units

        return paramunit.format(n, u)

    #needs to be generalized
    def plot(self, ax=None, JD=True, Date=True, usecomma=False, **mykwargs):
        
        if ax is None:
            fig, ax = plt.subplots()
        elif isinstance(ax, plt.Axes):
            fig = ax.figure
        else:
            raise ValueError("`ax` must be a matplotlib Axes instance or None")

        if JD is True:
            ax.plot(self._JDAY, self._raw_data, label = self._stype, 
                linestyle = self._plot_properties['line'], 
                color = self._plot_properties['color'], 
                marker = self._plot_properties['marker'],
                markersize = self._plot_properties['marker_size'],
                **mykwargs)
            ax.xaxis.set_major_locator( MaxNLocator(nbins = 9))
            ax.set_xlabel('Julain Day')

        elif JD and Date is True:
            ax2 = ax.twiny()
            ax2.plot(self._date, self._raw_data, linestyle='')
            ax2.set_label('Date', fontsize = 9)
            ax2.xaxis.set_major_locator( MaxNLocator(nbins = 9))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        elif JD is not True and Date is True:
            ax.plot(self._date, self._raw_data, label = self._stype, 
                linestyle = self._plot_properties['line'], 
                color = self._plot_properties['color'], 
                marker = self._plot_properties['marker'],
                markersize = self._plot_properties['marker_size'],
                **mykwargs)
            ax.xaxis.set_major_locator( MaxNLocator(nbins = 9))
            ax.set_xlabel('Date')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        ax.set_ylabel(self.paramunit(usecomma))
        matplotlib.rc('xtick', labelsize=8) 
        matplotlib.rc('ytick', labelsize=8)
        plt.gca().xaxis.set_major_locator( MaxNLocator(nbins = 7))

        plt.gca().yaxis.set_major_locator( MaxNLocator(nbins = 18))
        ax.xaxis.set_major_locator( MaxNLocator(nbins = 9))

        ax.legend(loc = 'best',  fontsize = 9)

        return fig

class Profile(object):

class Compare(object):
    def __init__(self, ts1, ts2):

        self._mse = None
        self._rmse = None

#these need an index object to be the same, either here or in the Timeseries fnx
        @property
        def mse(self):
            if self._mse is None:
                self._mse = ((self.ts1._raw_data - self.ts2._raw_data)** 2).mean()
            return self._mse

        @property
        def rmse(self):
            if self._rmse is None:
                self._rmse = np.sqrt(((self.ts1._raw_data - self.ts2._raw_data)** 2).mean())
            return self._rmse