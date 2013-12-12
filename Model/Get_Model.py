from __future__ import division
from sys import argv
from pylab import *
from decimal import *
import os
import pandas
import numpy as np
import time
from itertools import cycle
import matplotlib.dates as mdates


### need a Date2Julain
def Julian2Date(series, year, month, day):
    '''
    converts JDAY to mpl date.time object

        series: column of JDAY
        year, month, day: reference year

    '''
    refdate = date2num(mpl.dates.datetime.date(year, month, day))
    series_date = num2date(series + refdate - 1)

    return series_date


def Get_spr(file_path, elev_cols=3, Date=[1996, 1, 1]):
    '''
    Specifiy the path of a spr.opt file.
    Returns pandas dataframe
    '''
    spr = pandas.read_table('{0}'.format(file_path), na_values = 'm', delim_whitespace = True)
    
    for r in range(1,elev_cols + 1, 1):
        spr = spr.drop('Elevation.{0}'.format(r), axis=1)
    
    if Date is not None:
        spr['Date'] = Julian2Date(spr['Julian_day'], Date[0], month = Date[1], day = Date[2])

    return spr

def Get_tsr(file_path, Date=[1996, 1, 1]):
    '''
    Specifiy the path of a tsr.opt file
    Returns pandas dataframe
    '''
    tsr = pandas.read_table('{0}'.format(file_path), delim_whitespace = True, skiprows = 11)

    if Date is not None:
        tsr['Date'] = Julian2Date(tsr['JDAY'], Date[0], Date[1], Date[2])

    return tsr

def Get_qwo(file_path, Date=[1996, 1, 1]):
    '''
    Specifiy the path of a qwo.opt file
    Returns pandas dataframe
    '''
    qwo = pandas.read_table('{0}'.format(file_path), delim_whitespace = True, skiprows = 2)

    if Date is not None:
        qwo['Date'] = Julian2Date(qwo['JDAY'], Date[0], Date[1], Date[2])

    return qwo

def Get_two(file_path, Date=[1996, 1, 1]):
    '''
    Specifiy the path of a two.opt file
    Returns pandas dataframe
    '''
    two = pandas.read_table('{0}'.format(file_path), delim_whitespace = True, skiprows = 2)

    if Date is not None:
        two['Date'] = Julian2Date(two['JDAY'], Date[0], Date[1], Date[2])

    return two        
