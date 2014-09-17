import matplotlib as mpl
import matplotlib.dates as mdates

def julian2date(jd, base=(1947, 1, 1)):
    '''
    converts JDAY to mpl date.time object
    ---------
    Requires:
        jd: float, Julian day
        base: tuple, (year, month, day) i.e., the reference year

    '''
    refdate = mdates.date2num(mpl.dates.datetime.date(*base))
    jd_date = mdates.num2date(jd + refdate - 1)

    return jd_date

def date2julain(d, base=(1947, 1, 1)):
    bbase = mdates.date2num(datetime.datetime(*base)) - 1
    jd = mdates.date2num(d) - bbase
    return jd
