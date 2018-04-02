import datetime as dt
import os
import numpy as np
import wradlib as wrl
import io_func_PH
import zipfile
import shutil
import re

def _asdtformat(dtime, dtformat="%Y-%m-%d %H:%M:%S"):
    """Formats a datetime object <dtiem> to <dtformat>
    """
    try:
        return dtime.strftime(dtformat)
    except:
        if type(dtime) in (str, np.string_):
            return dtime
        else:
            raise Exception("Cannot format datetime %r to %r." % (dtime, dtformat))

def asdtformat(dtime, dtformat="%Y-%m-%d %H:%M:%S"):
    """Formats a datetime object <dtime> to <dtformat>
    """
    if type(dtime) in (list, tuple, np.ndarray):
        return np.array([_asdtformat(elem, dtformat) for elem in dtime])
    else:
        return _asdtformat(dtime, dtformat)

def make_twindow(tstart, tend, ttol=0.):
    """Defines a time window based on tstart, tend, and a tolerance.
    """
    if not type(tstart)==dt.datetime:
        tstart = dt.datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    if not type(tend)==dt.datetime:
        tend   = dt.datetime.strptime(tend  , "%Y-%m-%d %H:%M:%S")
    ttol = dt.timedelta(seconds=ttol)
    tstart = tstart - ttol
    tend = tend + ttol
    return tstart, tend

def avg_time(datetimes):
    total = sum(dt_.hour * 3600 + dt_.minute * 60 + dt_.second for dt_ in datetimes)
    avg = total / len(datetimes)
    minutes, seconds = divmod(int(avg), 60)
    hours, minutes = divmod(minutes, 60)
    return dt.datetime.combine(dt.date(int(datetimes.year.mean()),
                                       int(datetimes.month.mean()),
                                       int(datetimes.day.mean())),
                                       dt.time(hours, minutes, seconds))

def get_radarfile_attime(data_path, dtime, radarname, sweepid, variable, ttol):

    tstart, tend = asdtformat(dtime), asdtformat(dtime)
    tstart, tend = make_twindow(tstart, tend, ttol)

    files, datetimes = zipfiles_in_twindow(data_path, tstart, tend, radarname=radarname, sweepid=sweepid,variable=variable)
    return files, datetimes

def days_in_twindow(tstart, tend):
    """Determines the days in a time window defined by tstart and tend.
    """
    if not type(tstart)==dt.datetime:
        tstart = dt.datetime.strptime(tstart, "%Y-%m-%d %H:%M:%S")
    if not type(tend)==dt.datetime:
        tend   = dt.datetime.strptime(tend  , "%Y-%m-%d %H:%M:%S")
    startday = dt.datetime.strptime(tstart.strftime("%Y-%m-%d"), "%Y-%m-%d")
    endday   = dt.datetime.strptime(tend.strftime("%Y-%m-%d"), "%Y-%m-%d")
    if endday > startday:
        days = wrl.util.from_to(startday, endday, 86400)
    else:
        days=[startday,]
    return days

def zipfiles_in_twindow(datadir, tstart, tend, radarname, variable, sweepid=None):
    """Find and extract the radar data files in zip archive corresponding to a time window
    """

    if sweepid == None:
        fileendswith = r'\d\d-%s\.nc$'%(variable)
    else:
        fileendswith = "-%s-%s.nc" % (sweepid, variable)

    # containers for results
    files = []
    datetimes = []
    days = days_in_twindow(tstart, tend)
    for day in days:
        daydir = os.path.join(datadir, day.strftime("%Y\\%m\\%d"))
        #print daydir
        srczip = os.path.join(daydir, day.strftime("%Y%m%d_"+variable+".zip") )
        if zipfile.is_zipfile(srczip):
            # source file exists and is a zip file, now list content
            zf = zipfile.ZipFile(srczip, 'r')
        else:
            continue
        filesinzip = zf.namelist()
        if len(filesinzip) == 0:
            raise ValueError("Warning! There are no files for this day.")
            #raise
        # list all the netcdf data files in directory datadir
        for f in filesinzip:
            #if f.endswith(fileendswith):
            matches = re.search(fileendswith, f)
            if matches:
                filename = os.path.basename(f)
                if sweepid == None:
                    dtime = dt.datetime.strptime(filename, radarname+"-%Y%m%d-%H%M%S-"+matches.group(0))
                else:
                    dtime = dt.datetime.strptime(filename, radarname+"-%Y%m%d-%H%M%S"+fileendswith)
                if (dtime>=tstart) and (dtime<=tend):
                    ### extract file and remember the file in list <files>
                    ##files.append( zf.extract(f, path=daydir) )
                    # Our new way to make sure we do not extract the internal subdirectory structure
                    target = file(os.path.join(daydir, filename), "wb")
                    source = zf.open(f)
                    with source, target:
                        shutil.copyfileobj(source, target)
                    files.append(os.path.join(daydir, filename))
                    datetimes.append(dtime)
        zf.close()
    return files, np.array(datetimes)

def get_matching_gr_filenames(gpm_file, gr_data_path, radarname, sweepid, variable, ttol, ee):

    # read GPM data
    pr_data = io_func_PH.read_gpm(gpm_file)
    # Get mean time of overpass
    avetime = pr_data['date'][0] + (pr_data['date'][-1] - pr_data['date'][0])/2

    # identify radar data filenames with same times
    datafiles, datetimes = get_radarfile_attime(gr_data_path, dtime=avetime, radarname=radarname,
                                                sweepid=sweepid, variable=variable, ttol=ttol)

    # get the filename of the reference sweep closest to overpass time (ideally lowest sweep)
    fileendswith = "-%s-%s.nc" % (str(ee+1).zfill(2), variable)
    sweep_idx = []
    sweep_fnames = []
    for i in xrange(len(datafiles)):
        if datafiles[i].endswith(fileendswith):
            sweep_idx.append(i)
            sweep_fnames.append(datafiles[i])
    time_diff = []
    for dtime in datetimes[sweep_idx]:
        time_diff.append(abs(avetime - dtime))
    closest_ = time_diff.index(min(time_diff))

    # determine index of filename of sweep closest to overpass time
    closest_idx = datafiles.index(sweep_fnames[closest_])

    # get list of filenames that belong to the same volume scan
    #    this is a bit hardcoded, it assumes that there are 16 scans in a volume
    vol_filenames = datafiles[closest_idx-ee:closest_idx+(16-ee)]

    # remove extracted files that are not in the needed volume
    delete_list = [f for f in datafiles if f not in vol_filenames]

    # delete the files after reading
    for fname in delete_list:
        try:
            os.remove(fname)
        except:
            print("Cannot remove %s" % fname)

    return vol_filenames

def get_matching_gr_filenames_secondclosest(gpm_file, gr_data_path, radarname, sweepid, variable, ttol, ee):

    # read GPM data
    pr_data = io_func_PH.read_gpm(gpm_file)
    # Get mean time of overpass
    avetime = pr_data['date'][0] + (pr_data['date'][-1] - pr_data['date'][0])/2

    # identify radar data filenames with same times
    datafiles, datetimes = get_radarfile_attime(gr_data_path, dtime=avetime, radarname=radarname,
                                                sweepid=sweepid, variable=variable, ttol=ttol)

    # get the filename of the reference sweep closest to overpass time (ideally lowest sweep)
    fileendswith = "-%s-%s.nc" % (str(ee+1).zfill(2), variable)
    sweep_idx = []
    sweep_fnames = []
    for i in xrange(len(datafiles)):
        if datafiles[i].endswith(fileendswith):
            sweep_idx.append(i)
            sweep_fnames.append(datafiles[i])

    # if there are no volumes before or after the avetime within the ttol
    # (meaning radar stopped archiving for given time period)
    if len(sweep_idx) <= 1:
        raise ValueError

    time_diff = []
    for dtime in datetimes[sweep_idx]:
        time_diff.append(abs(avetime - dtime))
    closest_ = time_diff.index(min(time_diff))
    if avetime < datetimes[sweep_idx][closest_]:
        closest_2nd = closest_ - 1
    elif avetime >= datetimes[sweep_idx][closest_]:
        closest_2nd = closest_ + 1

    # if closest 2nd file does not exist
    # (meaning radar stopped archiving for given time period)
    if closest_2nd >= len(sweep_idx):
        raise ValueError

    print(sweep_fnames, closest_2nd)
    # determine index of filename of sweep closest to overpass time
    closest_2nd_idx = datafiles.index(sweep_fnames[closest_2nd])

    # get list of filenames that belong to the same volume scan
    #    this is a bit hardcoded, it assumes that there are 16 scans in a volume
    vol_filenames = datafiles[closest_2nd_idx-ee:closest_2nd_idx+(16-ee)]

    # remove extracted files that are not in the needed volume
    delete_list = [f for f in datafiles if f not in vol_filenames]

    # delete the files after reading
    for fname in delete_list:
        try:
            os.remove(fname)
        except:
            print("Cannot remove %s" % fname)

    return vol_filenames

def get_matching_gr_filenames_TRMM(trmm_file1, trmm_file2, gr_data_path, radarname, sweepid, variable, ttol, ee):

    # read TRMM data
    pr_data = io_func_PH.read_trmm_gdal(trmm_file1, trmm_file2)
    # # Get mean time of overpass
    avetime = pr_data['date'][0] + (pr_data['date'][-1] - pr_data['date'][0])/2

    # # identify radar data filenames with same times
    datafiles, datetimes = get_radarfile_attime(gr_data_path, dtime=avetime, radarname=radarname,
                                                sweepid=sweepid, variable=variable, ttol=ttol)

    # get the filename of the reference sweep closest to overpass time (ideally lowest sweep)
    fileendswith = "-%s-%s.nc" % (str(ee+1).zfill(2), variable)
    sweep_idx = []
    sweep_fnames = []
    for i in xrange(len(datafiles)):
        if datafiles[i].endswith(fileendswith):
            sweep_idx.append(i)
            sweep_fnames.append(datafiles[i])
    time_diff = []
    for dtime in datetimes[sweep_idx]:
        time_diff.append(abs(avetime - dtime))
    closest_ = time_diff.index(min(time_diff))

    # determine index of filename of sweep closest to overpass time
    closest_idx = datafiles.index(sweep_fnames[closest_])

    # get list of filenames that belong to the same volume scan
    #    this is a bit hardcoded, it assumes that there are 16 scans in a volume
    vol_filenames = datafiles[closest_idx-ee:closest_idx+(16-ee)]

    # remove extracted files that are not in the needed volume
    delete_list = [f for f in datafiles if f not in vol_filenames]

    # delete the files after reading
    for fname in delete_list:
        try:
            os.remove(fname)
        except:
            print("Cannot remove %s" % fname)

    return vol_filenames


def get_matching_gr_filenames_2015b(gpm_file, gr_data_path, radarname, sweepid, variable, ttol, ee):

    # read GPM data
    pr_data = io_func_PH.read_gpm(gpm_file)
    # Get mean time of overpass
    avetime = pr_data['date'][0] + (pr_data['date'][-1] - pr_data['date'][0])/2

    # identify radar data filenames with same times
    datafiles, datetimes = get_radarfile_attime(gr_data_path, dtime=avetime, radarname=radarname,
                                                sweepid=sweepid, variable=variable, ttol=ttol)

    # get the filename of the reference sweep closest to overpass time (ideally lowest sweep)
    fileendswith = "-%s-%s.nc" % (str(ee+1).zfill(2), variable)
    sweep_idx = []
    sweep_fnames = []
    for i in xrange(len(datafiles)):
        if datafiles[i].endswith(fileendswith):
            sweep_idx.append(i)
            sweep_fnames.append(datafiles[i])
    time_diff = []
    for dtime in datetimes[sweep_idx]:
        time_diff.append(abs(avetime - dtime))
    closest_ = time_diff.index(min(time_diff))

    # determine index of filename of sweep closest to overpass time
    closest_idx = datafiles.index(sweep_fnames[closest_])

    # get list of filenames that belong to the same volume scan
    #    this is a bit hardcoded, it assumes that there are 16 scans in a volume
    vol_filenames = datafiles[closest_idx-ee:closest_idx+(5-ee)]

    # remove extracted files that are not in the needed volume
    delete_list = [f for f in datafiles if f not in vol_filenames]

    # delete the files after reading
    for fname in delete_list:
        try:
            os.remove(fname)
        except:
            print("Cannot remove %s" % fname)

    return vol_filenames
