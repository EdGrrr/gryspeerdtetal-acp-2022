'''Do analysis and create plots for

Observing short-timescale cloud development to constrain aerosol-cloud interactions

Gryspeerdt et al., ACP, 2022'''

from csat2 import MODIS, ECMWF
from csat import CCCM
import matplotlib.pyplot as plt
import csat2.misc
import csat2.misc.plotting
import csat2.misc.stats
import numpy as np
import cartopy.crs as ccrs
import scipy.interpolate
import time
from netCDF4 import Dataset
import traceback
from tqdm import tqdm

import logging
logging.basicConfig(level=logging.DEBUG)


def nan_gf(image, sigma, truncate=4):
    '''Gaussian filter for 2D field with nans'''
    # From https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    V = image.copy()
    V[np.isnan(image)] = 0
    VV = scipy.ndimage.gaussian_filter(V, sigma=sigma, truncate=truncate)

    W = 0*image.copy()+1
    W[np.isnan(image)] = 0
    WW = scipy.ndimage.gaussian_filter(W, sigma=sigma, truncate=truncate)
    return VV/WW


def get_hist(opdata, scslice, ndbins, lwpbins):
    '''Produces a 2D LWP-Nd histogram for scslice region'''
    mask = np.isfinite(opdata['LWP'][scslice] +
                       opdata['aLWP'][scslice] +
                       opdata['Nd'][scslice] +
                       opdata['aNd'][scslice])
    num = np.histogram2d(
        opdata['LWP'][scslice][mask],
        opdata['Nd'][scslice][mask],
        bins=[lwpbins, ndbins])
    return num[0]


def get_flowfield(opdata, scslice, ndbins, lwpbins, lts_lim=0, sig=None):
    '''Calculates the Nd and LWP flowfields in Nd-LWP space for the scslice region

    Applies a 2D gaussian filter for smoothing'''
    mask = np.isfinite(opdata['LWP'][scslice] +
                       opdata['aLWP'][scslice] +
                       opdata['Nd'][scslice] +
                       opdata['aNd'][scslice])
    num = np.histogram2d(
        opdata['LWP'][scslice][mask],
        opdata['Nd'][scslice][mask],
        bins=[lwpbins, ndbins])[0]
    dNd = np.histogram2d(
        opdata['LWP'][scslice][mask],
        opdata['Nd'][scslice][mask],
        bins=[lwpbins, ndbins],
        weights=(opdata['aNd']-opdata['Nd'])[scslice][mask])[0]
    dLWP = np.histogram2d(
        opdata['LWP'][scslice][mask],
        opdata['Nd'][scslice][mask],
        bins=[lwpbins, ndbins],
        weights=(opdata['aLWP']-opdata['LWP'])[scslice][mask])[0]
    if sig:
        dNd_sq = np.histogram2d(
            opdata['LWP'][scslice][mask],
            opdata['Nd'][scslice][mask],
            bins=[lwpbins, ndbins],
            weights=((opdata['aNd']-opdata['Nd'])[scslice][mask])**2)[0]
        dLWP_sq = np.histogram2d(
            opdata['LWP'][scslice][mask],
            opdata['Nd'][scslice][mask],
            bins=[lwpbins, ndbins],
            weights=((opdata['aLWP']-opdata['LWP'])[scslice][mask])**2)[0]
        dNd_var = (dNd_sq/num) - (dNd/num)**2
        dNd_tval = (dNd/num)/(np.sqrt(dNd_var/num))

        scipy.stats.ttest_ind_from_stats(
            dNd/num, np.sqrt(dNd_var), num,
            0, 0, 1e10)

    dnd_ndlwp = np.where(num > 1, dNd/num, np.nan)
    dlwp_ndlwp = np.where(num > 1, dLWP/num, np.nan)

    return nan_gf(dlwp_ndlwp, sigma=1), nan_gf(dnd_ndlwp, sigma=1)


def get_backflowfield(opdata, scslice, ndbins, lwpbins, lts_lim=0):
    '''As get_flowfield, but calcualtes a back-flowfield - binning by the final Nd-LWP state'''
    mask = np.isfinite(opdata['LWP'][scslice] +
                       opdata['aLWP'][scslice] +
                       opdata['Nd'][scslice] +
                       opdata['aNd'][scslice])
    num = np.histogram2d(
        opdata['aLWP'][scslice][mask],
        opdata['aNd'][scslice][mask],
        bins=[lwpbins, ndbins])
    dNd = np.histogram2d(
        opdata['aLWP'][scslice][mask],
        opdata['aNd'][scslice][mask],
        bins=[lwpbins, ndbins],
        weights=(opdata['Nd']-opdata['aNd'])[scslice][mask])
    dLWP = np.histogram2d(
        opdata['aLWP'][scslice][mask],
        opdata['aNd'][scslice][mask],
        bins=[lwpbins, ndbins],
        weights=(opdata['LWP']-opdata['aLWP'])[scslice][mask])

    dnd_ndlwp = np.where(num[0] > 1, dNd[0]/num[0], np.nan)
    dlwp_ndlwp = np.where(num[0] > 1, dLWP[0]/num[0], np.nan)

    return nan_gf(dlwp_ndlwp, sigma=1), nan_gf(dnd_ndlwp, sigma=1)


def get_meansumfield(opdata, varname, scslice, ndbins, lwpbins, factor=1, min_no=1):
    '''Return the average field for 'varname' in Nd-LWP space
    reduces resolution by 'factor'''
    mask = np.isfinite(opdata['LWP'][scslice] +
                       opdata['aLWP'][scslice] +
                       opdata['Nd'][scslice] +
                       opdata['aNd'][scslice]+opdata[varname][scslice])
    varnum = np.histogram2d(
        opdata['LWP'][scslice][mask],
        opdata['Nd'][scslice][mask],
        bins=[lwpbins[::factor], ndbins[::factor]])
    varsum = np.histogram2d(
        opdata['LWP'][scslice][mask],
        opdata['Nd'][scslice][mask],
        bins=[lwpbins[::factor], ndbins[::factor]],
        weights=opdata[varname][scslice][mask])

    return np.where(varnum[0] > min_no, varsum[0]/varnum[0], np.nan), varnum[0]


def integrate_flowfield(ff_dlwp, ff_dnd, ndbins, lwpbins, tsteps=30, randscale=0, scale=1):
    # Consider these arrays as flow field
    # starting conditions
    s_nd, s_lwp = np.meshgrid(csat2.misc.stats.lin_av(
        ndbins), csat2.misc.stats.lin_av(lwpbins))

    # First deltas
    lwpind = np.clip(np.digitize(s_lwp, lwpbins)-1, 0, ff_dlwp.shape[0]-1)
    ndind = np.clip(np.digitize(s_nd, ndbins)-1, 0, ff_dlwp.shape[1]-1)
    d_nd = ff_dnd[lwpind, ndind]/scale
    d_lwp = ff_dlwp[lwpind, ndind]/scale

    # Remove points with no data (can ravel at this point too)
    smask = np.isfinite(d_nd+d_lwp)

    # Want a initial zero sensitivity, so match cases above and below the mean
    # and remove those with no match
    # Maximise the intersection?
    a = np.argmax([np.sum(smask[i:]*(smask[::-1][:-i]))
                  for i in range(1, smask.shape[0])])
    smask[(a+1):] *= smask[::-1][:-(a+1)]
    s_nd = s_nd[smask]
    s_lwp = s_lwp[smask]

    # Output arrays
    o_nd = np.zeros((tsteps+1, len(s_nd)))
    o_nd[0] = s_nd
    o_lwp = np.zeros((tsteps+1, len(s_lwp)))
    o_lwp[0] = s_lwp

    for i in range(tsteps):
        s_nd = o_nd[i]
        s_lwp = o_lwp[i]
        # Step forward
        lwpind = np.clip(np.digitize(s_lwp, lwpbins)-1, 0, ff_dlwp.shape[0]-1)
        ndind = np.clip(np.digitize(s_nd, ndbins)-1, 0, ff_dlwp.shape[1]-1)
        d_nd = ff_dnd[lwpind, ndind]/scale + \
            np.random.normal(scale=randscale, size=len(lwpind))
        d_lwp = ff_dlwp[lwpind, ndind]/scale + \
            np.random.normal(scale=randscale, size=len(lwpind))
        o_nd[i+1] = s_nd+d_nd
        o_lwp[i+1] = s_lwp+d_lwp

    # Return the historical locations of the points
    return o_nd, o_lwp


def read_subset(year, doy, sat, sds):
    try:
        terradata = MODIS.readin('subset', year, doy, sat=sat,
                                 sds=sds, col='61')
    except:
        terradata = MODIS.readin('subset', year, doy, sat=sat,
                                 sds=sds, col='6')
    return terradata


def read_cdnc(year, doy, sat, sds, versions):
    for version in versions:
        try:
            cdnc = MODIS.readin('cdnc_best', year, doy,
                                sds=sds,
                                version=version, sat=sat)
            return cdnc
        except:
            pass
    raise ValueError('No valid version')


def read_ecmwf(year, doy):
    # Get windspeed at Terra overpass
    uwind = ECMWF.readin('ERA5', year, doy, 'U-wind-component', level='1000hPa',
                         resolution='1grid', time='LST')[[1]].resample(time='1D').first()
    vwind = ECMWF.readin('ERA5', year, doy, 'V-wind-component', level='1000hPa',
                         resolution='1grid', time='LST')[[1]].resample(time='1D').first()

    ws = np.sqrt(uwind**2+vwind**2)
    cycle = list(range(180, 360))+list(range(180))
    ws = ws[:, cycle][:, :, ::-1]

    wlon = ws['lon'].values
    wlon[wlon > 180] -= 360
    ws.coords['lon'] = wlon

    return ws


def read_reff(year, doy):
    try:
        data = MODIS.readin('MOD08_D3', year, doy, [
                            'Cloud_Effective_Radius_Liquid_Histogram_Counts'], col='61')
    except:
        data = MODIS.readin('MOD08_D3', year, doy, [
                            'Cloud_Effective_Radius_Liquid_Histogram_Counts'], col='6')
    small_drops = data['Cloud_Effective_Radius_Liquid_Histogram_Counts'][:, :10].sum(
        axis=1).transpose(..., 'XDim', 'YDim')
    all_drops = data['Cloud_Effective_Radius_Liquid_Histogram_Counts'].sum(
        axis=1).transpose(..., 'XDim', 'YDim')
    return small_drops, all_drops


#########
# Setup #
#########

years = range(2007, 2021)
doys = range(1, 367)

save_directory = '/disk1/Users/erg10'
cloudsat_directory = '/home/erg10/csprecip'

ndbins = np.exp(np.arange(1.65, 6.8, 0.03))
lwpbins = np.exp(np.arange(0.1, 7.4, 0.03))
lonrange = [-30, 10]
latrange = [-40, -10]

# Readin from base files, or use saved data
readin = False

###################
# Readin the data #
###################

if readin:
    tasks = []
    for year in years:
        for doy in doys:
            tasks.append([year, doy])

    starttime = time.time()
    opdata = {}

    for task in tqdm(tasks):
        year, doy = task
        try:
            indata = {}

            terradata = read_subset(year, doy, sat='terra', sds=['CF_Ice', 'LWP_Liq']).rename(
                {'LWP_Liq': 'LWP'})
            cdnc = read_cdnc(year, doy, sat='terra', sds=['Nd_G18'],
                             versions=['1']).rename(
                                 {'Nd_G18': 'Nd'})
            indata['CF_Ice'] = terradata['CF_Ice']
            indata['Nd'] = cdnc['Nd']
            indata['LWP'] = terradata['LWP']
            indata['ws'] = read_ecmwf(year, doy)
            indata['small_drops'], indata['all_drops'] = read_reff(year, doy)

            for var in ['Nd', 'LWP', 'ws', 'small_drops', 'all_drops']:
                indata[var].values = np.where(
                    terradata['CF_Ice'] <= 0.1,
                    indata[var], np.nan)

            aquadata = read_subset(year, doy, sat='aqua', sds=['CF_Ice', 'LWP_Liq']).rename(
                {'LWP_Liq': 'aLWP'})
            cdnc = read_cdnc(year, doy, sat='aqua', sds=['Nd_G18'],
                             versions=['1']).rename(
                                 {'Nd_G18': 'aNd'})
            aquadata['aNd'] = cdnc['aNd']
            for var in ['aNd', 'aLWP']:
                aquadata[var].values = np.where(
                    aquadata['CF_Ice'] <= 0.1,
                    aquadata[var], np.nan)

            ###########################
            # Sort the advection step #
            ###########################
            adv_vars = ['aNd', 'aLWP']

            # The lon and lat positions for each gridbox
            exlon = (terradata['lon'].values[:, None] +
                     np.array([-0.375, -0.125, 0.125, 0.375])[None, :]).ravel()
            exlat = (terradata['lat'].values[:, None] +
                     np.array([-0.375, -0.125, 0.125, 0.375])[None, :]).ravel()

            start_time = csat2.misc.time.lst_to_utc(
                10.5, np.outer(exlon, np.ones(len(exlat))))*60

            end_time = csat2.misc.time.lst_to_utc(
                13.5, np.outer(exlon, np.ones(len(exlat))))*60

            # Account for dateline effects - shift to C6 style day of year
            start_time[start_time > start_time[0, 0]] -= 1440
            end_time[end_time < end_time[-1, -1]] += 1440

            current_time = start_time
            # TODO: Remove cases where start and end don't both have data
            current_time[np.isnan(end_time)] = np.nan

            trace_time = np.zeros(current_time.shape)

            # Work out the advected end locations - these should always exist, even if there is no data
            # Use the ERA5 winddata
            wind = ECMWF.ERA5WindData(res='1grid', level='1000hPa')

            # These arrays store the position of each trace. They start in the middle of each cell so that
            # converting them to an int gives the correct index
            itracelon, itracelat = np.meshgrid(
                np.arange(0.5, 1440., 1), np.arange(0.5, 720., 1), indexing='ij')
            # Ignore cases where the end time is before the start time (around
            # the dateline) - This shouldn't include any pixels as the array has been updated.
            itracelat[np.logical_not(end_time > start_time)] = np.nan
            itracelon[np.logical_not(end_time > start_time)] = np.nan

            for hour in range(30):
                if np.any(current_time < end_time):
                    # Advect these pixels
                    adv_mask = np.where((current_time <= (hour * 60)) &
                                        (current_time < end_time))
                    if len(adv_mask[0]) == 0:
                        continue
                    # How long to advect for (max one hour) in seconds
                    adv_time = np.where(
                        (end_time-current_time) > 60, 60, end_time-current_time)[adv_mask]
                    # For locations where the Terra data is from the previous day, just use the winds from midnight
                    u, v = wind.get_data_time(
                        csat2.misc.time.ydh_to_datetime(year, doy, hour))

                    # Locations of current trace
                    tlat = itracelat[adv_mask].astype('int')//4
                    tlon = itracelon[adv_mask].astype('int')//4

                    # Convert to km in time interval (Adv_time is minutes, u ms-1)
                    adv_du = 60*adv_time*u.values[tlon, tlat]/1000
                    adv_dv = 60*adv_time*v.values[tlon, tlat]/1000

                    # These are in units of 'gridboxes', hence the *4 as we are working at 0.25deg
                    dlat = -1 * adv_dv/111.111 * 4  # lat index increases southwards - MODIS grid
                    dlon = adv_du / \
                        (111.111*np.sin(np.deg2rad(itracelat[adv_mask]/4))) * 4

                    itracelat[adv_mask] += dlat
                    # Clip to +/1 0.1 degrees to avoid nans in longitude
                    itracelat = np.clip(itracelat, 0.1, itracelat.shape[1]-1.1)
                    itracelon[adv_mask] += dlon
                    itracelon = itracelon % (itracelon.shape[0])

                    current_time[adv_mask] += adv_time
                    trace_time[adv_mask] += adv_time

            # Convert itrace values to gridbox indicies
            itracelon = (itracelon//4).astype('int')
            itracelat = (itracelat//4).astype('int')

            for name in adv_vars:
                indata[name+'_noadv'] = aquadata[name]
                indata[name] = aquadata[name].copy()
                griddata = indata[name +
                                  '_noadv'].values[0][itracelon, itracelat]
                indata[name].values[0] = csat2.misc.stats.reduce_res(
                    griddata, 4)

            csat2.misc.l_dictkeysappend(opdata, indata)

        except:
            print(f'Failed {year} {doy}')

    csat2.misc.dlist.l_toarray(opdata, axis='time')
    print(time.time()-starttime)

    opdata['year'] = opdata['LWP']['time.year']
    opdata['doy'] = opdata['LWP']['time.dayofyear']
    print('Saving')
    csat2.misc.fileops.nc4_dump_mv(
        f'{save_directory}/lwp_nd_data_store.nc', opdata)

opdata = {}
with Dataset(f'{save_directory}/lwp_nd_data_store.nc') as ncdf:
    for name in ncdf.variables.keys():
        opdata[name] = ncdf.variables[name][:]

################################################
# Read the gridded cloudsat preciptiation data #
################################################

def get_precip_hist(ndbins, lwpbins, lonrange, latrange):
    bins = {}
    bins['LWP'] = lwpbins
    bins['CDNC'] = ndbins

    number = np.zeros((5,
                       len(bins['LWP']),
                       len(bins['CDNC'])))

    for year in range(2006, 2021):
        print(year, end=' ')
        for doy in doys:
            if doy % 10 == 0:
                print(doy, end=' ')
            try:
                input_filename = f'{cloudsat_directory}/{year}/csprecip.{year}.{doy:0>3}.nc'
                with Dataset(input_filename) as ncdf:
                    input_data = ncdf['Precip_flag'][:]
                total_cld = input_data.sum(axis=0)[None, :, ::-1]
                warmprecip_cld = input_data[1:4].sum(axis=0)[None, :, ::-1]
                coldprecip_cld = input_data[4:].sum(axis=0)[None, :, ::-1]

                aquadata = read_subset(year, doy, sat='aqua', sds=['CF_Ice', 'LWP_Liq']).rename(
                    {'LWP_Liq': 'aLWP'})
                cdnc = read_cdnc(year, doy, sat='aqua', sds=['Nd_best'],
                                 versions=['7']).rename(
                                     {'Nd_best': 'aNd'})
                aquadata['aNd'] = cdnc['aNd']
                for var in ['aNd', 'aLWP']:
                    aquadata[var].values = np.where(
                        aquadata['CF_Ice'] <= 0.1,
                        aquadata[var], np.nan)

                loninds = np.digitize(lonrange, aquadata['lon'])
                loninds.sort()
                latinds = np.digitize(latrange, aquadata['lat'])
                latinds.sort()

                dataslice = np.s_[:,
                                  loninds[0]:loninds[1],
                                  latinds[0]:latinds[1]]

                ndvals = aquadata['aNd'][dataslice].values.ravel()
                lwpvals = aquadata['aLWP'][dataslice].values.ravel()
                validmask = np.isfinite(ndvals+lwpvals)
                ndinds = np.clip(np.digitize(
                    ndvals[validmask], ndbins)-1, 0, len(ndbins)-1)
                lwpinds = np.clip(np.digitize(
                    lwpvals[validmask], lwpbins)-1, 0, len(lwpbins)-1)

                np.add.at(number[0], (lwpinds, ndinds),
                          total_cld[dataslice].ravel()[validmask])
                np.add.at(number[1], (lwpinds, ndinds),
                          warmprecip_cld[dataslice].ravel()[validmask])
                np.add.at(number[2], (lwpinds, ndinds),
                          coldprecip_cld[dataslice].ravel()[validmask])
                newmask = (total_cld[dataslice].ravel()[validmask] > 0)
                np.add.at(number[3], (lwpinds[newmask], ndinds[newmask]),
                          (warmprecip_cld/total_cld)[dataslice].ravel()[validmask][newmask])
                np.add.at(number[4], (lwpinds[newmask], ndinds[newmask]), 1)
            except:
                pass
        print('')
    return number


def get_cccm_precip_hist(ndbins, lwpbins, lonrange, latrange):
    bins = {}
    bins['LWP'] = lwpbins
    bins['CDNC'] = ndbins

    number = np.zeros((2,  # High/low res lwp/cdnc
                       4,  # Precip flag
                       len(bins['LWP'])-1,
                       len(bins['CDNC'])-1))

    # For each CCCM file, get the Nd and LWP values from the MODIS 1deg mean. Sum up and calculate PoP as normal
    doycount = 0
    for year in range(2007, 2012):
        print(year, end=' ')
        for doy in range(1, 366):
            try:
                if (doy % 10) == 0:
                    print(doy, end=' ', flush=True)
                cdnc = read_cdnc(year, doy, sat='aqua', sds=['Nd_best'],
                                 versions=['7'])

                data = CCCM.readin('HDF', year, doy, ['Cloud group area percent coverage',
                                                      'Cloud top source flag',
                                                      'Cloud layer base level height',
                                                      'Precipitation flag CloudSat',  # 0 is no precip
                                                      'Cloud Classification',
                                                      'CERES SW TOA flux - downwards',
                                                      'Mean group cloud particle phase from MODIS radiance (3.7)',
                                                      'Mean group visible optical depth from MODIS radiance',
                                                      'Mean group water particle radius from MODIS rad (3.7)',
                                                      ])

                # Put latitude in [-90, 90]
                lat = 90-data['Colatitude of CERES FOV at surface']
                lon = data['Longitude of CERES FOV at surface']

                precip = data['Precipitation flag CloudSat'].astype('int')
                ocean = (data['Cloud Classification'] % 10 == 1)

                nlayers = (data['Cloud top source flag'] > 0).sum(axis=-1)
                base = data['Cloud layer base level height']
                liquid_flag = (
                    data['Mean group cloud particle phase from MODIS radiance (3.7)'][:, 0] == 1)

                re = data['Mean group water particle radius from MODIS rad (3.7)'][:, 0]
                cod = data['Mean group visible optical depth from MODIS radiance'][:, 0]

                # Adiabatic LWP and Nd calculations (following Quaas/Wood 2006)
                hr_lwp = (5/9*1000*re*cod)/1000
                hr_cdnc = (1.37e-5*((1e-6*re)**-2.5)*(cod**0.5))/1e6

                base_mask = ((base[:, :, 0] < 4) *
                             (nlayers == 1) *
                             (liquid_flag == 1) *
                             (np.isfinite(precip)) *
                             (ocean)[:, None] *
                             (lon > min(lonrange))[:, None] &
                             (lon < max(lonrange))[:, None] &
                             (lat > min(latrange))[:, None] &
                             (lat < max(latrange))[:, None] &
                             (data['CERES SW TOA flux - downwards'] > 30)[:, None])

                #Bin indicies
                hr_cdnc_ind = np.digitize(
                    hr_cdnc[base_mask], bins['CDNC'], right=True) - 1
                hr_lwp_ind = np.digitize(
                    hr_lwp[base_mask], bins['LWP'], right=True) - 1

                latind = np.digitize(
                    lat[np.where(base_mask)[0]], cdnc['Nd_best'].coords['lat']) - 1
                lonind = np.digitize(
                    lon[np.where(base_mask)[0]], cdnc['Nd_best'].coords['lon']) - 1
                lr_cdnc = cdnc['Nd_best'][0].values[lonind, latind]

                aquadata = read_subset(year, doy, sat='aqua', sds=[
                                       'CF_Ice', 'LWP_Liq'])
                lr_lwp = aquadata['LWP_Liq'][0].values[lonind, latind]
                lr_icf = aquadata['CF_Ice'][0].values[lonind, latind]
                icfmask = (lr_icf <= 1)

                np.add.at(number[0], (precip[base_mask][icfmask],
                                      hr_lwp_ind[icfmask],
                                      hr_cdnc_ind[icfmask]), 1)

                lr_cdnc_ind = np.digitize(
                    lr_cdnc, bins['CDNC'], right=True) - 1
                lr_lwp_ind = np.digitize(lr_lwp, bins['LWP'], right=True) - 1

                np.add.at(number[1], (precip[base_mask][icfmask],
                                      lr_lwp_ind[icfmask],
                                      lr_cdnc_ind[icfmask]), 1)
                doycount += 1
            except KeyboardInterrupt:
                splek
            except:
                import traceback
                traceback.print_exc()
                pass
        print('')
    print(doycount)
    return number


if readin:
    precip_hist = get_cccm_precip_hist(ndbins, lwpbins, lonrange, latrange)
    csat2.misc.fileops.nc4_dump(
        f'{save_directory}/preciphist_data_store.nc', precip_hist)
    precip_hist2 = get_precip_hist(ndbins, lwpbins, lonrange, latrange)
    csat2.misc.fileops.nc4_dump(
        f'{save_directory}/preciphist2_data_store.nc', precip_hist2)

precip_hist = csat2.misc.fileops.nc_load(
    f'{save_directory}/preciphist_data_store.nc', vname='var')
precip_hist2 = csat2.misc.fileops.nc_load(
    f'{save_directory}/preciphist2_data_store.nc', vname='var')


######################
# Initial plot setup #
######################
cmap = plt.get_cmap('RdBu_r')
cmap.set_bad('grey')
nd_lim = 60
lwp_lim_lo = 0
lwp_lim_hi = 1000


#############
# Map plots #
#############

nd_lim = 60
lwp_lim_lo = 0
lwp_lim_hi = 1000

mask = ((opdata['Nd'] > nd_lim) &
        np.isfinite(opdata['LWP']) &
        np.isfinite(opdata['aLWP']) &
        (opdata['LWP'] > lwp_lim_lo) &
        (opdata['LWP'] < lwp_lim_hi))
diff_hi = np.nanmean(
    np.where(mask, opdata['aLWP']-opdata['LWP'], np.nan), axis=0)
mask = ((opdata['Nd'] < nd_lim) &
        np.isfinite(opdata['LWP']) &
        np.isfinite(opdata['aLWP']) &
        (opdata['LWP'] > lwp_lim_lo) &
        (opdata['LWP'] < lwp_lim_hi))
diff_lo = np.nanmean(
    np.where(mask, opdata['aLWP']-opdata['LWP'], np.nan), axis=0)

ax = plt.subplot2grid((1, 9), (0, 0), colspan=4, projection=ccrs.PlateCarree())
ax.imshow((diff_hi-diff_lo).transpose()[30:-30], vmin=-30, vmax=30,
          cmap=cmap,
          extent=[-180, 180, -60, 60],
          aspect='auto')
ax.set_title(r'$\Delta$dLWP')
ax.coastlines()

ax.plot([-20, 10, 10, -20, -20], [-10, -10, -30, -30, -10], c='k', lw=2)
ax.text(-5, -20, 'A', va='center', ha='center')
csat2.misc.plotting.plt_sublabel_index(0)

for i, lwp_lim in enumerate([[10, 50], [50, 100], [100, 300], [300, 1000]]):
    lwp_lim_lo = lwp_lim[0]
    lwp_lim_hi = lwp_lim[1]
    mask = ((opdata['Nd'] > nd_lim) &
            np.isfinite(opdata['LWP']) &
            np.isfinite(opdata['aLWP']) &
            (opdata['LWP'] > lwp_lim_lo) &
            (opdata['LWP'] < lwp_lim_hi))
    diff_hi = np.nanmean(
        np.where(mask, opdata['aLWP']-opdata['LWP'], np.nan), axis=0)
    mask = ((opdata['Nd'] < nd_lim) &
            np.isfinite(opdata['LWP']) &
            np.isfinite(opdata['aLWP']) &
            (opdata['LWP'] > lwp_lim_lo) &
            (opdata['LWP'] < lwp_lim_hi))
    diff_lo = np.nanmean(
        np.where(mask, opdata['aLWP']-opdata['LWP'], np.nan), axis=0)

    ax = plt.subplot2grid((2, 9), ([0, 0, 1, 1][i], [
                          4, 6, 4, 6][i]), colspan=2, projection=ccrs.PlateCarree())
    ax.imshow((diff_hi-diff_lo).transpose()[30:-30], vmin=-30, vmax=30,
              cmap=cmap,
              extent=[-180, 180, -60, 60],
              aspect='auto')
    ax.coastlines()
    csat2.misc.plotting.plt_sublabel(
        ['b', 'c', 'd', 'e'][i] +
        ') {}-{}gm'.format(lwp_lim_lo, lwp_lim_hi)+r'$^{-2}$', size=8)

ax = plt.subplot2grid((5, 9), (1, 8), rowspan=3)
csat2.misc.plotting.plt_cbar(
    -30, 30, cmap,
    r'$\Delta$dLWP '+r'(gm$^{-2}$)',
    orientation='vertical')

fig = plt.gcf()
fig.set_size_inches(7, 3)
fig.savefig('output/dlwp_by_ndbest_adv.pdf', bbox_inches='tight')
fig.clf()
del(fig)


################
# ND changes #
################

lwp_lim = 60
nd_lim_lo = 0
nd_lim_hi = 1000
mask = ((opdata['LWP'] > lwp_lim) &
        np.isfinite(opdata['Nd']) &
        np.isfinite(opdata['aNd']) &
        (opdata['Nd'] > nd_lim_lo) &
        (opdata['Nd'] < nd_lim_hi))
diff_hi = np.nanmean(
    np.where(mask, opdata['aNd']-opdata['Nd'], np.nan), axis=0)
mask = ((opdata['LWP'] < lwp_lim) &
        np.isfinite(opdata['Nd']) &
        np.isfinite(opdata['aNd']) &
        (opdata['Nd'] > nd_lim_lo) &
        (opdata['Nd'] < nd_lim_hi))
diff_lo = np.nanmean(
    np.where(mask, opdata['aNd']-opdata['Nd'], np.nan), axis=0)

ax = plt.subplot2grid((1, 9), (0, 0), colspan=4, projection=ccrs.PlateCarree())
ax.imshow((diff_hi-diff_lo).transpose()[30:-30], vmin=-30, vmax=30,
          cmap=cmap,
          interpolation='nearest',
          extent=[-180, 180, -60, 60],
          aspect='auto')
ax.set_title(r'$\Delta$dN$_d$')
ax.coastlines()
csat2.misc.plotting.plt_sublabel_index(0)

lwp_lim = 60
for i, nd_lim in enumerate([[0, 25], [25, 100], [100, 300]]):
    nd_lim_lo = nd_lim[0]
    nd_lim_hi = nd_lim[1]
    mask = ((opdata['LWP'] > lwp_lim) &
            np.isfinite(opdata['Nd']) &
            np.isfinite(opdata['aNd']) &
            (opdata['Nd'] > nd_lim_lo) &
            (opdata['Nd'] < nd_lim_hi))
    diff_hi = np.nanmean(
        np.where(mask, opdata['aNd']-opdata['Nd'], np.nan), axis=0)
    mask = ((opdata['LWP'] < lwp_lim) &
            np.isfinite(opdata['Nd']) &
            np.isfinite(opdata['aNd']) &
            (opdata['Nd'] > nd_lim_lo) &
            (opdata['Nd'] < nd_lim_hi))
    diff_lo = np.nanmean(
        np.where(mask, opdata['aNd']-opdata['Nd'], np.nan), axis=0)

    ax = plt.subplot2grid(
        (2, 9), ([0, 0, 1][i], [4, 6, 4][i]), colspan=2, projection=ccrs.PlateCarree())
    ax.imshow((diff_hi-diff_lo).transpose()[30:-30], vmin=-30, vmax=30,
              cmap=cmap,
              interpolation='nearest',
              extent=[-180, 180, -60, 60],
              aspect='auto')
    ax.coastlines()
    csat2.misc.plotting.plt_sublabel(
        ['b', 'c', 'd', 'e'][i] +
        ') {}-{}cm'.format(nd_lim_lo, nd_lim_hi)+r'$^{-3}$', size=8)

ax = plt.subplot2grid((5, 9), (1, 8), rowspan=3)
csat2.misc.plotting.plt_cbar(
    -30, 30, cmap,
    r'$\Delta$dN$_d$ 'r'(cm$^{-3}$)',
    orientation='vertical')

fig = plt.gcf()
fig.set_size_inches(7, 3)
fig.savefig('output/dndbest_by_lwp_adv.pdf', bbox_inches='tight')
fig.clf()
del(fig)

###############################
# Flowfield plot calculations #
###############################

re = 15
cod = np.array([0.01, 5000])
lwp15 = (np.log((5/9*1000*re*cod)/1000) -
         np.log(lwpbins[0]))/((np.log(lwpbins[1])-np.log(lwpbins[0])))
cdnc15 = (np.log((1.37e-5*((1e-6*re)**-2.5)*(cod**0.5))/1e6) -
          np.log(ndbins[0]))/((np.log(ndbins[1])-np.log(ndbins[0])))

factor = 5
fig_scslice = np.s_[:, (lonrange[0]+180):(lonrange[1]+180),
                    (90-latrange[1]):(90-latrange[0])]

# Get flowfields and mean Nd/LWP fields
ff_dlwp, ff_dnd = get_flowfield(opdata, fig_scslice, ndbins, lwpbins)
bff_dlwp, bff_dnd = get_backflowfield(opdata, fig_scslice, ndbins, lwpbins)
jhist = get_hist(opdata, fig_scslice, ndbins, lwpbins)

# Mean windspeeds
ws_mean, ws_num = get_meansumfield(
    opdata, 'ws', fig_scslice, ndbins, lwpbins, factor, min_no=5)

# Precipitation histograms
ph = csat2.misc.stats.reduce_res_axis(
    csat2.misc.stats.reduce_res_axis(
        precip_hist[:, :, :240, :170], factor, axis=2, func=np.nansum),
    factor, axis=3, func=np.nansum)
toplot = ph[:, 1:].sum(axis=1)/ph.sum(axis=1)

# PoP - high res Nd and LWP
pop_hr = toplot[0]
# PoP - low res (1x1degree) Nd and LWP
pop_lr = toplot[1]

# Probability of small drops (<15um)
sdrops_mean, sdrops_num = get_meansumfield(
    opdata, 'small_drops', fig_scslice, ndbins, lwpbins, factor, min_no=5)
# Probability of any reff retrieval
adrops_mean, adrops_num = get_meansumfield(
    opdata, 'all_drops', fig_scslice, ndbins, lwpbins, factor, min_no=5)

# Fraction of retrievals >15um
re_plot = (adrops_mean-sdrops_mean)/adrops_mean

# Axes interpolators
xticks = [10, 30, 100, 300]
xlocs = scipy.interpolate.interp1d(ndbins, np.arange(len(ndbins))-0.5)(xticks)
xlocsf = scipy.interpolate.interp1d(
    ndbins[::factor], np.arange(len(ndbins[::factor]))-0.5)(xticks)
yticks = [10, 30, 100, 300, 1000]
ylocs = scipy.interpolate.interp1d(
    lwpbins, np.arange(len(lwpbins))-0.5)(yticks)
ylocsf = scipy.interpolate.interp1d(
    lwpbins[::factor], np.arange(len(lwpbins[::factor]))-0.5)(yticks)


###################
# Flowfield plots #
###################

plt.subplot2grid((4, 2), (0, 0), rowspan=3)
plt.imshow(ff_dlwp, vmin=-100, vmax=100, cmap=cmap,
           origin='lower', interpolation='nearest', aspect='auto')
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.ylabel('LWP (gm$^{-2}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, yticks)
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'a)')

plt.subplot2grid((4, 10), (3, 1), colspan=3)
csat2.misc.plotting.plt_cbar(-100, 100, cmap, r'$\Delta$LWP (3 hours)',
                             ticks=[-100, -50, 0, 50, 100], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((4, 2), (0, 1), rowspan=3)
plt.imshow(ff_dnd, vmin=-40, vmax=40, cmap=cmap, origin='lower',
           interpolation='nearest', aspect='auto')
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, ['']*len(ylocs))
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'b)')

plt.subplot2grid((4, 10), (3, 6), colspan=3)
csat2.misc.plotting.plt_cbar(-40, 40, cmap, r'$\Delta$N$_d$ (3 hours)',
                             ticks=[-40, -20, 0, 20, 40], title_pos='below', aspect_ratio=0.05)

plt.subplots_adjust(hspace=0.5)
fig = plt.gcf()
fig.set_size_inches(6, 5)
fig.savefig('output/flowfield_adv.pdf', bbox_inches='tight')
plt.clf()
del(fig)

######################
# Relative flowfield #
######################

ff_dlwp_rel = ff_dlwp/csat2.misc.stats.lin_av(lwpbins)[:, None]
ff_dnd_rel = ff_dnd/csat2.misc.stats.lin_av(ndbins)[None, :]
bff_dlwp_rel = bff_dlwp/csat2.misc.stats.lin_av(lwpbins)[:, None]
bff_dnd_rel = bff_dnd/csat2.misc.stats.lin_av(ndbins)[None, :]
jhist_lwp = nan_gf(
    csat2.misc.stats.normalise(jhist/np.diff(lwpbins)[:, None], axis=0), sigma=1)
sjhist_lwp = nan_gf(
    csat2.misc.stats.normalise(jhist, axis=0), sigma=1)
jhist_nd = nan_gf(
    csat2.misc.stats.normalise(jhist/np.diff(ndbins)[None, :], axis=1), sigma=1)
sjhist_nd = nan_gf(
    csat2.misc.stats.normalise(jhist, axis=1), sigma=1)

# LWP and Nd contours
# These should not be normalised by bin width
# They need to show the fraction of total data
cvals = [0.25, 0.5, 0.75]
minval = 20
nd_c = [np.where(np.nansum(jhist, axis=1) > minval,
                 np.argmax(np.nancumsum(sjhist_nd, axis=1) > c, axis=1),
                 np.nan) for c in cvals]
lwp_c = [np.where(np.nansum(jhist, axis=0) > minval,
                  np.argmax(np.nancumsum(sjhist_lwp, axis=0) > c, axis=0),
                  np.nan)for c in cvals]


xticks = [10, 30, 100, 300]
xlocs = scipy.interpolate.interp1d(ndbins, np.arange(len(ndbins))-0.5)(xticks)
yticks = [10, 30, 100, 300, 1000]
ylocs = scipy.interpolate.interp1d(
    lwpbins, np.arange(len(lwpbins))-0.5)(yticks)


cmap = plt.get_cmap('seismic')

plt.subplot2grid((12, 2), (0, 0), rowspan=3)
plt.imshow(jhist_lwp, vmin=0, vmax=0.04, cmap=plt.get_cmap('WBGYR'),
           origin='lower', interpolation='nearest', aspect='auto')
for lwpc in lwp_c:
    mask = np.isfinite(lwpc)
    plt.plot(np.arange(len(lwpc))[mask], lwpc[mask], c='k')
plt.contour(nan_gf(jhist, sigma=1), levels=[
            30], colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(nan_gf(jhist, sigma=1), levels=[0, 30], colors=['k'], alpha=0.2)
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.ylabel('LWP (gm$^{-2}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, yticks)
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'a)')
plt.subplot2grid((12, 10), (3, 1), colspan=3)
csat2.misc.plotting.plt_cbar(0, 4, plt.get_cmap('WBGYR'), r'P(LWP|N$_d$) (%/gm$^{-2}$)',
                             ticks=[0, 1, 2, 3, 4], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((12, 2), (0, 1), rowspan=3)
plt.imshow(nan_gf(jhist_nd, sigma=1), vmin=0, vmax=0.04, cmap=plt.get_cmap(
    'WBGYR'), origin='lower', interpolation='nearest', aspect='auto')
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask], np.arange(len(ndc))[mask], c='k')
plt.contour(nan_gf(jhist, sigma=1), levels=[
            30], colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(nan_gf(jhist, sigma=1), levels=[0, 30], colors=['k'], alpha=0.2)
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, ['']*len(ylocs))
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'b)')

plt.subplot2grid((12, 10), (3, 6), colspan=3)
csat2.misc.plotting.plt_cbar(0, 4, plt.get_cmap('WBGYR'), r'P(N$_d$|LWP) (%/cm$^{-3}$)',
                             ticks=[0, 1, 2, 3, 4], title_pos='below', aspect_ratio=0.05)


# Basic flowfields
plt.subplot2grid((12, 2), (4, 0), rowspan=3)
plt.imshow(100*ff_dlwp_rel/3, vmin=-50, vmax=50, cmap=cmap,
           origin='lower', interpolation='nearest', aspect='auto')
for lwpc in lwp_c:
    mask = np.isfinite(lwpc)
    plt.plot(np.arange(len(lwpc))[mask], lwpc[mask], c='k')
plt.contour(nan_gf(jhist, sigma=1), levels=[
            30], colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(nan_gf(jhist, sigma=1), levels=[0, 30], colors=['k'], alpha=0.2)
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.ylabel('LWP (gm$^{-2}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, yticks)
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'c)')

plt.subplot2grid((12, 10), (7, 1), colspan=3)
csat2.misc.plotting.plt_cbar(-100, 100, cmap, r'dLWP (%/hr)',
                             ticks=[-50, -25, 0, 25, 50], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((12, 2), (4, 1), rowspan=3)
plt.imshow(100*ff_dnd_rel/3, vmin=-30, vmax=30, cmap=cmap,
           origin='lower', interpolation='nearest', aspect='auto')
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask], np.arange(len(ndc))[mask], c='k')
plt.contour(nan_gf(jhist, sigma=1), levels=[
            30], colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(nan_gf(jhist, sigma=1), levels=[0, 30], colors=['k'], alpha=0.2)
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, ['']*len(ylocs))
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'd)')

plt.subplot2grid((12, 10), (7, 6), colspan=3)
csat2.misc.plotting.plt_cbar(-100, 100, cmap, r'dN$_d$ (%/hr)',
                             ticks=[-30, -15, 0, 15, 30], title_pos='below', aspect_ratio=0.05)

# Back flowfields
plt.subplot2grid((12, 2), (8, 0), rowspan=3)
plt.imshow(100*bff_dlwp_rel/3, vmin=-50, vmax=50, cmap=cmap,
           origin='lower', interpolation='nearest', aspect='auto')
for lwpc in lwp_c:
    mask = np.isfinite(lwpc)
    plt.plot(np.arange(len(lwpc))[mask], lwpc[mask], c='k')
plt.contour(nan_gf(jhist, sigma=1), levels=[
            30], colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(nan_gf(jhist, sigma=1), levels=[0, 30], colors=['k'], alpha=0.2)
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.ylabel('LWP (gm$^{-2}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, yticks)
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'e)')

plt.subplot2grid((12, 10), (11, 1), colspan=3)
csat2.misc.plotting.plt_cbar(-100, 100, cmap, r'dLWP (%/hr)',
                             ticks=[-50, -25, 0, 25, 50], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((12, 2), (8, 1), rowspan=3)
plt.imshow(100*bff_dnd_rel/3, vmin=-30, vmax=30, cmap=cmap,
           origin='lower', interpolation='nearest', aspect='auto')
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask], np.arange(len(ndc))[mask], c='k')
plt.contour(nan_gf(jhist, sigma=1), levels=[
            30], colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(nan_gf(jhist, sigma=1), levels=[0, 30], colors=['k'], alpha=0.2)
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocs, xticks)
plt.yticks(ylocs, ['']*len(ylocs))
plt.ylim(60, 190)
csat2.misc.plotting.plt_sublabel(r'f)')

plt.subplot2grid((12, 10), (11, 6), colspan=3)
csat2.misc.plotting.plt_cbar(-100, 100, cmap, r'dN$_d$ (%/hr)',
                             ticks=[-30, -15, 0, 15, 30], title_pos='below', aspect_ratio=0.05)

plt.subplots_adjust(hspace=0.5)
fig = plt.gcf()
fig.set_size_inches(6, 12)
fig.savefig('output/flowfield_adv_rel.pdf', bbox_inches='tight')
plt.clf()
del(fig)

#########################
# Meteorological fields #
#########################

cmap = plt.get_cmap('RdBu_r')

lwpgrid, ndgrid = np.meshgrid(lwpbins[::5], ndbins[::5])
exps = [['KK', 2.47, 1.79, 'k'],
        ['TC80', 7/3, 1/3, 'r'],
        ['B94', 4.7, 3.3, 'g'],
        ['LD04', 3, 1, 'b']]

# Pop plots
plt.subplot2grid((4, 4), (0, 0), rowspan=3)
plt.imshow(100*pop_hr, vmin=0, vmax=100, cmap=plt.get_cmap('WBGYR'),
           origin='lower', interpolation='nearest', aspect='auto')
plt.contour(ph[:, :].sum(axis=1)[0], levels=[30],
            colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(ph[:, :].sum(axis=1)[0], levels=[0, 30], colors=['k'], alpha=0.2)
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask]/5, np.arange(len(ndc))[mask]/5, c='k')
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.ylabel('LWP (gm$^{-2}$)')
xlim = plt.xlim()
#plt.plot(cdnc15, lwp15, c='k')
for ep in exps:
    acr = (lwpgrid**(ep[1]/2)*ndgrid**(-ep[2])).transpose()
    lev = acr[28, 17]
    plt.contour(acr, levels=[lev], colors=[ep[3]], label=ep[0])
plt.xticks(xlocsf, xticks)
plt.yticks(ylocsf, yticks)
plt.ylim(60/factor, 190/factor)
plt.xlim(*xlim)
csat2.misc.plotting.plt_sublabel(r'a)')

plt.subplot2grid((4, 20), (3, 1), colspan=3)
csat2.misc.plotting.plt_cbar(0, 100, plt.get_cmap('WBGYR'), r'PoP (%)',
                             ticks=[0, 25, 50, 75, 100], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((4, 4), (0, 1), rowspan=3)
plt.imshow(100*pop_lr, vmin=0, vmax=100, cmap=plt.get_cmap('WBGYR'),
           origin='lower', interpolation='nearest', aspect='auto')
plt.contour(ph[:, :].sum(axis=1)[1], levels=[30],
            colors='grey', linestyles=[':', '--'], linewidths=1)
plt.contourf(ph[:, :].sum(axis=1)[1], levels=[0, 30], colors=['k'], alpha=0.2)
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask]/5, np.arange(len(ndc))[mask]/5, c='k')
xlim = plt.xlim()
for ep in exps:
    acr = (lwpgrid**(ep[1]/2)*ndgrid**(-ep[2])).transpose()
    lev = acr[28, 17]
    plt.contour(acr, levels=[lev], colors=[ep[3]], label=ep[0])
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocsf, xticks)
plt.yticks(ylocsf, ['']*len(ylocs))
plt.ylim(60/factor, 190/factor)
plt.xlim(*xlim)
csat2.misc.plotting.plt_sublabel(r'b)')

plt.subplot2grid((4, 20), (3, 6), colspan=3)
csat2.misc.plotting.plt_cbar(0, 100, plt.get_cmap('WBGYR'), r'PoP (%)',
                             ticks=[0, 25, 50, 75, 100], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((4, 4), (0, 2), rowspan=3)
plt.imshow(100*re_plot, vmin=0, vmax=100, cmap=plt.get_cmap('WBGYR'),
           origin='lower', interpolation='nearest', aspect='auto')
plt.contour(adrops_num, levels=[30], colors='grey',
            linestyles=[':', '--'], linewidths=1)
plt.contourf(adrops_num, levels=[0, 30], colors=['k'], alpha=0.2)
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask]/5, np.arange(len(ndc))[mask]/5, c='k')
xlim = plt.xlim()
for ep in exps:
    acr = (lwpgrid**(ep[1]/2)*ndgrid**(-ep[2])).transpose()
    lev = acr[28, 17]
    plt.contour(acr, levels=[lev], colors=[ep[3]], label=ep[0])
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocsf, xticks)
plt.yticks(ylocsf, ['']*len(ylocs))
plt.ylim(60/factor, 190/factor)
plt.xlim(*xlim)
csat2.misc.plotting.plt_sublabel(r'c)')

plt.subplot2grid((4, 20), (3, 11), colspan=3)
csat2.misc.plotting.plt_cbar(0, 100, plt.get_cmap('WBGYR'), r'P(r$_e$>15$\mu$m) (%)',
                             ticks=[0, 25, 50, 75, 100], title_pos='below', aspect_ratio=0.05)

plt.subplot2grid((4, 4), (0, 3), rowspan=3)
plt.imshow(ws_mean, vmin=2, vmax=8, cmap=plt.get_cmap('WBGYR'),
           origin='lower', interpolation='nearest', aspect='auto')
plt.contour(ws_num, levels=[30], colors='grey',
            linestyles=[':', '--'], linewidths=1)
plt.contourf(ws_num, levels=[0, 30], colors=['k'], alpha=0.2)
for ndc in nd_c:
    mask = np.isfinite(ndc)
    plt.plot(ndc[mask]/5, np.arange(len(ndc))[mask]/5, c='k')
xlim = plt.xlim()
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.xticks(xlocsf, xticks)
plt.yticks(ylocsf, ['']*len(ylocs))
plt.ylim(60/factor, 190/factor)
plt.xlim(*xlim)
csat2.misc.plotting.plt_sublabel(r'd)')

plt.subplot2grid((4, 20), (3, 16), colspan=3)
csat2.misc.plotting.plt_cbar(2, 8, plt.get_cmap('WBGYR'), '10m Windspeed'+r' (m$^{-1}$)', nticks=4,
                             ticks=[2, 4, 6, 8], title_pos='below', aspect_ratio=0.05)

plt.subplots_adjust(hspace=0.5)
fig = plt.gcf()
fig.set_size_inches(8, 4)
fig.savefig('output/metfield.pdf', bbox_inches='tight')
plt.clf()
del(fig) 



##########################
# Sensitivity time plots #
##########################

# Timescale for integration (in hours)
tscale = 3
# Define the flowfield variables
ff_dlwp, ff_dnd = get_flowfield(opdata, fig_scslice, ndbins, lwpbins)
# Integrate forwards
o_nd, o_lwp = integrate_flowfield(
    ff_dlwp, ff_dnd, ndbins, lwpbins, tsteps=90, scale=tscale)
vo_nd, vo_lwp = integrate_flowfield(
    ff_dlwp, ff_dnd, ndbins, lwpbins, randscale=20/tscale, tsteps=90, scale=tscale)

ts = 8*tscale
plt.subplot(131)
plt.scatter(np.log(o_nd[0][::2]), np.log(o_lwp[0][::2]), s=2, c='lightgrey')
plt.scatter(np.log(o_nd[1*tscale][::5]),
            np.log(o_lwp[1*tscale][::5]), s=2, c='grey')
csat2.misc.plotting.plt_bestfit(np.log(o_nd[ts][::5]), np.log(
    o_lwp[ts][::5]), s=2, c='k', stats=False)
plt.xticks(np.log([10, 30, 100, 300]), [10, 30, 100, 300])
plt.yticks(np.log([10, 30, 100, 300]), [10, 30, 100, 300])
plt.xlim(np.log(10), np.log(300))
plt.ylim(np.log(10), np.log(300))
plt.xlabel(r'N$_d$ (cm$^{-3}$)')
plt.ylabel(r'LWP (gm$^{-2}$)')
csat2.misc.plotting.plt_sublabel('a)', size=10)

plt.subplot(132)
csat2.misc.plotting.plt_bestfit(np.log(o_nd[0][::5]), np.log(
    o_lwp[ts][::5]), s=2, c='k', stats=False)
plt.xticks(np.log([10, 30, 100, 300]), [10, 30, 100, 300])
plt.yticks(np.log([10, 30, 100, 300]), [10, 30, 100, 300])
plt.xlim(np.log(10), np.log(300))
plt.ylim(np.log(10), np.log(300))
plt.xlabel(r'Initial N$_d$ (cm$^{-3}$)')
csat2.misc.plotting.plt_sublabel(r'b)', size=10)

plt.subplot(133)
slopes_initial_nd = [csat2.misc.stats.nanlinregress(
    np.log(o_nd[0]), np.log(o_lwp[i])).slope for i in range(len(o_nd))]
slopes = [csat2.misc.stats.nanlinregress(
    np.log(o_nd[i]), np.log(o_lwp[i])).slope for i in range(len(o_nd))]

plt.plot(slopes, label=r'N$_d$-LWP')
plt.plot(slopes_initial_nd, label=r'Initial N$_d$-LWP')

nslopes_initial_nd = [csat2.misc.stats.nanlinregress(
    np.log(vo_nd[0]), np.log(vo_lwp[i])).slope for i in range(len(vo_nd))]
nslopes = [csat2.misc.stats.nanlinregress(
    np.log(vo_nd[i]), np.log(vo_lwp[i])).slope for i in range(len(vo_nd))]

plt.plot(nslopes, label=r'N$_d$-LWP (noise)')
plt.plot(nslopes_initial_nd, label=r'Initial N$_d$-LWP (noise)')
plt.legend(fontsize=8, frameon=False, labelspacing=0.15, loc='lower left')
plt.ylim(-1, 0)
plt.xlim(0, 10)
plt.xticks([0, 5*tscale, 10*tscale], [0, 15, 30])
plt.xlabel('Time (hours)')
plt.ylabel(r'$\frac{d\ln LWP}{d\ln N_d}$')
csat2.misc.plotting.plt_sublabel('c)', size=10)

plt.subplots_adjust(wspace=0.5)

fig = plt.gcf()
fig.set_size_inches(9, 3.5)
fig.savefig('output/sensitivity_time_adv.pdf', bbox_inches='tight')
fig.clf()
del(fig)


####################
# Get the map data #
####################

output_slopes = np.zeros((8, 36, 18))*np.nan
for i in range(0, 36):
    print(i)
    for j in range(0, 18):
        scslice = np.s_[:, 10*i:10*(i+1), 10*j:10*(j+1)]

        ff_dlwp, ff_dnd = get_flowfield(opdata, scslice, ndbins, lwpbins)
        # Integrate forwards
        o_nd, o_lwp = integrate_flowfield(ff_dlwp, ff_dnd, ndbins, lwpbins, tsteps=10)

        try:
            slopes_initial_nd = [csat2.misc.stats.nanlinregress(
                np.log(o_nd[0]), np.log(o_lwp[i])).slope for i in range(len(o_nd))]
            slopes = [csat2.misc.stats.nanlinregress(
                np.log(o_nd[i]), np.log(o_lwp[i])).slope for i in range(len(o_nd))]
            
            output_slopes[0, i, j] = np.nanmean(slopes[-5:])
            output_slopes[1, i, j] = np.nanmean(slopes_initial_nd[-5:])
            output_slopes[2, i, j] = np.nanmin(slopes)
            output_slopes[3, i, j] = np.nanmin(slopes_initial_nd)
            output_slopes[4, i, j] = np.nanmax(slopes)
            output_slopes[5, i, j] = np.nanmax(slopes_initial_nd)
            output_slopes[6, i, j] = np.nanmean(slopes[6:8])
            output_slopes[7, i, j] = np.nanmean(slopes_initial_nd[6:8])
        except:
            pass


#######################
# Make some map plots #
#######################

slopes = np.zeros(output_slopes[0].shape)*np.nan
slopes_ind = np.zeros(output_slopes[0].shape)*np.nan

slopes = np.where((np.abs(output_slopes[2]) > np.abs(output_slopes[4])) &
                  (np.abs(output_slopes[4]) < 0.02),
                  output_slopes[2], slopes)
slopes = np.where((np.abs(output_slopes[2]) < np.abs(output_slopes[4])) &
                  (np.abs(output_slopes[2]) < 0.02),
                  output_slopes[4], slopes)
slopes = output_slopes[6]
slopes_ind = np.where((np.abs(output_slopes[3]) > np.abs(output_slopes[5])) &
                      (np.abs(output_slopes[5]) < 0.2),
                      output_slopes[3], slopes_ind)
slopes_ind = np.where((np.abs(output_slopes[3]) < np.abs(output_slopes[5])) &
                      (np.abs(output_slopes[3]) < 0.2),
                      output_slopes[5], slopes_ind)
slopes_ind = output_slopes[7]


vmax = 1
plt.subplot2grid((7, 1), (0, 0), rowspan=3, projection=ccrs.PlateCarree())
plt.imshow(slopes.transpose()[3:-3], cmap=cmap, vmin=-vmax,
           vmax=vmax, extent=[-180, 180, -60, 60], aspect='auto')
plt.gca().coastlines()
csat2.misc.plotting.plt_sublabel('a) Instananeous measurement')

plt.subplot2grid((7, 1), (3, 0), rowspan=3, projection=ccrs.PlateCarree())
plt.imshow(slopes_ind.transpose()[3:-3], cmap=cmap, vmin=-
           vmax, vmax=vmax, extent=[-180, 180, -60, 60], aspect='auto')
plt.gca().coastlines()
csat2.misc.plotting.plt_sublabel(r'b) Initial N$_d$')

plt.subplot2grid((7, 5), (6, 1), colspan=3)
csat2.misc.plotting.plt_cbar(-vmax, vmax, cmap, r'N$_d$-LWP sensitivity', ticks=[-vmax, -vmax/2, 0, vmax/2, vmax],
                             aspect_ratio=0.05)

plt.subplots_adjust(hspace=0.4)
fig = plt.gcf()
fig.set_size_inches(4, 5)
fig.savefig('output/sens_maps_adv.pdf', bbox_inches='tight')
