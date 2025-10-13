import glob
import argparse
import pathlib
import pygrib
import json
import pygridder as pg

import numpy as np
from scipy import stats
from shapely.geometry import shape
import pandas as pd

import pyproj
import datetime as dt
import glob

import math
import warnings
warnings.filterwarnings('ignore')

from fipsvars import fipsToState, cig_dists, mags, levs
from utils import col_names_outlook, col_names_wind, col_names_hail, col_names_href, models
from utils import gather_features, gather_ct_features, fix_neg, add_distribution_empirical_corr, flatten_list, return_proportion
from make_plots import make_plots
from pdb import set_trace as st


path_to_covoutlooks = '/Users/josephpicca/projects/impacts/transfer-assets/transfer-portal/lsr-pred-verf/outlook/'
cov_outlooks_1300_2023 = glob.glob(path_to_covoutlooks + 'torn*day1_grib2_1300_2023*')
cov_outlooks_1630_2023 = glob.glob(path_to_covoutlooks + 'torn*day1_grib2_1630_2023*')
cov_outlooks_1300_2024 = glob.glob(path_to_covoutlooks + 'torn*day1_grib2_1300_2024*')
cov_outlooks_1630_2024 = glob.glob(path_to_covoutlooks + 'torn*day1_grib2_1630_2024*')

path_to_conotlks = '/Users/josephpicca/projects/impacts/transfer-assets/transfer-portal/lsr-pred-verf/condint/torn_ci_npz/'
path_to_href = '/Users/josephpicca/projects/impacts/transfer-assets/transfer-portal/lsr-pred-verf/href/'

impacts_grids_file = '../data/impact-grids-5km.npz'
cwa_file = '../data/cwas.npz'
corr_file = '../data/correlation-matrices.npz'

with np.load(impacts_grids_file) as NPZ:
    population = NPZ["population"]
    proj = pyproj.Proj(NPZ["srs"].item())
    geod = pyproj.Geod(f'{proj} +a=6371200 +b=6371200')
    lons = NPZ["lons"]
    lats = NPZ["lats"]
    X = NPZ["X"]
    Y = NPZ["Y"]
    dx = NPZ["dx"]
    dy = NPZ["dy"]
    state = NPZ["state"]
    county = NPZ["county"]

with np.load(cwa_file) as NPZ:
    wfo = NPZ['cwas']

map_func = np.vectorize(lambda x: fipsToState.get(x, '0'))
wfo_state_2d = np.char.add(wfo.astype('str'),map_func(state))

with np.load(corr_file,allow_pickle=True) as NPZ:
    corr_wind = NPZ['wind']
    corr_hail = NPZ['hail']
    corr_dict = {'wind': corr_wind, 'hail': corr_hail}

    area_order = NPZ['area_order']

cov_outlooks = cov_outlooks_1300_2023 + cov_outlooks_1630_2023 + cov_outlooks_1300_2024 + cov_outlooks_1630_2024

# Read grib file
def read_ndfd_grib_file(grbfile,which='torn'):
    """ Read an SPC Outlook NDFD Grib2 File """
    if which == 'torn':
        with pygrib.open(grbfile.as_posix()) as GRB:
            try:
                vals = GRB[1].values.filled(-1)
            except AttributeError:
                vals = GRB[1].values
        return vals
    else:
        with pygrib.open(grbfile.as_posix().replace('torn',which)) as GRB:
            try:
                vals = GRB[1].values.filled(-1)
            except AttributeError:
                vals = GRB[1].values
        return vals
    
def read_con_npz_file(npzfile,which='torn'):
    # Read continuous cig file
    if which == 'torn':
        with np.load(npzfile.as_posix()) as NPZ:
            vals = NPZ['vals']
    else:
        with np.load(npzfile.as_posix().replace('torn',which)) as NPZ:
            vals = NPZ['vals']

    vals[vals < 1] = 0
    vals[np.logical_and(vals >= 1, vals < 2)] = 1
    vals[vals >= 2] = 2

    return vals

def get_doy(X):
    angle = np.arctan2(X.toy_sin[0],X.toy_cos[0])

    # Normalize to [0, 1)
    normalized_doy = (angle % (2 * np.pi)) / (2 * np.pi)

    # Convert to day of year (1–366)
    day_of_year = int(round(normalized_doy * 366))

    return day_of_year

def generate_features_preds(cov_outlook):
    cov_outlook_path = pathlib.Path(cov_outlook)
    otlk_ts = cov_outlook_path.as_posix().split('_')[-1]
    year = int(otlk_ts[:4])
    month = int(otlk_ts[4:6])
    day = int(otlk_ts[6:8])
    valid_hr = int(cov_outlook_path.as_posix().split('_')[-2][:2])
    otlkdt = dt.datetime(year, month, day, valid_hr)
    otlkdt_formatted = otlkdt.strftime('%Y%m%d%H%M')

    # read coverage and continuous files
    torn_cov = read_ndfd_grib_file(cov_outlook_path, which='torn')
    hail_cov = read_ndfd_grib_file(cov_outlook_path, which='hail')
    wind_cov = read_ndfd_grib_file(cov_outlook_path, which='wind')

    # Check if coverage files have probs > 0
    if hail_cov.max() == 0 and wind_cov.max() == 0:
        return None
    
    con_otlk_match = glob.glob(path_to_conotlks + f'torn_ci_day1_{otlkdt_formatted}_*.npz')[0]
    con_file = pathlib.Path(con_otlk_match)
    
    torn_con = read_con_npz_file(con_file, which='torn')
    hail_con = read_con_npz_file(con_file, which='hail')
    wind_con = read_con_npz_file(con_file, which='wind')

    # get wfo-states in outlooks
    wfo_st_windoutlook = np.unique(wfo_state_2d[wind_cov > 0])
    wfo_st_hailoutlook = np.unique(wfo_state_2d[hail_cov > 0])
    wfo_st_tornoutlook = np.unique(wfo_state_2d[torn_cov > 0])

    wfo_st_alloutlook = np.concatenate([wfo_st_windoutlook,wfo_st_hailoutlook,wfo_st_tornoutlook])
    wfo_st_unique = np.unique(wfo_st_alloutlook[[ '0' not in s for s in wfo_st_alloutlook]])

    # Outlook based features
    df_otlk = gather_features(wfo_st_unique, wfo_state_2d, otlkdt, 
                        wind_cov, hail_cov, torn_cov, 
                        wind_con, hail_con, torn_con, 
                        population)
    df_otlk.columns = col_names_outlook  # Rename columns to assist merge with HREF features
    df_otlk['doy'] = get_doy(df_otlk)

    ct_files = glob.glob(f'{path_to_href}{otlkdt.year}{otlkdt.month:02d}{otlkdt.day:02d}/thunder/spc_post.t00z.hrefct.1hr.f*')
    ct_files.sort()

    ct_arrs = []

    if len(ct_files) < 48:
        return None
    
    for i,file in enumerate(ct_files[otlkdt.hour-1:35]):

        with pygrib.open(file) as GRB:
            
            try:
                vals = GRB.values.filled(-1)
                lats_href, lons_href = GRB.latlons()
            except AttributeError:
                vals = GRB[1].values
                lats_href, lons_href = GRB[1].latlons()

        ct_arrs.append(vals)

    val_stack = np.stack(ct_arrs,axis=0)

    # get all ndfd grid boxes associated with wfo-st's touched by an outlook
    wfo_st_impacted = np.isin(wfo_state_2d,wfo_st_unique)

    # get 1d list of wfo-st for each ndfd grid box
    wfo_st_1d = wfo_state_2d[wfo_st_impacted]

    G_href = pg.Gridder(lons_href, lats_href)

    # find the href grid indices for these ndfd grids
    idx_href = G_href.grid_points(lons[wfo_st_impacted],lats[wfo_st_impacted])

    # Get all HREFCT probabilities for all times in each CWA-ST
    ct_vals = []
    for idx in idx_href:
        ct_vals.append(val_stack[:,idx[0],idx[1]])

    ct_vals = np.array(ct_vals)

    df_href = gather_ct_features(wfo_st_unique, wfo_st_1d, otlkdt, ct_vals, population, wfo_state_2d)
    df_href.columns = col_names_href # Rename columns to assist merge with outlook features
    
    df_all = df_otlk.merge(df_href, on=['otlk_timestamp','wfo_st_list'], how='inner').fillna(0)

    # df_all.to_csv(f'../feature-files/{otlkdt_formatted}_features.csv',index=False)

    wfo_list_trans = models['le'].transform(df_all.wfo_st_list.str.slice(0,3))

    X = df_all.iloc[:,2:]
    X['wfo'] = wfo_list_trans
    # Xsig = X[X.columns.drop(list(X.filter(regex='pop')))]
    Xwind = X[col_names_wind]
    Xhail = X[col_names_hail]

    wind_preds = np.round(models['wind'].predict(Xwind))
    wind_preds = fix_neg(wind_preds)

    hail_preds = np.round(models['hail'].predict(Xhail))
    hail_preds = fix_neg(hail_preds)

    df_preds = pd.DataFrame({
        'wfost': df_all.wfo_st_list,
        'wind': wind_preds,
        'hail': hail_preds,
        # 'sigwind': sigwind_preds,
        # 'sighail': sighail_preds,
    })

    big_preds, med_preds, small_preds = add_distribution_empirical_corr(df_preds,corr_dict,area_order)

    def add_sig_counts(df_preds):

        final_sigwind_counts = []
        final_sighail_counts = []
        
        for idx,row in df_preds.iterrows():
            # wind
            cov_probs = wind_cov[wfo_state_2d == row.wfost]
            con_probs = wind_con[wfo_state_2d == row.wfost]
            cumulative_weights = cov_probs.cumsum()
        
            if cov_probs.sum() == 0:
        
                sigwind_dists = np.zeros(10000).astype(int)
        
            else:
        
                all_reps_sum = int(row.wind_dists.sum())
                _locs = np.random.randint(
                            4.9, cumulative_weights.max(), size=all_reps_sum)
                locs = cumulative_weights.searchsorted(_locs)
            
                con_reports = con_probs[locs]
            
                all_ratings = []
            
                for con_lvl in levs:
                    con_inds = con_reports == con_lvl

                    if con_lvl == 0:

                        mo = int(otlk_ts[4:6])
                        loc = row.wfost[:3]
                        sig_wind_pct = return_proportion(mo,loc,'wind')

                        all_ratings.append(
                            np.random.choice(mags, size=con_inds.sum(), 
                                            replace=True, p=[1-sig_wind_pct,sig_wind_pct])
                        )

                    else:
                
                        all_ratings.append(
                            np.random.choice(mags, size=con_inds.sum(),
                                                replace=True, p=cig_dists['wind'][str(con_lvl)])
                        )
            
                all_ratings_flat = flatten_list(all_ratings)
                np.random.shuffle(all_ratings_flat)
                _sims = np.split(all_ratings_flat, row.wind_dists.astype(int).cumsum())[:-1]
                sigwind_dists = [arr.sum() for arr in _sims]
        
            final_sigwind_counts.append(np.array(sigwind_dists))
        
            # hail
            cov_probs = hail_cov[wfo_state_2d == row.wfost]
            con_probs = hail_con[wfo_state_2d == row.wfost]
            cumulative_weights = cov_probs.cumsum()
        
            if cov_probs.sum() == 0:
        
                sighail_dists = np.zeros(10000).astype(int)
        
            else:
        
                all_reps_sum = int(row.hail_dists.sum())
                _locs = np.random.randint(
                            4.9, cumulative_weights.max(), size=all_reps_sum)
                locs = cumulative_weights.searchsorted(_locs)
            
                con_reports = con_probs[locs]
            
                all_ratings = []
            
                for con_lvl in levs:
                    con_inds = con_reports == con_lvl

                    if con_lvl == 0:

                        mo = int(otlk_ts[4:6])
                        loc = row.wfost[:3]
                        sig_hail_pct = return_proportion(mo,loc,'hail')

                        all_ratings.append(
                            np.random.choice(mags, size=con_inds.sum(), 
                                            replace=True, p=[1-sig_hail_pct,sig_hail_pct])
                        )
                    else:
                
                        all_ratings.append(
                            np.random.choice(mags, size=con_inds.sum(),
                                                replace=True, p=cig_dists['hail'][str(con_lvl)])
                        )
            
                all_ratings_flat = flatten_list(all_ratings)
                np.random.shuffle(all_ratings_flat)
                _sims = np.split(all_ratings_flat, row.hail_dists.astype(int).cumsum())[:-1]
                sighail_dists = [arr.sum() for arr in _sims]
        
            final_sighail_counts.append(np.array(sighail_dists))

        df_preds['sigwind_dists'] = final_sigwind_counts
        df_preds['sighail_dists'] = final_sighail_counts

        return df_preds

    big_preds = add_sig_counts(big_preds)
    med_preds = add_sig_counts(med_preds)
    small_preds = add_sig_counts(small_preds)

    all_preds = pd.concat([big_preds,med_preds,small_preds])

    wind_bool = np.array([int(bool((wind_cov[wfo_state_2d == place]).sum())) for place in all_preds.wfost])
    hail_bool = np.array([int(bool((hail_cov[wfo_state_2d == place]).sum())) for place in all_preds.wfost])

    all_preds['hail_dists'] = all_preds.hail_dists * hail_bool
    all_preds['sighail_dists'] = all_preds.sighail_dists * hail_bool

    all_preds['wind_dists'] = all_preds.wind_dists * wind_bool
    all_preds['sigwind_dists'] = all_preds.sigwind_dists * wind_bool

    loc_list = []
    # National Verification
    summed_nat = all_preds.sum()
    wind_percs = np.percentile(summed_nat.wind_dists,q=np.arange(5,100,5))
    sigwind_percs = np.percentile(summed_nat.sigwind_dists,q=np.arange(5,100,5))
    hail_percs = np.percentile(summed_nat.hail_dists,q=np.arange(5,100,5))
    sighail_percs = np.percentile(summed_nat.sighail_dists,q=np.arange(5,100,5))

    all_percs = np.concatenate([wind_percs,sigwind_percs,hail_percs,sighail_percs])

    loc_list.append('National')

    # State Verification
    affected_states = np.unique(all_preds.wfost.str.slice(3,5))
    for affected in affected_states:
        affected_df = all_preds[all_preds.wfost.str.slice(3,5) == affected]

        summed_state = affected_df.sum()
        wind_percs = np.percentile(summed_state.wind_dists,q=np.arange(5,100,5))
        sigwind_percs = np.percentile(summed_state.sigwind_dists,q=np.arange(5,100,5))
        hail_percs = np.percentile(summed_state.hail_dists,q=np.arange(5,100,5))
        sighail_percs = np.percentile(summed_state.sighail_dists,q=np.arange(5,100,5))

        state_percs = np.concatenate([wind_percs,sigwind_percs,hail_percs,sighail_percs])
        all_percs = np.vstack([all_percs,state_percs])

        loc_list.append(affected)

    return all_percs, loc_list, otlkdt_formatted


for cov_outlook in cov_outlooks[:1]:
    all_percs, loc_list, otlkdt_formatted = generate_features_preds(cov_outlook)

    df_out = pd.DataFrame(all_percs)
    percentiles = list(range(5, 100, 5))
    labels = (
        [f'wind_perc_{i}' for i in percentiles] +
        [f'sigwind_perc_{i}' for i in percentiles] +
        [f'hail_perc_{i}' for i in percentiles] +
        [f'sighail_perc_{i}' for i in percentiles]
    )
    df_out.columns = labels
    df_out['location'] = loc_list

    df_out.to_csv(f'../prediction-percs-corr/{otlkdt_formatted}_pred-percs.csv',index=False)





