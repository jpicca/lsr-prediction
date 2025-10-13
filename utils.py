import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal, norm
import math
import pickle
from fipsvars import big_cwast, med_cwast, small_cwast, find_dist

from pdb import set_trace as st

with open('../../ml-data/trained-models/wfo-label-encoder.model','rb') as f:
    wfo_label_encoder = pickle.load(f)

with open('../../ml-data/trained-models/det/hgb-det_wind-simple.model','rb') as f:
    wind_model = pickle.load(f)

with open('../../ml-data/trained-models/det/hgb-det_hail-simple.model','rb') as f:
    hail_model = pickle.load(f)

with open('../../ml-data/trained-models/det/hgb-det_sigwind-nopopfeat-withwindcount.model','rb') as f:
    sigwind_model = pickle.load(f)

with open('../../ml-data/trained-models/det/hgb-det_sighail-nopopfeat-withhailcount.model','rb') as f:
    sighail_model = pickle.load(f)


models = {
    'le': wfo_label_encoder,
    'wind': wind_model,
    'hail': hail_model,
    'sigwind': sigwind_model,
    'sighail': sighail_model
}

col_names_outlook = [
    'otlk_timestamp','wfo_st_list',
        'toy_sin','toy_cos','maxhaz','medhaz','minhaz','popwfo','maxtor','medtor','mintor',
        'areator', 'area2tor', 'area5tor', 'area10tor', 'area15tor', 'area30tor', 'area45tor', 'area60tor',
        'areacigtor', 'areacig0tor', 'areacig1tor', 'areacig2tor',
        'poptor', 'pop2tor', 'pop5tor', 'pop10tor', 'pop15tor', 'pop30tor', 'pop45tor', 'pop60tor',
        'popcig0tor', 'popcig1tor', 'popcig2tor',
        'poptordensity',
        'maxhail','medhail','minhail',
        'areahail', 'area5hail', 'area15hail', 'area30hail', 'area45hail', 'area60hail',
        'areacighail', 'areacig0hail', 'areacig1hail', 'areacig2hail',
        'pophail', 'pop5hail', 'pop15hail', 'pop30hail', 'pop45hail', 'pop60hail',
        'popcig0hail', 'popcig1hail', 'popcig2hail',
        'pophaildensity',
        'maxwind','medwind','minwind',
        'areawind', 'area5wind', 'area15wind', 'area30wind', 'area45wind', 'area60wind',
        'areacigwind', 'areacig0wind', 'areacig1wind', 'areacig2wind',
        'popwind', 'pop5wind', 'pop15wind', 'pop30wind', 'pop45wind', 'pop60wind',
        'popcig0wind', 'popcig1wind', 'popcig2wind',
        'popwinddensity'
]

col_names_wind = ['maxwind', 'medwind', 'minwind', 'areawind', 'area5wind', 'area15wind',
       'area30wind', 'area45wind', 'area60wind', 'areacigwind', 'areacig0wind',
       'areacig1wind', 'areacig2wind', 'popwind', 'pop5wind', 'pop15wind',
       'pop30wind', 'pop45wind', 'pop60wind', 'popcig0wind', 'popcig1wind',
       'popcig2wind', 'popwinddensity', 'wfo', 'doy', 'maxct', 'medct',
       'maxhourct', 'medhourct', 'hourofmax_sin', 'hourofmax_cos', 'sumct',
       'sumhourct', 'sumpopct', 'sumhourpopct', 'maxpopct', 'medpopct',
       'maxhourpopct', 'medhourpopct']

col_names_hail = ['maxhail', 'medhail', 'minhail', 'areahail', 'area5hail', 'area15hail',
       'area30hail', 'area45hail', 'area60hail', 'areacighail', 'areacig0hail',
       'areacig1hail', 'areacig2hail', 'pophail', 'pop5hail', 'pop15hail',
       'pop30hail', 'pop45hail', 'pop60hail', 'popcig0hail', 'popcig1hail',
       'popcig2hail', 'pophaildensity', 'wfo', 'doy', 'maxct', 'medct',
       'maxhourct', 'medhourct', 'hourofmax_sin', 'hourofmax_cos', 'sumct',
       'sumhourct', 'sumpopct', 'sumhourpopct', 'maxpopct', 'medpopct',
       'maxhourpopct', 'medhourpopct']

col_names_href = [
        'otlk_timestamp','wfo_st_list',
        'maxct','medct',
        'maxhourct','medhourct',
        'hourofmax_sin','hourofmax_cos',
        'sumct','sumhourct','sumpopct','sumhourpopct',
        'maxpopct','medpopct','maxhourpopct','medhourpopct'
]

def fix_neg(arr):
    """ Fix negative values in an array by replacing them with 0"""
    arr[arr <= 0] = 0
    return arr

def flatten_list(_list):
    return np.array([item for sublist in _list for item in sublist])

def gather_features(wfo_st_unique, wfo_state_2d, bdt, 
                    wind_cov, hail_cov, torn_cov, 
                    wind_cig, hail_cig, torn_cig, 
                    population):
    
    # features
    # all
    toy_sin,toy_cos = [],[]
    otlk_timestamp = []
    wfo_st_list = []
    maxhaz,medhaz,minhaz = [],[],[]
    popwfo = []
    
    # tor
    maxtor,medtor,mintor = [],[],[]
    areator, area2tor, area5tor, area10tor, area15tor, area30tor, area45tor, area60tor = [],[],[],[],[],[],[],[]
    areacigtor, areacig0tor, areacig1tor, areacig2tor = [],[],[],[]
    poptor, pop2tor, pop5tor, pop10tor, pop15tor, pop30tor, pop45tor, pop60tor = [],[],[],[],[],[],[],[]
    popcig0tor, popcig1tor, popcig2tor = [],[],[]
    poptordensity = []
    
    # hail
    maxhail,medhail,minhail = [],[],[]
    areahail, area5hail, area15hail, area30hail, area45hail, area60hail = [],[],[],[],[],[]
    areacighail, areacig0hail, areacig1hail, areacig2hail = [],[],[],[]
    pophail, pop5hail, pop15hail, pop30hail, pop45hail, pop60hail = [],[],[],[],[],[]
    popcig0hail, popcig1hail, popcig2hail = [],[],[]
    pophaildensity = []
    
    # wind
    maxwind,medwind,minwind = [],[],[]
    areawind, area5wind, area15wind, area30wind, area45wind, area60wind = [],[],[],[],[],[]
    areacigwind, areacig0wind, areacig1wind, areacig2wind = [],[],[],[]
    popwind, pop5wind, pop15wind, pop30wind, pop45wind, pop60wind = [],[],[],[],[],[]
    popcig0wind, popcig1wind, popcig2wind = [],[],[]
    popwinddensity = []
    
    
    for wfo_st in wfo_st_unique:
        truth_array = wfo_st == wfo_state_2d
        
        # features
        # all
        toy_sin.append(math.sin(2*math.pi*bdt.timetuple().tm_yday/366))
        toy_cos.append(math.cos(2*math.pi*bdt.timetuple().tm_yday/366))
        otlk_timestamp.append(bdt.strftime('%Y%m%d%H'))
        wfo_st_list.append(wfo_st)
        maxhaz.append(np.max([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        minhaz.append(np.min([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        medhaz.append(np.median([wind_cov[truth_array],torn_cov[truth_array],hail_cov[truth_array]]))
        popwfo.append(population[truth_array].sum())
        
    
        # wind
        maxwind.append(np.max(wind_cov[truth_array]))
        minwind.append(np.min(wind_cov[truth_array]))
        medwind.append(np.median(wind_cov[truth_array]))
        
        areawind.append(np.sum(wind_cov[truth_array] > 0))
        area5wind.append(np.sum(wind_cov[truth_array] == 0.05))
        area15wind.append(np.sum(wind_cov[truth_array] == 0.15))
        area30wind.append(np.sum(wind_cov[truth_array] == 0.30))
        area45wind.append(np.sum(wind_cov[truth_array] == 0.45))
        area60wind.append(np.sum(wind_cov[truth_array] == 0.60))
    
        areacigwind.append(np.sum(wind_cig[truth_array] > 0))
        areacig0wind.append(np.sum(np.logical_and(wind_cig[truth_array] == 0,wind_cov[truth_array] > 0)))
        areacig1wind.append(np.sum(wind_cig[truth_array] == 1))
        areacig2wind.append(np.sum(wind_cig[truth_array] == 2))
    
        popwind.append(np.sum((wind_cov[truth_array] > 0)*population[truth_array]))
        pop5wind.append(np.sum((wind_cov[truth_array] == 0.05)*population[truth_array]))
        pop15wind.append(np.sum((wind_cov[truth_array] == 0.15)*population[truth_array]))
        pop30wind.append(np.sum((wind_cov[truth_array] == 0.30)*population[truth_array]))
        pop45wind.append(np.sum((wind_cov[truth_array] == 0.45)*population[truth_array]))
        pop60wind.append(np.sum((wind_cov[truth_array] == 0.60)*population[truth_array]))
    
        popcig0wind.append(np.sum((np.logical_and(wind_cig[truth_array] == 0,wind_cov[truth_array] > 0))*population[truth_array]))
        popcig1wind.append(np.sum((wind_cig[truth_array] == 1)*population[truth_array]))
        popcig2wind.append(np.sum((wind_cig[truth_array] == 2)*population[truth_array]))
    
        popwinddensity.append(np.sum((wind_cov[truth_array] > 0)*population[truth_array]) / np.sum(wind_cov[truth_array] > 0))
        
    
        # hail
        maxhail.append(np.max(hail_cov[truth_array]))
        minhail.append(np.min(hail_cov[truth_array]))
        medhail.append(np.median(hail_cov[truth_array]))
        
        areahail.append(np.sum(hail_cov[truth_array] > 0))
        area5hail.append(np.sum(hail_cov[truth_array] == 0.05))
        area15hail.append(np.sum(hail_cov[truth_array] == 0.15))
        area30hail.append(np.sum(hail_cov[truth_array] == 0.30))
        area45hail.append(np.sum(hail_cov[truth_array] == 0.45))
        area60hail.append(np.sum(hail_cov[truth_array] == 0.60))
    
        areacighail.append(np.sum(hail_cig[truth_array] > 0))
        areacig0hail.append(np.sum(np.logical_and(hail_cig[truth_array] == 0,hail_cov[truth_array] > 0)))
        areacig1hail.append(np.sum(hail_cig[truth_array] == 1))
        areacig2hail.append(np.sum(hail_cig[truth_array] == 2))
    
        pophail.append(np.sum((hail_cov[truth_array] > 0)*population[truth_array]))
        pop5hail.append(np.sum((hail_cov[truth_array] == 0.05)*population[truth_array]))
        pop15hail.append(np.sum((hail_cov[truth_array] == 0.15)*population[truth_array]))
        pop30hail.append(np.sum((hail_cov[truth_array] == 0.30)*population[truth_array]))
        pop45hail.append(np.sum((hail_cov[truth_array] == 0.45)*population[truth_array]))
        pop60hail.append(np.sum((hail_cov[truth_array] == 0.60)*population[truth_array]))
    
        popcig0hail.append(np.sum((np.logical_and(hail_cig[truth_array] == 0,hail_cov[truth_array] > 0))*population[truth_array]))
        popcig1hail.append(np.sum((hail_cig[truth_array] == 1)*population[truth_array]))
        popcig2hail.append(np.sum((hail_cig[truth_array] == 2)*population[truth_array]))
    
        pophaildensity.append(np.sum((hail_cov[truth_array] > 0)*population[truth_array]) / np.sum(hail_cov[truth_array] > 0))
        
        # tor
        maxtor.append(np.max(torn_cov[truth_array]))
        mintor.append(np.min(torn_cov[truth_array]))
        medtor.append(np.median(torn_cov[truth_array]))
        
        areator.append(np.sum(torn_cov[truth_array] > 0))
        area2tor.append(np.sum(torn_cov[truth_array] == 0.02))
        area5tor.append(np.sum(torn_cov[truth_array] == 0.05))
        area10tor.append(np.sum(torn_cov[truth_array] == 0.10))
        area15tor.append(np.sum(torn_cov[truth_array] == 0.15))
        area30tor.append(np.sum(torn_cov[truth_array] == 0.30))
        area45tor.append(np.sum(torn_cov[truth_array] == 0.45))
        area60tor.append(np.sum(torn_cov[truth_array] == 0.60))
    
        areacigtor.append(np.sum(torn_cig[truth_array] > 0))
        areacig0tor.append(np.sum(np.logical_and(torn_cig[truth_array] == 0,torn_cov[truth_array] > 0)))
        areacig1tor.append(np.sum(torn_cig[truth_array] == 1))
        areacig2tor.append(np.sum(torn_cig[truth_array] == 2))
    
        poptor.append(np.sum((torn_cov[truth_array] > 0)*population[truth_array]))
        pop2tor.append(np.sum((torn_cov[truth_array] == 0.02)*population[truth_array]))
        pop5tor.append(np.sum((torn_cov[truth_array] == 0.05)*population[truth_array]))
        pop10tor.append(np.sum((torn_cov[truth_array] == 0.10)*population[truth_array]))
        pop15tor.append(np.sum((torn_cov[truth_array] == 0.15)*population[truth_array]))
        pop30tor.append(np.sum((torn_cov[truth_array] == 0.30)*population[truth_array]))
        pop45tor.append(np.sum((torn_cov[truth_array] == 0.45)*population[truth_array]))
        pop60tor.append(np.sum((torn_cov[truth_array] == 0.60)*population[truth_array]))
    
        popcig0tor.append(np.sum((np.logical_and(torn_cig[truth_array] == 0,torn_cov[truth_array] > 0))*population[truth_array]))
        popcig1tor.append(np.sum((torn_cig[truth_array] == 1)*population[truth_array]))
        popcig2tor.append(np.sum((torn_cig[truth_array] == 2)*population[truth_array]))
    
        poptordensity.append(np.sum((torn_cov[truth_array] > 0)*population[truth_array]) / np.sum(torn_cov[truth_array] > 0))

    df_features = pd.DataFrame([
        otlk_timestamp,wfo_st_list,
        toy_sin,toy_cos,maxhaz,medhaz,minhaz,popwfo,maxtor,medtor,mintor,
        areator, area2tor, area5tor, area10tor, area15tor, area30tor, area45tor, area60tor,
        areacigtor, areacig0tor, areacig1tor, areacig2tor,
        poptor, pop2tor, pop5tor, pop10tor, pop15tor, pop30tor, pop45tor, pop60tor,
        popcig0tor, popcig1tor, popcig2tor,
        poptordensity,
        maxhail,medhail,minhail,
        areahail, area5hail, area15hail, area30hail, area45hail, area60hail,
        areacighail, areacig0hail, areacig1hail, areacig2hail,
        pophail, pop5hail, pop15hail, pop30hail, pop45hail, pop60hail,
        popcig0hail, popcig1hail, popcig2hail,
        pophaildensity,
        maxwind,medwind,minwind,
        areawind, area5wind, area15wind, area30wind, area45wind, area60wind,
        areacigwind, areacig0wind, areacig1wind, areacig2wind,
        popwind, pop5wind, pop15wind, pop30wind, pop45wind, pop60wind,
        popcig0wind, popcig1wind, popcig2wind,
        popwinddensity
    ]).T

    return df_features

def gather_ct_features(wfo_st_unique, wfo_st_1d, bdt, ct_vals, population, wfo_state_2d):
    """ Gather features for HREFCT probabilities """
    
    # features
    # all
    otlk_timestamp = []
    wfo_st_list = []
    
    # wind
    maxct,medct = [],[]
    maxhourct,medhourct = [],[]
    hourofmax_sin,hourofmax_cos = [],[]
    sumct = []
    sumhourct = []
    maxpopct,medpopct = [],[]
    sumpopct = []
    sumhourpopct = []
    maxhourpopct,medhourpopct = [],[]

    # get all HREFCT probabilities for all times, wfo-st by wfo-st
    for wfo_st in wfo_st_unique:
        wfo_st_probs = ct_vals[wfo_st == wfo_st_1d]

        # features
        # all
        otlk_timestamp.append(bdt.strftime('%Y%m%d%H'))
        wfo_st_list.append(wfo_st)
        
        maxct.append(np.max(wfo_st_probs))
        medct.append(np.median(wfo_st_probs))
        
        maxhourct.append(np.mean(wfo_st_probs,axis=0).max())
        medhourct.append(np.median(np.mean(wfo_st_probs,axis=0)))

        hourofmax_sin.append(math.sin(((bdt.hour + np.argmax(np.mean(wfo_st_probs,axis=0))) % 24)/24))
        hourofmax_cos.append(math.cos(((bdt.hour + np.argmax(np.mean(wfo_st_probs,axis=0))) % 24)/24))

        sumct.append(wfo_st_probs.sum())
        sumhourct.append(np.mean(wfo_st_probs,axis=0).sum())
        sumpopct.append((wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis]).sum())
        sumhourpopct.append(np.mean(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis],axis=0).sum())

        maxpopct.append((wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis]).max())
        medpopct.append(np.median(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis]))
        maxhourpopct.append(np.mean(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis],axis=0).max())
        medhourpopct.append(np.median(np.mean(wfo_st_probs*population[wfo_st == wfo_state_2d][:, np.newaxis],axis=0)))

    df_ctfeatures = pd.DataFrame([
        otlk_timestamp,wfo_st_list,
        maxct,medct,
        maxhourct,medhourct,
        hourofmax_sin,hourofmax_cos,
        sumct,sumhourct,sumpopct,sumhourpopct,
        maxpopct,medpopct,maxhourpopct,medhourpopct
    ]).T

    return df_ctfeatures

def sample_corr_residuals(predictions, all_areas, size_categories, hazard,
                          correlation_matrix, area_order, marginal_distributions,
                          n_sims=10000):
    """ Sample correlated residuals using a given correlation matrix and marginal distributions """
    n_areas = len(predictions)

    mvn = multivariate_normal(mean=np.zeros(correlation_matrix.shape[0]), 
                              cov=correlation_matrix)
    normal_samples = mvn.rvs(size=n_sims)

    if normal_samples.ndim == 1:
        normal_samples = normal_samples.reshape(-1, 1)
    uniform_samples = norm.cdf(normal_samples)

    area_to_corr_idx = {area: i for i, area in enumerate(area_order)}

    correlated_residuals = []

    for i, area_name in enumerate(all_areas):
        if area_name in area_to_corr_idx:
            corr_idx = area_to_corr_idx[area_name]
            residual_samples = np.percentile(
                marginal_distributions[i], 
                uniform_samples[:, corr_idx] * 100
            )
        else:
            residual_samples = np.random.choice(marginal_distributions[i], size=n_sims, replace=True)

        correlated_residuals.append(residual_samples)
    
    return correlated_residuals

def add_distribution_empirical_corr(df, corr_dict, area_order):
    """ Add distribution columns to the dataframe using empirical correlation matrices """

    all_areas = df.wfost.unique()
    for hazard in ['wind', 'hail']:
        all_predictions = []
        all_sizes = []

        for area in all_areas:
            pred_value = df[df.wfost == area][hazard].values[0]
            all_predictions.append(pred_value)

            if area in big_cwast:
                all_sizes.append('big')
            elif area in med_cwast:
                all_sizes.append('medium')
            elif area in small_cwast:
                all_sizes.append('small')

        marginal_distributions = []
        for pred,size in zip(all_predictions,all_sizes):
            residual_samples = find_dist(int(pred),hazard,size)
            marginal_distributions.append(residual_samples)

        correlated_residuals = sample_corr_residuals(
            all_predictions, all_areas, all_sizes, hazard,
            corr_dict[hazard], area_order, marginal_distributions
        )

        simulated_counts_list = []

        for i, area in enumerate(all_areas):
            residuals = correlated_residuals[i]
            det_pred = all_predictions[i]
            simulated_counts = det_pred - residuals
            simulated_counts = fix_neg(simulated_counts)

            simulated_counts_list.append(simulated_counts)
        
        df.loc[:,f'{hazard}_dists'] = simulated_counts_list

        big_preds = df[df.wfost.isin(big_cwast)]
        med_preds = df[df.wfost.isin(med_cwast)]
        small_preds = df[df.wfost.isin(small_cwast)]

    return big_preds, med_preds, small_preds

def add_distribution(df):
    """ Add distribution columns to the dataframe """

    big_preds = df[df.wfost.isin(big_cwast)]
    med_preds = df[df.wfost.isin(med_cwast)]
    small_preds = df[df.wfost.isin(small_cwast)]

    for hazard in ['wind', 'hail']:
    # for hazard in ['wind', 'hail', 'sigwind', 'sighail']:
        big_dists = big_preds[hazard] - big_preds[hazard].apply(lambda x: find_dist(x,hazard,'big'))
        big_preds[f'{hazard}_dists'] = np.array(big_dists.apply(lambda x: fix_neg(x)))

        med_dists = med_preds[hazard] - med_preds[hazard].apply(lambda x: find_dist(x,hazard,'medium'))
        med_preds[f'{hazard}_dists'] = np.array(med_dists.apply(lambda x: fix_neg(x)))

        small_dists = small_preds[hazard] - small_preds[hazard].apply(lambda x: find_dist(x,hazard,'small'))
        small_preds[f'{hazard}_dists'] = np.array(small_dists.apply(lambda x: fix_neg(x)))
    
    st()

    return big_preds, med_preds, small_preds

def return_proportion(month,wfo,hazard):
    if month in [12,1,2]:
        season = 'Winter'
    elif month in [3,4,5]:
        season = 'Spring'
    elif month in [6,7,8]:
        season = 'Summer'
    elif month in [9,10,11]:
        season = 'Fall'

    if wfo in ["EKA", "LOX", "STO", "SGX", "MTR", "HNX","MFR", "PDT", "PQR","SEW", "OTX","REV", "VEF", "LKN","PSR", "TWC", "FGZ",
                "SLC","BOI", "PIH","MSO","TFX","RIW","GJT","EPZ"]:
        region = 'West'
    elif wfo in ["BOU", "PUB","CYS","BYZ", "GGW","BIS","ABR", "UNR","LBF", "GID","GLD", "DDC","AMA", "LUB", "MAF","ABQ"]:
        region = 'High Plains'
    elif wfo in ["OUN", "TSA","ICT","FWD", "SJT", "EWX"]:
        region = 'Plains'
    elif wfo in ["FSD","DLH", "MPX", "FGF","OAX","DMX", "DVN","TOP", "EAX", "SGF", "LSX","LOT", "ILX","IND", "IWX",
                "DTX", "APX", "GRR", "MQT","GRB", "ARX", "MKX"]:
        region = 'Midwest'
    elif wfo in ["BMX", "HUN", "MOB","JAN","FFC","CHS", "CAE", "GSP","JAX", "KEY", "MLB", "MFL", "TAE", "TBW","MEG", "MRX", "OHX",
                "LMK", "JKL", "PAH","LCH", "LIX", "SHV","LZK","HGX", "CRP", "BRO"]:
        region = 'Southeast'
    elif wfo in ["PHI","PBZ", "CTP","RLX","LWX", "RNK", "AKQ","MHX", "RAH", "ILM","CLE", "ILN","CAR", "GYX","BOX",
                "ALY", "BGM", "BUF", "OKX","BTV"]:
        region = 'Northeast'

    proportions = {'wind': {'West': {'Winter': 0.09523809523809523,
                    'Spring': 0.032490974729241874,
                    'Summer': 0.049143708116157855,
                    'Fall': 0.07954545454545454},
                    'High Plains': {'Winter': 0.0392156862745098,
                    'Spring': 0.160075329566855,
                    'Summer': 0.1625668449197861,
                    'Fall': 0.15789473684210525},
                    'Plains': {'Winter': 0.12017167381974249,
                    'Spring': 0.1680933852140078,
                    'Summer': 0.16076447442383363,
                    'Fall': 0.14008941877794337},
                    'Midwest': {'Winter': 0.16909620991253643,
                    'Spring': 0.09427860696517414,
                    'Summer': 0.08757637474541752,
                    'Fall': 0.0836092715231788},
                    'Southeast': {'Winter': 0.056451612903225805,
                    'Spring': 0.05987093690248566,
                    'Summer': 0.03495624425856984,
                    'Fall': 0.037642397226349676},
                    'Northeast': {'Winter': 0.013916500994035786,
                    'Spring': 0.016916780354706683,
                    'Summer': 0.019506098022877054,
                    'Fall': 0.0076481835564053535}},
                    'hail': {'West': {'Winter': 0.0,
                    'Spring': 0.0,
                    'Summer': 0.036036036036036036,
                    'Fall': 0.030534351145038167},
                    'High Plains': {'Winter': 0.06557377049180328,
                    'Spring': 0.07942583732057416,
                    'Summer': 0.10298507462686567,
                    'Fall': 0.04953560371517028},
                    'Plains': {'Winter': 0.0410958904109589,
                    'Spring': 0.07186678352322524,
                    'Summer': 0.10559006211180125,
                    'Fall': 0.07168458781362007},
                    'Midwest': {'Winter': 0.045146726862302484,
                    'Spring': 0.04868603042876902,
                    'Summer': 0.09302325581395349,
                    'Fall': 0.07397003745318352},
                    'Southeast': {'Winter': 0.06479113384484228,
                    'Spring': 0.07476212052560036,
                    'Summer': 0.05352363960749331,
                    'Fall': 0.0603448275862069},
                    'Northeast': {'Winter': 0.09090909090909091,
                    'Spring': 0.03322995126273815,
                    'Summer': 0.04234769687964339,
                    'Fall': 0.010869565217391304}}}
    
    return proportions[hazard][region][season]
