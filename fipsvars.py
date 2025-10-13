from scipy import stats
import numpy as np

nsims = 10000

fipsToState = {
    0:"0",
    2:"AK",
    1:"AL",
    5:"AR",
    60:"AS",
    4:"AZ",
    6:"CA",
    8:"CO",
    9:"CT",
    11:"DC",
    10:"DE",
    12:"FL",
    13:"GA",
    66:"GU",
    15:"HI",
    19:"IA",
    16:"ID",
    17:"IL",
    18:"IN",
    20:"KS",
    21:"KY",
    22:"LA",
    25:"MA",
    24:"MD",
    23:"ME",
    26:"MI",
    27:"MN",
    29:"MO",
    28:"MS",
    30:"MT",
    37:"NC",    
    38:"ND",
    31:"NE",
    33:"NH",
    34:"NJ",
    35:"NM",
    32:"NV",
    36:"NY",
    39:"OH",
    40:"OK",
    41:"OR",
    42:"PA",
    72:"PR",
    44:"RI",
    45:"SC",
    46:"SD",
    47:"TN",
    48:"TX",
    49:"UT",
    51:"VA",
    78:"VI",
    50:"VT",
    53:"WA",
    55:"WI",
    54:"WV",
    56:"WY"
}

col_names = [
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
        'popwinddensity','maxct','medct',
        'maxhourct','medhourct',
        'hourofmax_sin','hourofmax_cos',
        'sumct','sumhourct','sumpopct','sumhourpopct',
        'maxpopct','medpopct','maxhourpopct','medhourpopct'
]

col_names_wind = ['maxwind', 'medwind', 'minwind', 'areawind', 'area5wind', 'area15wind',
       'area30wind', 'area45wind', 'area60wind', 'areacigwind', 'areacig0wind',
       'areacig1wind', 'areacig2wind', 'popwind', 'pop5wind', 'pop15wind',
       'pop30wind', 'pop45wind', 'pop60wind', 'popcig0wind', 'popcig1wind',
       'popcig2wind', 'popwinddensity', 'wfo', 'doy', 'maxct', 'medct',
       'maxhourct', 'medhourct', 'hourofmax_sin', 'hourofmax_cos', 'sumct',
       'sumhourct', 'sumpopct', 'sumhourpopct', 'maxpopct', 'medpopct',
       'maxhourpopct', 'medhourpopct']

# 0 non-sig, 1 sig
mags = [0,1]
levs = [0,1,2,3]

cig_dists = {
    'wind': 
    {
        '0': [0.94,0.06],
        '1': [0.87,0.13],
        '2': [0.77,0.23],
        '3': [0.66,0.34]
    },
    'hail': 
    {
        '0': [0.92,0.08],
        '1': [0.83,0.17],
        '2': [0.75,0.25],
        # dummy
        '3': [0.65,0.35]
    }
}

def find_dist(val,haz,size):

    val = int(val)

    # Use empirical dists for all 0 predictions, regardless of hazard or cwa-st size
    if val == 0:
        # EMPIRICAL DIST
        return np.random.choice(dists_final[haz][size][str(val)][0], nsims,
                            replace=True, p=dists_final[haz][size][str(val)][1])
    else:
        if haz == 'hail':
            if size == 'small':
                # EMPIRICAL DIST
                return np.random.choice(dists_final[haz][size]['>0'][0], nsims,
                                    replace=True, p=dists_final[haz][size]['>0'][1])
            elif size == 'medium':
                # EMPIRICAL DIST
                if val < 6:
                    return np.random.choice(dists_final[haz][size][str(val)][0], nsims,
                                        replace=True, p=dists_final[haz][size][str(val)][1])
                else:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['>5']
            else:
                if val < 6:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size][str(val)][0], nsims,
                                        replace=True, p=dists_final[haz][size][str(val)][1])
                elif val < 10:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size]['6-9'][0], nsims,
                                        replace=True, p=dists_final[haz][size]['6-9'][1])
                elif val < 20:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size]['10-19'][0], nsims,
                                        replace=True, p=dists_final[haz][size]['10-19'][1])
                elif val < 30:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['20-29']
                else:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['>29']                         
        elif haz == 'wind':
            if size == 'small':
                if val < 3:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size][str(val)][0], nsims,
                                        replace=True, p=dists_final[haz][size][str(val)][1])
                else:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size]['>2'][0], nsims,
                                        replace=True, p=dists_final[haz][size]['>2'][1])
            elif size == 'medium':
                if val < 6:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size][str(val)][0], nsims,
                                        replace=True, p=dists_final[haz][size][str(val)][1])
                elif val < 10:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size]['6-9'][0], nsims,
                                        replace=True, p=dists_final[haz][size]['6-9'][1])
                elif val < 15:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['10-14']
                else:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['>14']
            else:
                if val < 6:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size][str(val)][0], nsims,
                                        replace=True, p=dists_final[haz][size][str(val)][1])
                elif val < 10:
                    # EMPIRICAL DIST
                    return np.random.choice(dists_final[haz][size]['6-9'][0], nsims,
                                        replace=True, p=dists_final[haz][size]['6-9'][1])    
                elif val < 15:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['10-14']
                elif val < 20:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['15-19']  
                elif val < 30:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['20-29']
                else:
                    # PARAMETRIC DIST
                    dist2use = dists_final[haz][size]['>29']
                
    # return np.round(dist2use.rvs(nsims))
    return dist2use.rvs(nsims).astype(int)

dists_final = {
    'hail': {
        'small': {
            '0': [np.arange(0,-5,-1),np.array([0.982,0.014,0.002,0.001,0.001])],
            '>0': [np.arange(-5,4,1),np.array([0.008, 0.015, 0.023, 0.031, 0.108, 0.231, 0.538, 0.023, 0.023])]
        },
        'medium': {
            '0':[np.arange(-11,1,1),np.array([3.000e-04, 2.000e-04, 2.000e-04, 2.000e-04, 2.000e-04, 5.000e-04,
                                                1.000e-03, 5.000e-04, 1.900e-03, 9.500e-03, 3.080e-02, 9.547e-01])],
            '1': [np.arange(-7,2,1),np.array([0.0015, 0.0015, 0.0044, 0.0044, 0.019 , 0.0439, 0.0936, 0.1404,
                                                0.6913])],   
            '2':[np.arange(-9,3,1),np.array([0.0129, 0.0065, 0.0129, 0.0065, 0.0065, 0.0065, 0.0323, 0.0903,
                                                0.129 , 0.1226, 0.174, 0.4   ])],
            '3': [np.arange(-7,4,1),np.array([0.0189, 0.0189, 0.0189, 0.0189, 0.0943, 0.0755, 0.2075, 0.1132,
                                                0.1132, 0.1132, 0.2075])],
            '4': [np.arange(-3,5,1),np.array([0.1667, 0.1333, 0.0667, 0.2333, 0.1   , 0.1   , 0.0667, 0.1333])],
            '5': [np.arange(-4,6,1),np.array([0.06, 0.176, 0.176, 0.123, 0.059, 0.059, 0.059, 0.059, 0.059,
                                                0.1700000000000001])],
            '>5': stats.gamma(a=7011.31, loc=-288.324, scale=0.040851)
        },
        'big': {
            '0': [np.arange(-6,1,1),np.array([0.001, 0.001, 0.001, 0.005, 0.01, 0.042, 0.94])],
            '1': [np.arange(-14,2,1),np.array([6.000e-04, 6.000e-04, 1.200e-03, 2.300e-03, 1.700e-03, 1.200e-03,
                                                1.200e-03, 1.700e-03, 1.200e-03, 5.200e-03, 5.800e-03, 2.300e-02,
                                                4.430e-02, 7.710e-02, 1.605e-01, 6.724e-01])],
            '2': [np.arange(-11,3,1),np.array([0.0017, 0.0035, 0.0052, 0.0069, 0.0121, 0.0087, 0.0087, 0.0104,
                                                0.0346, 0.0588, 0.1228, 0.1453, 0.1453, 0.436 ])],
            '3': [np.arange(-6,4,1),np.array([0.0073, 0.0219, 0.0365, 0.062 , 0.0949, 0.1423, 0.1496, 0.0912,
                                                0.1022, 0.2921])],
            '4': [np.arange(-8,5,1),np.array([0.0055, 0.0055, 0.0055, 0.0109, 0.0273, 0.0656, 0.1257, 0.1639,
                                                0.1366, 0.0984, 0.0546, 0.1148, 0.1857])],
            '5': [np.arange(-9,6,1),np.array([0.0089, 0.0089, 0.0179, 0.0179, 0.0357, 0.0357, 0.0714, 0.125 ,
                                                0.1786, 0.125 , 0.0536, 0.0446, 0.0625, 0.0625, 0.1518])],
            '6-9': [np.arange(-14,10,1),np.array([0.0039, 0.0039, 0.0039, 0.0039, 0.0039, 0.0079, 0.0118, 0.0118,
                                                0.0276, 0.0512, 0.0472, 0.0748, 0.1102, 0.1417, 0.122 , 0.0945,
                                                0.0551, 0.0354, 0.0354, 0.0157, 0.0787, 0.0354, 0.0197, 0.0044])],
            '10-19': [np.arange(-13,15,1),np.array([0.0056, 0.0056, 0.0112, 0.0112, 0.0168, 0.0279, 0.0335, 0.0447,
                                                0.0615, 0.0726, 0.0838, 0.0894, 0.0894, 0.0838, 0.0782, 0.0615,
                                                0.0503, 0.0391, 0.0279, 0.0223, 0.0168, 0.0168, 0.0112, 0.0112,
                                                0.0112, 0.0056, 0.0056, 0.0053])],
            '20-29': stats.t(df=6.55909, loc=-2.2212, scale=7.04736),
            '>29': stats.t(df=1.8691, loc=-4.47263, scale=4.64306)
        }
    },
    'wind': {
        'small': {
            '0': [np.arange(-4,1,1),np.array([0.001, 0.002, 0.005, 0.028, 0.964])],
            '1': [np.arange(-6,2,1),np.array([0.002, 0.005, 0.012, 0.019, 0.031, 0.065, 0.118, 0.748])],
            '2': [np.arange(-4,3,1),np.array([0.086, 0.057, 0.029, 0.114, 0.129, 0.243, 0.342])],
            '>2': [np.arange(-8,6,1),np.array([0.039, 0.039, 0.039, 0.059, 0.02 , 0.059, 0.098, 0.078, 0.118,
                                                0.098, 0.137, 0.157, 0.02 , 0.039])]
        },
        'medium': {
            '0': [np.arange(-7,1,1),np.array([0.001, 0.002, 0.002, 0.002, 0.008, 0.012, 0.044, 0.929])],
            '1': [np.arange(-14,2,1),np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.002, 0.006,
                                                0.005, 0.011, 0.024, 0.039, 0.071, 0.117, 0.717])],
            '2': [np.arange(-14,3,1),np.array([0.002, 0.002, 0.002, 0.002, 0.004, 0.004, 0.009, 0.017, 0.004,
                                                0.024, 0.015, 0.048, 0.039, 0.115, 0.12 , 0.115, 
                                                0.478])],
            '3': [np.arange(-12,4,1),np.array([0.004, 0.005, 0.005, 0.005, 0.01 , 0.005, 0.019, 0.029, 0.029,
                                                0.063, 0.082, 0.13 , 0.106, 0.092, 0.116, 0.3  ])],
            '4': [np.arange(-6,5,1),np.array([0.008, 0.031, 0.038, 0.038, 0.092, 0.138, 0.1  , 0.094, 0.131,
                                                0.092, 0.238])],
            '5': [np.arange(-9,6,1),np.array([0.0135, 0.0135, 0.0405, 0.0135, 0.0676, 0.0676, 0.0676, 0.0676,
                                                0.1351, 0.0946, 0.0676, 0.0405, 0.0811, 0.0405, 0.1892])],
            '6-9': [np.arange(-12,10,1),np.array([0.007, 0.021, 0.021, 0.007, 0.041, 0.034, 0.048, 0.062, 0.048,
                                                0.055, 0.062, 0.089, 0.082, 0.082, 0.048, 0.034, 0.014, 0.048,
                                                0.11 , 0.041, 0.034, 0.012])],
            '10-14': stats.t(df=1.83624e+09, loc=-1.44445, scale=6.59944),
            '>14': stats.t(df=5.9964, loc=-2.8666, scale=7.55634)
        },
        'big': {
            '0': [np.arange(-9,1,1),np.array([0.   , 0.   , 0.001, 0.   , 0.002, 0.002, 0.007, 0.014, 0.042,
                                                0.932])],
            '1': [np.arange(-9,2,1),np.array([0.001, 0.001, 0.002, 0.002, 0.003, 0.007, 0.018, 0.03 , 0.071,
                                                0.145, 0.72 ])],
            '2': [np.arange(-9,3,1),np.array([0.003, 0.002, 0.001, 0.007, 0.014, 0.019, 0.041, 0.048, 0.094,
                                                0.112, 0.192, 0.467])],
            '3': [np.arange(-10,4,1), np.array([0.002, 0.002, 0.003, 0.008, 0.015, 0.009, 0.024, 0.038, 0.054,
                                                0.113, 0.122, 0.139, 0.148, 0.323])],
            '4': [np.arange(-11,5,1),np.array([0.003, 0.003, 0.003, 0.003, 0.003, 0.01 , 0.02 , 0.041, 0.044,
                                                0.061, 0.098, 0.125, 0.118, 0.132,
                                                0.135, 0.201])],
            '5': [np.arange(-9,6,1),np.array([0.0043, 0.0043, 0.0085, 0.0128, 0.0214, 0.0385, 0.0598, 0.0684,
                                                0.094 , 0.1111, 0.0855, 0.0641, 
                                                0.0769, 0.1282, 0.2222])],
            '6-9': [np.arange(-16,10,1), np.array([0.002, 0.003, 0.004, 0.005, 0.014, 0.011, 0.005, 0.018, 0.009,
                                                0.014, 0.025, 0.032, 0.039, 0.062, 0.078, 0.092, 0.101, 0.087,
                                                0.078, 0.055, 0.048, 0.062, 0.085, 0.034, 0.023, 0.014])],
            '10-14': stats.t(df=3.40037, loc=-0.654843, scale=4.63763),
            '15-19': stats.t(df=2.49147, loc=-0.732559, scale=6.46662),
            '20-29': stats.t(df=1.7052, loc=-1.86668, scale=4.81388),
            '>29': stats.t(df=1.4699, loc=-3.46713, scale=6.00389)
        }
    },
}

#### CWA-ST size categorized

small_cwast = ['ABQAZ',
 'ABRMN',
 'ABRND',
 'AKQMD',
 'ALYCT',
 'ALYMA',
 'ALYVT',
 'AMANM',
 'ARXIL',
 'BISMT',
 'BISSD',
 'BMXMS',
 'BOINV',
 'BOXNH',
 'BOXRI',
 'BTVNH',
 'BUFPA',
 'CLEPA',
 'CTPMD',
 'CTPWV',
 'CYSCO',
 'DDCCO',
 'DDCOK',
 'DMXMO',
 'DTXOH',
 'DVNMO',
 'EAXNE',
 'EPZAZ',
 'FFCAL',
 'FGFSD',
 'FWDOK',
 'GJTAZ',
 'GJTNM',
 'GLDNE',
 'GSPTN',
 'GYXMA',
 'GYXVT',
 'HUNMS',
 'HUNTN',
 'ICTOK',
 'INDIL',
 'JANAR',
 'JKLTN',
 'JKLVA',
 'KEYFL',
 'LBFCO',
 'LMKTN',
 'LUBNM',
 'LWXDC',
 'LZKMS',
 'LZKOK',
 'MEGAL',
 'MEGMO',
 'MFRNV',
 'MKXIL',
 'MPXIA',
 'MPXSD',
 'MQTWI',
 'MRXAL',
 'MRXGA',
 'MRXKY',
 'MRXNC',
 'MSOOR',
 'OAXKS',
 'OAXMO',
 'OHXAL',
 'OHXKY',
 'OKXNJ',
 'OKXPA',
 'OTXOR',
 'PAHAR',
 'PAHIN',
 'PAHTN',
 'PBZMD',
 'PHIDE',
 'PIHMT',
 'PIHNV',
 'PIHUT',
 'PUBNM',
 'PUBOK',
 'RAHSC',
 'RIWCO',
 'RIWID',
 'RIWMT',
 'RIWUT',
 'RLXKY',
 'RLXVA',
 'RNKTN',
 'SGFAR',
 'SGFOK',
 'SHVOK',
 'SLCAZ',
 'SLCNV',
 'SLCWY',
 'TFXID',
 'TSATX',
 'UNRMT',
 'UNRNE']

med_cwast = ['AKQNC',
 'AMAOK',
 'ARXIA',
 'ARXMN',
 'BGMPA',
 'BOXCT',
 'BOXMA',
 'BROTX',
 'BTVNY',
 'BTVVT',
 'BYZWY',
 'CAEGA',
 'CHSGA',
 'CHSSC',
 'CYSNE',
 'DVNIL',
 'EAXKS',
 'EPZTX',
 'FSDIA',
 'FSDMN',
 'FSDNE',
 'GIDKS',
 'GLDCO',
 'GSPGA',
 'GSPSC',
 'HUNAL',
 'ILMNC',
 'ILMSC',
 'ILNIN',
 'ILNKY',
 'IWXMI',
 'IWXOH',
 'JANLA',
 'JAXGA',
 'LCHTX',
 'LIXMS',
 'LMKIN',
 'LOTIN',
 'LSXIL',
 'LWXMD',
 'LWXWV',
 'MAFNM',
 'MEGAR',
 'MFLFL',
 'MHXNC',
 'MLBFL',
 'MOBFL',
 'MOBMS',
 'MPXWI',
 'MRXVA',
 'OAXIA',
 'OKXCT',
 'OKXNY',
 'OUNTX',
 'PAHIL',
 'PAHKY',
 'PAHMO',
 'PBZOH',
 'PBZWV',
 'PHIMD',
 'PHINJ',
 'PHIPA',
 'PQRWA',
 'PSRCA',
 'RLXOH',
 'RNKNC',
 'RNKWV',
 'SGFKS',
 'SHVAR',
 'TAEAL',
 'TAEGA',
 'TSAAR']

big_cwast = ['ABQNM',
 'ABRSD',
 'AKQVA',
 'ALYNY',
 'AMATX',
 'APXMI',
 'ARXWI',
 'BGMNY',
 'BISND',
 'BMXAL',
 'BOIID',
 'BOIOR',
 'BOUCO',
 'BUFNY',
 'BYZMT',
 'CAESC',
 'CARME',
 'CLEOH',
 'CRPTX',
 'CTPPA',
 'CYSWY',
 'DDCKS',
 'DLHMN',
 'DLHWI',
 'DMXIA',
 'DTXMI',
 'DVNIA',
 'EAXMO',
 'EKACA',
 'EPZNM',
 'EWXTX',
 'FFCGA',
 'FGFMN',
 'FGFND',
 'FGZAZ',
 'FSDSD',
 'FWDTX',
 'GGWMT',
 'GIDNE',
 'GJTCO',
 'GJTUT',
 'GLDKS',
 'GRBWI',
 'GRRMI',
 'GSPNC',
 'GYXME',
 'GYXNH',
 'HGXTX',
 'HNXCA',
 'ICTKS',
 'ILNOH',
 'ILXIL',
 'INDIN',
 'IWXIN',
 'JANMS',
 'JAXFL',
 'JKLKY',
 'LBFNE',
 'LCHLA',
 'LIXLA',
 'LKNNV',
 'LMKKY',
 'LOTIL',
 'LOXCA',
 'LSXMO',
 'LUBTX',
 'LWXVA',
 'LZKAR',
 'MAFTX',
 'MEGMS',
 'MEGTN',
 'MFRCA',
 'MFROR',
 'MKXWI',
 'MOBAL',
 'MPXMN',
 'MQTMI',
 'MRXTN',
 'MSOID',
 'MSOMT',
 'MTRCA',
 'OAXNE',
 'OHXTN',
 'OTXID',
 'OTXWA',
 'OUNOK',
 'PBZPA',
 'PDTOR',
 'PDTWA',
 'PIHID',
 'PQROR',
 'PSRAZ',
 'PUBCO',
 'RAHNC',
 'REVCA',
 'REVNV',
 'RIWWY',
 'RLXWV',
 'RNKVA',
 'SEWWA',
 'SGFMO',
 'SGXCA',
 'SHVLA',
 'SHVTX',
 'SJTTX',
 'SLCUT',
 'STOCA',
 'TAEFL',
 'TBWFL',
 'TFXMT',
 'TOPKS',
 'TSAOK',
 'TWCAZ',
 'UNRSD',
 'UNRWY',
 'VEFAZ',
 'VEFCA',
 'VEFNV']