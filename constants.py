import os

ZECMIP_LOCAL_DIR = '/g/data/w40/ab2313/zecmip_stabilisation'
MODULE_DIR = '/home/563/ab2313/Documents/zecmip_stabilisation/src'
ZECMIP_STABILISATION_DIR = '/g/data/w40/ab2313/zecmip_stabilisation'
ZECMIP_MULTI_WINDOW_PARAMS = {'start': 10, 'stop': 41, 'step': 1}


# Information about the length of availabile time in each model
# and the recomended maximum window length (the length of time - the minimum window (10))/10

MODEL_WINDOW_INFORMATION = {
    'MPI-ESM1-2-LR': {'time_length': 185, 'max_window': 59},
     'ACCESS-ESM1-5': {'time_length': 101, 'max_window': 31},
     'CESM2': {'time_length': 150, 'max_window': 47},
     'CanESM5': {'time_length': 100, 'max_window': 30},
     'GFDL-ESM4': {'time_length': 200, 'max_window': 64},
     'GISS-E2-1-G-CC': {'time_length': 180, 'max_window': 57},
     'MIROC-ES2L': {'time_length': 248, 'max_window': 80},
     'NorESM2-LM': {'time_length': 100, 'max_window': 30},
     'UKESM1-0-LL': {'time_length': 330, 'max_window': 201} # 107 from algorigthm 
}





