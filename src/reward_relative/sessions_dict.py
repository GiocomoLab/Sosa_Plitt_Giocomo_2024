# Dictionaries of session meta-data per animal
import numpy as np

single_plane = {
    
    'GCAMP2': ({'date': '29_09_2022', 'scene': 'Env1_LocationA', 'session': 1, 'scan': 3, 'exp_day': 1},
               {'date': '30_09_2022', 'scene': 'Env1_LocationA', 'session': 3, 'scan': 2, 'exp_day': 2},
               {'date': '01_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 3, 'exp_day': 3},
               {'date': '02_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 4},
               {'date': '03_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 5},
               {'date': '04_10_2022', 'scene': 'Env1_LocationA', 'session': 3, 'scan': 1, 'exp_day': 6},
               {'date': '05_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 7},
               {'date': '06_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 8},
               {'date': '07_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 3, 'exp_day': 9},
               {'date': '08_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 4, 'exp_day': 10},
               {'date': '09_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 11},
               {'date': '10_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 3, 'exp_day': 12},
               {'date': '11_10_2022', 'scene': 'Env1_LocationA', 'session': 3, 'scan': 4, 'exp_day': 13},
               {'date': '12_10_2022', 'scene': 'Env1_LocationA', 'session': 4, 'scan': 2, 'exp_day': 14},
              ),
    
    'GCAMP3': ({'date': '01_10_2022', 'scene': 'Env1_LocationC', 'session': 1, 'scan': 3, 'exp_day': 1},
               {'date': '02_10_2022', 'scene': 'Env1_LocationC', 'session': 3, 'scan': 1, 'exp_day': 2},
               {'date': '03_10_2022', 'scene': 'Env1_LocationC_to_A', 'session': 1, 'scan': 3, 'exp_day': 3},
               {'date': '04_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 4},
               {'date': '05_10_2022', 'scene': 'Env1_LocationA_to_B', 'session': 1, 'scan': 4, 'exp_day': 5},
               {'date': '06_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 4, 'exp_day': 6},
               {'date': '07_10_2022', 'scene': 'Env1_LocationB_to_C', 'session': 1, 'scan': 1, 'exp_day': 7},
               {'date': '08_10_2022', 'scene': 'Env1_C_to_Env2_B', 'session': 1, 'scan': 2, 'exp_day': 8},
               {'date': '09_10_2022', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 2, 'exp_day': 9},
               {'date': '10_10_2022', 'scene': 'Env2_LocationB_to_A', 'session': 1, 'scan': 1, 'exp_day': 10},
               {'date': '11_10_2022', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 2, 'exp_day': 11},
               {'date': '12_10_2022', 'scene': 'Env2_LocationA_to_C', 'session': 1, 'scan': 2, 'exp_day': 12},
               {'date': '13_10_2022', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 3, 'exp_day': 13},
               {'date': '14_10_2022', 'scene': 'Env2_LocationC_to_B', 'session': 1, 'scan': 2, 'exp_day': 14},
               {'date': '15_10_2022', 'scene': 'Env2_B_to_Env1_C', 'session': 1, 'scan': 2, 'exp_day': 15},
              ),
              
    
    'GCAMP4': ({'date': '06_10_2022', 'scene': 'Env1_LocationB', 'session': 1, 'scan': 4, 'exp_day': 1},
               {'date': '07_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 2},
               {'date': '08_10_2022', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': 2, 'exp_day': 3},
               {'date': '09_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 4},
               {'date': '10_10_2022', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 2, 'exp_day': 5},
               {'date': '11_10_2022', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 2, 'exp_day': 6},
               {'date': '12_10_2022', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 2, 'exp_day': 7},
               {'date': '13_10_2022', 'scene': 'Env1_B_to_Env2_C', 'session': 1, 'scan': 2, 'exp_day': 8},
               {'date': '14_10_2022', 'scene': 'Env2_LocationC', 'session': 4, 'scan': 2, 'exp_day': 9},
                   #{'date': '14_10_2022', 'scene': 'Env2_LocationC', 'session': 6, 'scan': 5, 'exp_day': 9},
               {'date': '15_10_2022', 'scene': 'Env2_LocationC_to_A', 'session': 1, 'scan': 2, 'exp_day': 10},
               {'date': '16_10_2022', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 3, 'exp_day': 11},
               {'date': '17_10_2022', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': 3, 'exp_day': 12},
               {'date': '18_10_2022', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 2, 'exp_day': 13},
               {'date': '19_10_2022', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': 2, 'exp_day': 14},
               {'date': '20_10_2022', 'scene': 'Env2_C_to_Env1_B', 'session': 1, 'scan': 5, 'exp_day': 15},
              ),
    
    'GCAMP5': ({'date': '29_09_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 7, 'exp_day': 1},
               {'date': '30_09_2022', 'scene': 'Env1_LocationA', 'session': 3, 'scan':36, 'exp_day': 2},
               {'date': '01_10_2022', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 2, 'exp_day': 3},
               {'date': '02_10_2022', 'scene': 'Env1_LocationC', 'session': 2, 'scan': np.nan, 'exp_day': 4}, #VR only starts
               {'date': '03_10_2022', 'scene': 'Env1_LocationC_to_B', 'session': 4, 'scan': np.nan, 'exp_day': 5},
               {'date': '04_10_2022', 'scene': 'Env1_LocationB', 'session': 1, 'scan': np.nan, 'exp_day': 6},
               (
                   {'date': '05_10_2022', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': np.nan, 'exp_day': 7},
                   {'date': '05_10_2022', 'scene': 'Env1_LocationB_to_A', 'session': 2, 'scan': np.nan, 'exp_day': 7},
               ),
               {'date': '06_10_2022', 'scene': 'Env1_A_to_Env2_B', 'session': 1, 'scan': np.nan, 'exp_day': 8},
               {'date': '07_10_2022', 'scene': 'Env2_LocationB', 'session': 2, 'scan': np.nan, 'exp_day': 9},
               {'date': '08_10_2022', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': np.nan, 'exp_day': 10},
               {'date': '09_10_2022', 'scene': 'Env2_LocationC', 'session': 2, 'scan': np.nan, 'exp_day': 11},
               {'date': '10_10_2022', 'scene': 'Env2_LocationC_to_A', 'session': 1, 'scan': np.nan, 'exp_day': 12},
               {'date': '11_10_2022', 'scene': 'Env2_LocationA', 'session': 2, 'scan': np.nan, 'exp_day': 13},
               {'date': '12_10_2022', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': np.nan, 'exp_day': 14},
              ),
    
    'GCAMP6': ({'date': '17_10_2022', 'scene': 'Env1_LocationB', 'session': 1, 'scan': 4, 'exp_day': 1},
               {'date': '18_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 2},
               {'date': '19_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 3, 'exp_day': 3},
               {'date': '20_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 4},
               {'date': '21_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 5},
               {'date': '22_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 6},
               {'date': '23_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 3, 'exp_day': 7},
               {'date': '24_10_2022', 'scene': 'Env1_LocationB', 'session': 3, 'scan': 2, 'exp_day': 8},
               {'date': '25_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 3, 'exp_day': 9},
               {'date': '26_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 1, 'exp_day': 10},
               {'date': '27_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 11},
               {'date': '28_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 6, 'exp_day': 12},
               {'date': '29_10_2022', 'scene': 'Env1_LocationB', 'session': 3, 'scan': 4, 'exp_day': 13},
               {'date': '30_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 14},
              ),
    
    'GCAMP7': ({'date': '16_10_2022', 'scene': 'Env1_LocationA', 'session': 1, 'scan': 4, 'exp_day': 1},
               {'date': '17_10_2022', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 3, 'exp_day': 2},
               {'date': '18_10_2022', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 3, 'exp_day': 3},
               {'date': '19_10_2022', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 4, 'exp_day': 4},
               {'date': '20_10_2022', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 2, 'exp_day': 5},
               {'date': '21_10_2022', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 6},
               {'date': '22_10_2022', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': 2, 'exp_day': 7},
               {'date': '23_10_2022', 'scene': 'Env1_A_to_Env2_B', 'session': 1, 'scan': 2, 'exp_day': 8},
               {'date': '24_10_2022', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 3, 'exp_day': 9},
               {'date': '25_10_2022', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': 2, 'exp_day': 10},
               {'date': '26_10_2022', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 2, 'exp_day': 11},
               {'date': '27_10_2022', 'scene': 'Env2_LocationC_to_A', 'session': 2, 'scan': 1, 'exp_day': 12},
               {'date': '28_10_2022', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 3, 'exp_day': 13},
               {'date': '29_10_2022', 'scene': 'Env2_LocationA_to_B', 'session': 2, 'scan': 4, 'exp_day': 14},
               {'date': '30_10_2022', 'scene': 'Env2_B_to_Env1_A', 'session': 1, 'scan': 3, 'exp_day': 15},
              ),
    
    'GCAMP10': ({'date': '20_02_2023', 'scene': 'RunningTraining_scan', 'session': 16, 'scan': 7, 'exp_day': 0},
                {'date': '21_02_2023', 'scene': 'Env1_LocationC', 'session': 1, 'scan': 7, 'exp_day': 1},
                # {'date': '22_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 3, 'exp_day': 2}, # weird static discharge stopped the scan prematurely, no sync at end of session
                {'date': '22_02_2023', 'scene': 'Env1_LocationC', 'session': 3, 'scan': 7, 'exp_day': 2},
                {'date': '23_02_2023', 'scene': 'Env1_LocationC', 'session': 3, 'scan': 3, 'exp_day': 3},
                {'date': '24_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 3, 'exp_day': 4},
                {'date': '25_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 2, 'exp_day': 5},
                {'date': '26_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 8, 'exp_day': 6},
                {'date': '27_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 2, 'exp_day': 7},
                {'date': '28_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 10, 'exp_day': 8},
                {'date': '01_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 2, 'exp_day': 9},
                {'date': '02_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 2, 'exp_day': 10},
                {'date': '03_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 8, 'exp_day': 11},
                {'date': '04_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 10, 'exp_day': 12},
                {'date': '05_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 11, 'exp_day': 13},
                {'date': '06_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 8, 'exp_day': 14},
                {'date': '07_03_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 1, 'exp_day': 15},
               ),
    
    'GCAMP11': ({'date': '21_02_2023', 'scene': 'Env1_LocationB', 'session': 1, 'scan': np.nan, 'exp_day': 1},
                {'date': '22_02_2023', 'scene': 'Env1_LocationB', 'session': 3, 'scan': np.nan, 'exp_day': 2},
                {'date': '23_02_2023', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': 2, 'exp_day': 3},
                {'date': '24_02_2023', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 4},
                {'date': '25_02_2023', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 8, 'exp_day': 5},
                {'date': '26_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 7, 'exp_day': 6},
                {'date': '27_02_2023', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 2, 'exp_day': 7},
                {'date': '28_02_2023', 'scene': 'Env1_B_to_Env2_C', 'session': 1, 'scan': 2, 'exp_day': 8},
                {'date': '01_03_2023', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 8, 'exp_day': 9},
                {'date': '02_03_2023', 'scene': 'Env2_LocationC_to_A', 'session': 2, 'scan': 11, 'exp_day': 10},
                {'date': '03_03_2023', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 5, 'exp_day': 11},
                {'date': '04_03_2023', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': 3, 'exp_day': 12},
                {'date': '05_03_2023', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 7, 'exp_day': 13},
                {'date': '06_03_2023', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': 6, 'exp_day': 14},
                {'date': '07_03_2023', 'scene': 'Env2_C_to_Env1_B', 'session': 1, 'scan': 2, 'exp_day': 15},
               ),
    
    'GCAMP12': ({'date': '20_02_2023', 'scene': 'RunningTraining_scan', 'session': 2, 'scan': 6, 'exp_day': 0},
                {'date': '21_02_2023', 'scene': 'Env1_LocationB', 'session': 1, 'scan': 1, 'exp_day': 1},
                   # {'date': '21_02_2023', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 4, 'exp_day': 1}, # other FOV
                {'date': '22_02_2023', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 5, 'exp_day': 2},
                {'date': '23_02_2023', 'scene': 'Env1_LocationB_to_C', 'session': 2, 'scan': 5, 'exp_day': 3},
                {'date': '24_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 6, 'exp_day': 4},
                {'date': '25_02_2023', 'scene': 'Env1_LocationC_to_A', 'session': 1, 'scan': 5, 'exp_day': 5},
                {'date': '26_02_2023', 'scene': 'Env1_LocationA', 'session': 3, 'scan': 4, 'exp_day': 6},
                {'date': '27_02_2023', 'scene': 'Env1_LocationA_to_B', 'session': 1, 'scan': 2, 'exp_day': 7},
                {'date': '28_02_2023', 'scene': 'Env1_B_to_Env2_A', 'session': 1, 'scan': 2, 'exp_day': 8},
                {'date': '01_03_2023', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 5, 'exp_day': 9},
                {'date': '02_03_2023', 'scene': 'Env2_LocationA_to_C', 'session': 1, 'scan': 5, 'exp_day': 10},
                {'date': '03_03_2023', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 11, 'exp_day': 11},
                {'date': '04_03_2023', 'scene': 'Env2_LocationC_to_B', 'session': 2, 'scan': 8, 'exp_day': 12},
                {'date': '05_03_2023', 'scene': 'Env2_LocationB', 'session': 3, 'scan': 4, 'exp_day': 13},
                {'date': '06_03_2023', 'scene': 'Env2_LocationB_to_A', 'session': 1, 'scan': 3, 'exp_day': 14},
                {'date': '07_03_2023', 'scene': 'Env2_A_to_Env1_B', 'session': 1, 'scan': 5, 'exp_day': 15},
               ),
    
    'GCAMP13': ({'date': '20_02_2023', 'scene': 'RunningTraining_scan', 'session': 10, 'scan': 6, 'exp_day': 0},
                {'date': '21_02_2023', 'scene': 'Env1_LocationC', 'session': 1, 'scan': 2, 'exp_day': 1},
                {'date': '22_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 2, 'exp_day': 2},
                {'date': '23_02_2023', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 1, 'exp_day': 3},
                {'date': '24_02_2023', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 4, 'exp_day': 4},
                {'date': '25_02_2023', 'scene': 'Env1_LocationB_to_A', 'session': 2, 'scan': 5, 'exp_day': 5},
                {'date': '26_02_2023', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 6},
                {'date': '27_02_2023', 'scene': 'Env1_LocationA_to_C', 'session': 4, 'scan': 6, 'exp_day': 7},
                {'date': '28_02_2023', 'scene': 'Env1_C_to_Env2_A', 'session': 1, 'scan': 2, 'exp_day': 8},
                {'date': '01_03_2023', 'scene': 'Env2_LocationA', 'session': 3, 'scan': 5, 'exp_day': 9},
                {'date': '02_03_2023', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': 2, 'exp_day': 10},
                {'date': '03_03_2023', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 2, 'exp_day': 11},
                {'date': '04_03_2023', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': 2, 'exp_day': 12},
                # {'date': '05_03_2023', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 2, 'exp_day': 13}, # wrong depth, too high
                {'date': '05_03_2023', 'scene': 'Env2_LocationC', 'session': 4, 'scan': 5, 'exp_day': 13}, # use this one, correct depth
                {'date': '06_03_2023', 'scene': 'Env2_LocationC_to_A', 'session': 1, 'scan': 2, 'exp_day': 14},
                {'date': '07_03_2023', 'scene': 'Env2_A_to_Env1_C', 'session': 1, 'scan': 1, 'exp_day': 15},
               ),
    
    'GCAMP14': ({'date': '20_02_2023', 'scene': 'RunningTraining_scan', 'session': 13, 'scan': 3, 'exp_day': 0},
                {'date': '21_02_2023', 'scene': 'Env1_LocationA', 'session': 1, 'scan': 3, 'exp_day': 1},
                {'date': '22_02_2023', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 2, 'exp_day': 2},
                {'date': '23_02_2023', 'scene': 'Env1_LocationA_to_B', 'session': 1, 'scan': 2, 'exp_day': 3},
                {'date': '24_02_2023', 'scene': 'Env1_LocationB', 'session': 3, 'scan': 8, 'exp_day': 4},
                {'date': '25_02_2023', 'scene': 'Env1_LocationB_to_C', 'session': 1, 'scan': 8, 'exp_day': 5},
                {'date': '26_02_2023', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 5, 'exp_day': 6},
                {'date': '27_02_2023', 'scene': 'Env1_LocationC_to_A', 'session': 1, 'scan': 9, 'exp_day': 7},
                {'date': '28_02_2023', 'scene': 'Env1_A_to_Env2_C', 'session': 2, 'scan': 7, 'exp_day': 8},
                {'date': '01_03_2023', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 8, 'exp_day': 9},
                {'date': '02_03_2023', 'scene': 'Env2_LocationC_to_B', 'session': 2, 'scan': 8, 'exp_day': 10},
                {'date': '03_03_2023', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 5, 'exp_day': 11},
                {'date': '04_03_2023', 'scene': 'Env2_LocationB_to_A', 'session': 1, 'scan': 7, 'exp_day': 12},
                {'date': '05_03_2023', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 8, 'exp_day': 13},
                {'date': '06_03_2023', 'scene': 'Env2_LocationA_to_C', 'session': 1, 'scan': 5, 'exp_day': 14},
                {'date': '07_03_2023', 'scene': 'Env2_C_to_Env1_A', 'session': 4, 'scan': 11, 'exp_day': 15},
               ),
    
    'GCAMP15': ({'date': '24_03_2024', 'scene': 'RunningTraining_scan', 'session': 3, 'scan': 9, 'exp_day': 0},
                {'date': '25_03_2024', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 1, 'exp_day': 1},
                {'date': '26_03_2024', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 3, 'exp_day': 2},
                {'date': '27_03_2024', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 6, 'exp_day': 3},
                {'date': '28_03_2024', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 6, 'exp_day': 4},
                {'date': '29_03_2024', 'scene': 'Env1_LocationC_to_B', 'session': 2, 'scan': 5, 'exp_day': 5},
                {'date': '30_03_2024', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 2, 'exp_day': 6},
                {'date': '31_03_2024', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': 2, 'exp_day': 7},
                {'date': '01_04_2024', 'scene': 'Env1_A_to_Env2_B', 'session': 2, 'scan': 6, 'exp_day': 8},
                {'date': '02_04_2024', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 3, 'exp_day': 9},
                {'date': '03_04_2024', 'scene': 'Env2_LocationB_to_C', 'session': 2, 'scan': 8, 'exp_day': 10},
                {'date': '04_04_2024', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 4, 'exp_day': 11},
                {'date': '05_04_2024', 'scene': 'Env2_LocationC_to_A', 'session': 1, 'scan': 2, 'exp_day': 12},
                {'date': '06_04_2024', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 5, 'exp_day': 13},
                {'date': '07_04_2024', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': 3, 'exp_day': 14},
                {'date': '08_04_2024', 'scene': 'Env2_B_to_Env1_A', 'session': 1, 'scan': 2, 'exp_day': 15},
                ({'date': '12_04_2024', 'scene': 'Env1_ShrinkStretch_A_to_C', 'session': 2, 'scan': 2, 'exp_day': 16}, #stopped just after stretch transition
                 {'date': '12_04_2024', 'scene': 'Env1_ShrinkStretch_A_to_C', 'session': 3, 'scan': 4, 'exp_day': 16},
                ),
                {'date': '13_04_2024', 'scene': 'Env1_StretchShrink_C_to_B', 'session': 2, 'scan': 4, 'exp_day': 17},
               ),
    
    'GCAMP19': ({'date': '24_03_2024', 'scene': 'RunningTraining_scan', 'session': 2, 'scan': 9, 'exp_day': 0},
                {'date': '25_03_2024', 'scene': 'Env1_LocationC', 'session': 1, 'scan': 5, 'exp_day': 1},
                {'date': '26_03_2024', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 5, 'exp_day': 2},
                {'date': '27_03_2024', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 12, 'exp_day': 3},
                {'date': '28_03_2024', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 1, 'exp_day': 4},
                {'date': '29_03_2024', 'scene': 'Env1_LocationB_to_A', 'session': 2, 'scan': 4, 'exp_day': 5},
                {'date': '30_03_2024', 'scene': 'Env1_LocationA', 'session': 3, 'scan': 2, 'exp_day': 6},
                {'date': '31_03_2024', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 2, 'exp_day': 7},
                {'date': '01_04_2024', 'scene': 'Env1_C_to_Env2_A', 'session': 1, 'scan': 2, 'exp_day': 8},
                {'date': '02_04_2024', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 2, 'exp_day': 9},
                {'date': '03_04_2024', 'scene': 'Env2_LocationA_to_B', 'session': 3, 'scan': 9, 'exp_day': 10},
                {'date': '04_04_2024', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 2, 'exp_day': 11},
                {'date': '05_04_2024', 'scene': 'Env2_LocationB_to_C', 'session': 3, 'scan': 10, 'exp_day': 12},
                {'date': '06_04_2024', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 2, 'exp_day': 13},
                {'date': '07_04_2024', 'scene': 'Env2_LocationC_to_A', 'session': 1, 'scan': 2, 'exp_day': 14},
                {'date': '08_04_2024', 'scene': 'Env2_A_to_Env1_C', 'session': 1, 'scan': 2, 'exp_day': 15},
                {'date': '12_04_2024', 'scene': 'Env1_ShrinkStretch_C_to_B', 'session': 2, 'scan': 2, 'exp_day': 16},
                {'date': '13_04_2024', 'scene': 'Env1_StretchShrink_B_to_A', 'session': 2, 'scan': 3, 'exp_day': 17},
               ),
    
}

multi_plane = {
    
    'GCAMP17': ({'date': '25_03_2024', 'scene': 'RunningTraining_scan', 'session': 4, 'scan': 9, 'exp_day': 0},
                {'date': '26_03_2024', 'scene': 'Env2_LocationB', 'session': 1, 'scan': 6, 'exp_day': 1},
                {'date': '27_03_2024', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 9, 'exp_day': 2},
                {'date': '28_03_2024', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': 9, 'exp_day': 3},
                {'date': '29_03_2024', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 8, 'exp_day': 4},
                {'date': '30_03_2024', 'scene': 'Env2_LocationC_to_A', 'session': 1, 'scan': 5, 'exp_day': 5},
                {'date': '31_03_2024', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 5, 'exp_day': 6},
                {'date': '01_04_2024', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': 2, 'exp_day': 7},
                {'date': '02_04_2024', 'scene': 'Env2_B_to_Env1_A', 'session': 1, 'scan': 7, 'exp_day': 8},
                {'date': '03_04_2024', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 11, 'exp_day': 9},
                # {'date': '04_04_2024', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 7, 'exp_day': 9}, ## accidental repeat of day 9
                {'date': '05_04_2024', 'scene': 'Env1_LocationA_to_C', 'session': 1, 'scan': 6, 'exp_day': 10},
                {'date': '06_04_2024', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 8, 'exp_day': 11}, #not copied
                {'date': '07_04_2024', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 6, 'exp_day': 12},
                {'date': '08_04_2024', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 7, 'exp_day': 13},
                {'date': '09_04_2024', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': 4, 'exp_day': 14},
                {'date': '11_04_2024', 'scene': 'Env1_A_to_Env2_B', 'session': 1, 'scan': 2, 'exp_day': 15},
                {'date': '12_04_2024', 'scene': 'Env2_ShrinkStretch_B_to_C', 'session': 2, 'scan': 7, 'exp_day': 16},
                {'date': '13_04_2024', 'scene': 'Env2_StretchShrink_C_to_A', 'session': 2, 'scan': 8, 'exp_day': 17},
               ),
    
    'GCAMP18': ({'date': '24_03_2024', 'scene': 'RunningTraining_scan', 'session': 2, 'scan': 12, 'exp_day': 0}, # single plane this day
                {'date': '25_03_2024', 'scene': 'Env2_LocationA', 'session': 1, 'scan': 8, 'exp_day': 1},
                {'date': '26_03_2024', 'scene': 'Env2_LocationA', 'session': 2, 'scan': 8, 'exp_day': 2},
                {'date': '27_03_2024', 'scene': 'Env2_LocationA_to_B', 'session': 1, 'scan': 2, 'exp_day': 3},
                {'date': '28_03_2024', 'scene': 'Env2_LocationB', 'session': 2, 'scan': 4, 'exp_day': 4},
                {'date': '29_03_2024', 'scene': 'Env2_LocationB_to_C', 'session': 1, 'scan': 7, 'exp_day': 5},
                {'date': '30_03_2024', 'scene': 'Env2_LocationC', 'session': 2, 'scan': 5, 'exp_day': 6},
                {'date': '31_03_2024', 'scene': 'Env2_LocationC_to_A', 'session': 2, 'scan': 8, 'exp_day': 7},
                {'date': '01_04_2024', 'scene': 'Env2_A_to_Env1_C', 'session': 2, 'scan': 9, 'exp_day': 8},
                {'date': '02_04_2024', 'scene': 'Env1_LocationC', 'session': 2, 'scan': 3, 'exp_day': 9},
                {'date': '03_04_2024', 'scene': 'Env1_LocationC_to_B', 'session': 1, 'scan': 12, 'exp_day': 10},
                {'date': '04_04_2024', 'scene': 'Env1_LocationB', 'session': 2, 'scan': 5, 'exp_day': 11},
                {'date': '05_04_2024', 'scene': 'Env1_LocationB_to_A', 'session': 1, 'scan': 13, 'exp_day': 12},
                {'date': '06_04_2024', 'scene': 'Env1_LocationA', 'session': 2, 'scan': 5, 'exp_day': 13},
                {'date': '07_04_2024', 'scene': 'Env1_LocationA_to_C', 'session': 4, 'scan': 14, 'exp_day': 14},
                {'date': '08_04_2024', 'scene': 'Env1_C_to_Env2_A', 'session': 1, 'scan': 6, 'exp_day': 15},
                {'date': '12_04_2024', 'scene': 'Env2_ShrinkStretch_A_to_B', 'session': 2, 'scan': 7, 'exp_day': 16},
                {'date': '13_04_2024', 'scene': 'Env2_StretchShrink_B_to_C', 'session': 2, 'scan': 6, 'exp_day': 17},
               ),


    'GRABDA2-21': ({'date': '16_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 1, 'scan': 4, 'exp_day': 1},
                   {'date': '18_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan': 5, 'exp_day': 3},
                   {'date': '20_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan': 2, 'exp_day': 5},
                   {'date': '21_01_2022', 'scene': 'NeuroMods_Day6Image', 'session': 1, 'scan': 3, 'exp_day': 6},
                   {'date': '22_01_2022', 'scene': 'NeuroMods_LocationB', 'session': 2, 'scan': 3, 'exp_day': 7},
                   {'date': '23_01_2022', 'scene': 'NeuroMods_Day8Image', 'session': 1, 'scan': 2, 'exp_day': 8},
                   {'date': '24_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan': 2, 'exp_day': 9},
                   {'date': '25_01_2022', 'scene': 'NM_MorphBlocks', 'session': 1, 'scan': 3, 'exp_day': 10},
                   {'date': '27_01_2022', 'scene': 'NM_RandomMorphs', 'session': 2, 'scan': 2, 'exp_day': 12},
                   {'date': '29_01_2022', 'scene': 'NM_RandomMorphs', 'session': 2, 'scan': 2, 'exp_day': 14},
                   {'date': '30_01_2022', 'scene': 'NM_Morph1ToDreamLand', 'session': 1, 'scan': 2, 'exp_day': 15},
                   {'date': '01_02_2022', 'scene': 'NM_Morph1ToDreamLand', 'session': 2, 'scan': 3, 'exp_day': 17},
                   {'date': '02_02_2022', 'scene': 'NM_DreamLandToPizzaLand', 'session': 3, 'scan': 4, 'exp_day': 18},
                   {'date': '04_02_2022', 'scene': 'NM_PizzaLandOnly', 'session': 2, 'scan': 2, 'exp_day': 20},
                   ),

    'GRABDA2-22': ({'date': '16_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 1, 'scan': 3, 'exp_day': 1},
                   {'date': '18_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan': 2, 'exp_day': 3},
                   {'date': '20_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan': 1, 'exp_day': 5},
                   {'date': '21_01_2022', 'scene': 'NeuroMods_Day6Image', 'session': 1, 'scan': 5, 'exp_day': 6},
                   {'date': '22_01_2022', 'scene': 'NeuroMods_LocationB', 'session': 1, 'scan': 5, 'exp_day': 7},
                   {'date': '23_01_2022', 'scene': 'NeuroMods_Day8Image', 'session': 2, 'scan': 2, 'exp_day': 8},
                   {'date': '24_01_2022', 'scene': 'NeuroMods_LocationA', 'session': 2, 'scan': 4, 'exp_day': 9},
                   {'date': '25_01_2022', 'scene': 'NM_MorphBlocks', 'session': 1, 'scan': 5, 'exp_day': 10},
                   {'date': '27_01_2022', 'scene': 'NM_RandomMorphs', 'session': 3, 'scan': 2, 'exp_day': 12},
                   {'date': '29_01_2022', 'scene': 'NM_RandomMorphs', 'session': 2, 'scan': 2, 'exp_day': 14},
                   {'date': '30_01_2022', 'scene': 'NM_Morph1ToDreamLand', 'session': 1, 'scan': 2, 'exp_day': 15},
                   {'date': '01_02_2022', 'scene': 'NM_Morph1ToDreamLand', 'session': 2, 'scan': 5, 'exp_day': 17},
                   {'date': '02_02_2022', 'scene': 'NM_DreamLandToPizzaLand', 'session': 2, 'scan': 4, 'exp_day': 18},
                   {'date': '04_02_2022', 'scene': 'NM_PizzaLandOnly', 'session': 2, 'scan': 2, 'exp_day': 20},
                   ),


}
