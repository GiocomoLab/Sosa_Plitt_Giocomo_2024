'''
Path dictionary for msosa on computer 'firebird'.

Example usage in a jupyter notebook:
` from reward_relative.path_dict_firebird import path_dictionary as path dict `

'''

import os

# ## REMOTE ###
# RCLONE_DATA_ROOT = "DATA"

###  LOCAL  ###
HOME = os.path.expanduser("~")

DATA_ROOT = os.path.join("/data","2P")  # parent path to data
PP_ROOT = DATA_ROOT # path to preprocessed data
SBX_ROOT = os.path.join("/mnt","oak","InVivoDA","2P_Data") # scanbox data path, if different from preprocessed data path
GDRIVE_ROOT = os.path.join("/mnt","gdrive","2P_Data") # alternate raw data path - MetaLearn_GDrive: mounted via rclone

GIT_ROOT = os.path.join(HOME,"local_repos","2p_repos")

FIG_DIR = os.path.join(DATA_ROOT,"fig_scratch")


path_dictionary = {
    "preprocessed_root": PP_ROOT,
    "sbx_root": SBX_ROOT,
    "gdrive_root": GDRIVE_ROOT,
    "VR_Data": os.path.join(PP_ROOT,"VR_Data"),
    "git_repo_root": GIT_ROOT,
    "TwoPUtils": os.path.join(GIT_ROOT,"TwoPUtils"),
    "home": HOME,
    "fig_dir": FIG_DIR,
}


os.makedirs(path_dictionary['fig_dir'],exist_ok=True)
