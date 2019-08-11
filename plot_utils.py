#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:13:12 2018

@author: boray
"""

# plot utilities
import matplotlib.pyplot as plt
import os

def paper_fig_settings(addtosize=0):
    plt.style.use('seaborn-white')
    plt.rc('figure',dpi=144)
    plt.rc('text', usetex=False)
    plt.rc('axes',titlesize=16+addtosize)
    plt.rc('xtick', labelsize=12+addtosize)
    plt.rc('ytick', labelsize=12+addtosize)
    plt.rc('axes', labelsize=14+addtosize)
    plt.rc('legend', fontsize=10+addtosize)
    #plt.rc('text.latex', preamble=r'\usepackage{cmbright}')

from datetime import datetime
def save_fig(fig_id, tight_layout=True, prefix="", postfix=""):
    now = datetime.now()
    timestr = now.strftime("%Y%m%d-%H%M%S")
    path = os.path.join(prefix, "outputs", 
                        postfix, fig_id + "_"+timestr+".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)