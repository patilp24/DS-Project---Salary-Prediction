# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 22:15:28 2023

@author: Dell
"""

import glassdoor_scraper as gs
import pandas as pd

path = "C:/Users/Dell/OneDrive/Desktop/Udemy Course/Projects/DS Project - Salary Prediction/chromedriver"

df = gs.get_jobs("Data Scientist", 15, False, path, 15)