# -*- coding: utf-8 -*-
"""
Dynamic MFA model for residential buildings in the Netherlands

@author: Raquel Kuperus
"""
#%% Import modules and input data

import pandas as pd
import numpy as np
import scipy.stats
from os import chdir
import matplotlib.pyplot as plt
import math

chdir('C:/Users/raque/OneDrive/Documents/COIN Project/DSM')

# Imports the stock composition of 2018, showing the UFA constructed between 1900 and 2018 that is still in stock in 2018
stock_composition_2018 = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT Composition of 2018 stock')
stock_composition_2018 = stock_composition_2018.set_index(['Year'])

# Imports the projected UFA constructed per housing type between 2019 and 2050
new_construction_2019_to_2050 = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT New construction')
new_construction_2019_to_2050 = new_construction_2019_to_2050.set_index(['Year'])

# Imports the historic and projected household stock development from 1921 to 2100
household_stock = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT Household stock')
household_stock = household_stock.set_index(['Year'])

# Imports the assumed share of modular new construction from 1900 to 2100 under a baseline and modular scenario
percent_modular = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT % modular').loc[1:]

# Imports the assumed number of temporary houses built from 1900 to 2100 under a baseline and modular scenario
number_temporary = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT # temporary').loc[1:]

# Imports the material intensities for traditional and modular housing types
material_intensities = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT Material intensities')

# Imports the percent of material that is reusable from a modular building
reuse_percent = pd.read_excel(r'DSM data.xlsx', sheet_name='INPUT Reuse %')

#%% Survival curve for houses built before 2019

# Number of rows for data from 1900-2100
timesteps = np.arange(0, 201)

# Parameters
pre_2019_mean_lifetime = 120
pre_2019_curve_shape = 2.95
pre_2019_curve_scale = pre_2019_mean_lifetime/math.gamma(1+1/pre_2019_curve_shape)

# Weibull distribution survival curve
pre_2019_survival_curve = scipy.stats.weibull_min.sf(timesteps, pre_2019_curve_shape, 0, pre_2019_curve_scale)
plt.plot(pre_2019_survival_curve)
plt.show()

#%% Survival curve for permanent houses built in 2019 or later

# Parameters
new_permanent_mean_lifetime = 75
new_permanent_curve_shape = 2.95
new_permanent_curve_scale = new_permanent_mean_lifetime/math.gamma(1+1/new_permanent_curve_shape)

# Weibull distribution survival curve
new_permanent_survival_curve = scipy.stats.weibull_min.sf(timesteps, new_permanent_curve_shape, 0, new_permanent_curve_scale)
plt.plot(new_permanent_survival_curve)
plt.show()

#%% Survival curve for temporary houses built in 2019 or later

# Parameters
new_temporary_lifetime = 15

# Fixed lifetime survival curve
new_temporary_survival_curve = np.zeros_like(timesteps)
new_temporary_survival_curve[0:new_temporary_lifetime] = 1

plt.plot(new_temporary_survival_curve)
plt.show()

#%% Given composition of the 2018 stock, calculate the inflows from 1900 to 2018

# Create empty matrix and fill with shifted survival curves
pre_2019_survival_curve_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)
for time in timesteps:
    pre_2019_survival_curve_matrix.loc[time:, time] = pre_2019_survival_curve[0:201 - time]

# Select the row of the survival curve matrix for 2018, with all columns between 1900 and 2018
timestep_2018 = 118
share_surviving_in_2018 = pre_2019_survival_curve_matrix.loc[timestep_2018,:timestep_2018]

# Given the survival curve, and the amount of each cohort that still exists in 2018, calculate the size of the original inflows
stock_composition_2018.index = share_surviving_in_2018.index
historical_inflows = stock_composition_2018.div(share_surviving_in_2018, axis=0)

# Rename columns
column_rename = dict(zip(historical_inflows.columns, new_construction_2019_to_2050.columns))
historical_inflows.rename(columns=column_rename, inplace=True)

# Sum inflows of all housing types to compare with CBS data
historical_inflows_sum = historical_inflows.sum(axis=1)
historical_inflows_sum.loc[1:].plot()

# Historical inflows are not necessarily correct but will yield more accurate approximation of outflows because the existing stock in 2018 is correct

#%% Stock driven model to calculate the projected number of homes constructed from from 2051-2100 based on the projected number of households in stock

max_year = household_stock.shape[0]
timesteps_2 = np.arange(0, max_year)

# Create empty matrix and fill with shifted survival curves
stock_driven_survival_curve_matrix = pd.DataFrame(0, index=timesteps_2, columns=timesteps_2)
for time in timesteps_2:
    if time <= timestep_2018:
        stock_driven_survival_curve_matrix.loc[time:, time] = pre_2019_survival_curve[0:max_year - time]
    else:
        stock_driven_survival_curve_matrix.loc[time:, time] = new_permanent_survival_curve[0:max_year - time]

# Create empty matrix
stock_driven_cohort_survival_matrix = pd.DataFrame(0, index=timesteps_2, columns=timesteps_2)

# Iteratively calculate the inflow from the stock and multiply the inflows by the survival curves
for time in timesteps_2:
    household_stock['Household inflow'].iloc[time] = (household_stock['Household stock'].iloc[time] - stock_driven_cohort_survival_matrix.loc[time, :].sum()) / stock_driven_survival_curve_matrix.loc[time, time]
    stock_driven_cohort_survival_matrix.loc[:, time] = stock_driven_survival_curve_matrix.loc[:, time] * household_stock['Household inflow'].iloc[time]

# Set row index to years instead of timesteps
stock_driven_cohort_survival_matrix.index = household_stock.index

# Divide the household inflow between the three dwelling types, and convert to UFA constructed
proportion_single_family_homes = 0.48
proportion_multi_family_homes = 0.52
within_single_proportion_detached_semi_detached = 0.34
within_single_proportion_terraced = 0.66

UFA_detached_semi_detached = 195
UFA_terraced = 123
UFA_apartment = 69

household_stock['Detached or semi-detached houses constructed (m2 UFA)']= household_stock['Household inflow'] * proportion_single_family_homes * within_single_proportion_detached_semi_detached * UFA_detached_semi_detached
household_stock['Terraced houses constructed (m2 UFA)']= household_stock['Household inflow'] * proportion_single_family_homes * within_single_proportion_terraced * UFA_terraced
household_stock['Apartments constructed (m2 UFA)']= household_stock['Household inflow'] * proportion_multi_family_homes * UFA_apartment

# Delete the rows before 2023 because new construction values for those years are taken from other sources
household_stock = household_stock.iloc[102:,:] #!!! 102 to start at 2023, or 130 to start at 2051

#%% Creating a single dataframe with the inflows from 1900-2100

columns_to_merge = ['Detached or semi-detached houses constructed (m2 UFA)', 'Terraced houses constructed (m2 UFA)', 'Apartments constructed (m2 UFA)']

new_construction_2019_to_2022 = new_construction_2019_to_2050.loc[2019:2022,:]

#!!! Change which 2019-2050 data to use by using new_construction_2019_to_2022 or new_construction_2019_to_2050 (also have to change row 148)
merged_df = pd.concat([historical_inflows, new_construction_2019_to_2022, household_stock[columns_to_merge]], ignore_index=True)
merged_df = merged_df.loc[1:,:] #Remove first row to remove large initial inflow

merged_df.plot()

years = np.arange(1901,2101)
merged_df.insert(0,'Year', years)

#%% Create dwelling inflow data for 1901-2100 based on the different scenarios for the market share of modular building

#!!! Select scenario
scenario = 'Baseline scenario' #Alternatives are 'Baseline scenario' or 'Modular scenario'

dwelling_inflows = pd.DataFrame()

dwelling_inflows['Tr_DSDH'] = merged_df['Detached or semi-detached houses constructed (m2 UFA)'] * (1-percent_modular[scenario])
dwelling_inflows['M_P_DSDH'] = (merged_df['Detached or semi-detached houses constructed (m2 UFA)'] * percent_modular[scenario]) - (number_temporary[scenario]//3) * UFA_detached_semi_detached
dwelling_inflows['M_Te_DSDH'] = (number_temporary[scenario]//3) * UFA_detached_semi_detached

dwelling_inflows['Tr_TH'] = merged_df['Terraced houses constructed (m2 UFA)'] * (1-percent_modular[scenario])
dwelling_inflows['M_P_TH'] = (merged_df['Terraced houses constructed (m2 UFA)'] * percent_modular[scenario]) - (number_temporary[scenario]//3) * UFA_terraced
dwelling_inflows['M_Te_TH'] = (number_temporary[scenario]//3) * UFA_terraced

dwelling_inflows['Tr_A'] = merged_df['Apartments constructed (m2 UFA)'] * (1-percent_modular[scenario])
dwelling_inflows['M_P_A'] = (merged_df['Apartments constructed (m2 UFA)'] * percent_modular[scenario]) - (number_temporary[scenario]//3) * UFA_apartment
dwelling_inflows['M_Te_A'] = (number_temporary[scenario]//3) * UFA_apartment

#%% Inflow driven model to calculate dwelling stocks and outflows (1901-2100)

timesteps = np.arange(1,201)

dwelling_stocks = pd.DataFrame()
dwelling_nas = pd.DataFrame()
dwelling_outflows = pd.DataFrame()

for dwelling_type in dwelling_inflows:

    # Create empty matrices
    survival_curve_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)
    cohort_survival_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)

    # Fill survival_curve_matrix with the shifted survival curves depending on dwelling type and whether construction is before or after 2019
    for time in timesteps:
        if time <= timestep_2018:
            survival_curve = pre_2019_survival_curve
        else:
            if dwelling_type.startswith('M_Te'):
                survival_curve = new_temporary_survival_curve
            else:
                survival_curve = new_permanent_survival_curve
        survival_curve_matrix.loc[time:, time] = survival_curve[0:201 - time]

        # Multiply dwelling inflows by the shifted survival curves
        cohort_survival_matrix.loc[:, time] = survival_curve_matrix.loc[:, time] * dwelling_inflows[dwelling_type].loc[time]

    # Calculate stocks, net addition to stocks, and outflows
    dwelling_stocks[dwelling_type] = cohort_survival_matrix.sum(axis=1)
    dwelling_nas[dwelling_type] = np.diff(dwelling_stocks[dwelling_type], prepend=0)  # prepending 0 assumes no initial stock, so that there can be a nas value for the first year
    dwelling_nas.index = dwelling_stocks.index
    dwelling_outflows[dwelling_type] = dwelling_inflows[dwelling_type] - dwelling_nas[dwelling_type]

#%% Converting dwelling inflows to material inflows

# Create a list of the different materials
materials = list(material_intensities.columns[3:])

# !!! Select whether to run model for modular houses that use conventional materials or biobased modular houses
modular_materialization = 'Modular (conventional materials)' #Alternatives are 'Modular (conventional materials)' or 'Modular (biobased materials)'

# Create a dictionary to store all the dataframes for the inflows of the different materials
material_inflows = {}

# Cycle through all the materials
for material in materials:

    # Creating a dataframe for each material
    material_inflows[f'{material}_inflows'] = pd.DataFrame()

    # And multiplying the dwelling inflows by the corressponding material intensities
    for dwelling_type in dwelling_inflows:
        if dwelling_type.startswith('Tr'):
            material_inflows[f'{material}_inflows'].loc[:44, dwelling_type] = dwelling_inflows[dwelling_type] * material_intensities.loc[material_intensities['House type'] == dwelling_type].iloc[0][material]
            material_inflows[f'{material}_inflows'].loc[45:69, dwelling_type] = dwelling_inflows[dwelling_type] * material_intensities.loc[material_intensities['House type'] == dwelling_type].iloc[1][material]
            material_inflows[f'{material}_inflows'].loc[70:99, dwelling_type] = dwelling_inflows[dwelling_type] * material_intensities.loc[material_intensities['House type'] == dwelling_type].iloc[2][material]
            material_inflows[f'{material}_inflows'].loc[100:, dwelling_type] = dwelling_inflows[dwelling_type] * material_intensities.loc[material_intensities['House type'] == dwelling_type].iloc[3][material]
        else:
            material_inflows[f'{material}_inflows'][dwelling_type] = dwelling_inflows[dwelling_type] * material_intensities.loc[material_intensities['Building method'] == modular_materialization].iloc[0][material]

#%% Inflow driven model to calculate material stocks and outflows

# Create dictionaries to store all the dataframes for the material stocks, net addition to stocks, and outflows
material_stocks = {}
material_nas = {}
material_outflows = {}

# For each material, run an inflow driven model to calculate material stocks and outflows
for material in materials:

    material_stocks[f'{material}_stocks'] = pd.DataFrame()
    material_nas[f'{material}_nas'] = pd.DataFrame()
    material_outflows[f'{material}_outflows'] = pd.DataFrame()

    for dwelling_type in material_inflows[f'{material}_inflows']:

        # Create empty matrices
        survival_curve_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)
        cohort_survival_matrix = pd.DataFrame(0, index=timesteps, columns=timesteps)

        # Fill survival_curve_matrix with the shifted survival curves depending on dwelling type and whether construction is before or after 2019
        for time in timesteps:
            if time <= timestep_2018:
                survival_curve = pre_2019_survival_curve
            else:
                if dwelling_type.startswith('M_Te'):
                    survival_curve = new_temporary_survival_curve
                else:
                    survival_curve = new_permanent_survival_curve
            survival_curve_matrix.loc[time:, time] = survival_curve[0:201 - time]

            # Multiply dwelling inflows by the shifted survival curves
            cohort_survival_matrix.loc[:, time] = survival_curve_matrix.loc[:, time] * material_inflows[f'{material}_inflows'][dwelling_type].loc[time]

        # Calculate stocks, net addition to stocks, and outflows
        material_stocks[f'{material}_stocks'][dwelling_type] = cohort_survival_matrix.sum(axis=1)
        material_nas[f'{material}_nas'][dwelling_type] = np.diff(material_stocks[f'{material}_stocks'][dwelling_type], prepend=0)  # prepending 0 assumes no initial stock, so that there can be a nas value for the first year
        material_nas[f'{material}_nas'].index = material_stocks[f'{material}_stocks'].index
        material_outflows[f'{material}_outflows'][dwelling_type] = material_inflows[f'{material}_inflows'][dwelling_type] - material_nas[f'{material}_nas'][dwelling_type]

#%% Extract the material inflows and outflows for 2019, 2030, 2050, and 2100 and calculate the amount of outflow that can be reused

for key, df in material_inflows.items():
    material_inflows[key] = df.set_index(pd.Index(years))

for key, df in material_outflows.items():
    material_outflows[key] = df.set_index(pd.Index(years))

results = pd.DataFrame()

for material in materials:
    inflow_2019 = material_inflows[f'{material}_inflows'].loc[2019].sum()
    outflow_2019 = material_outflows[f'{material}_outflows'].loc[2019].sum()
    modular_outflow_2019 = material_outflows[f'{material}_outflows'].loc[2019, ['M_P_DSDH','M_Te_DSDH','M_P_TH', 'M_Te_TH', 'M_P_A', 'M_Te_A']].sum()
    reusable_outflow_2019 = modular_outflow_2019 * reuse_percent[material].loc[0]

    inflow_2030 = material_inflows[f'{material}_inflows'].loc[2030].sum()
    outflow_2030 = material_outflows[f'{material}_outflows'].loc[2030].sum()
    modular_outflow_2030 = material_outflows[f'{material}_outflows'].loc[2030, ['M_P_DSDH','M_Te_DSDH','M_P_TH', 'M_Te_TH', 'M_P_A', 'M_Te_A']].sum()
    reusable_outflow_2030 = modular_outflow_2030 * reuse_percent[material].loc[0]

    inflow_2050 = material_inflows[f'{material}_inflows'].loc[2050].sum()
    outflow_2050 = material_outflows[f'{material}_outflows'].loc[2050].sum()
    modular_outflow_2050 = material_outflows[f'{material}_outflows'].loc[2050, ['M_P_DSDH','M_Te_DSDH','M_P_TH', 'M_Te_TH', 'M_P_A', 'M_Te_A']].sum()
    reusable_outflow_2050 = modular_outflow_2050 * reuse_percent[material].loc[0]

    inflow_2100 = material_inflows[f'{material}_inflows'].loc[2100].sum()
    outflow_2100 = material_outflows[f'{material}_outflows'].loc[2100].sum()
    modular_outflow_2100 = material_outflows[f'{material}_outflows'].loc[2100, ['M_P_DSDH','M_Te_DSDH','M_P_TH', 'M_Te_TH', 'M_P_A', 'M_Te_A']].sum()
    reusable_outflow_2100 = modular_outflow_2100 * reuse_percent[material].loc[0]

    results[material] = [inflow_2019, outflow_2019, reusable_outflow_2019,
                         inflow_2030, outflow_2030, reusable_outflow_2030,
                         inflow_2050, outflow_2050, reusable_outflow_2050,
                         inflow_2100, outflow_2100, reusable_outflow_2100]

results.insert(0, "Year", [2019,2019,2019,2030,2030,2030,2050,2050,2050,2100,2100,2100])
results.insert(1, "Flow", ['Inflow', 'Outflow', 'Reusable', 'Inflow', 'Outflow', 'Reusable', 'Inflow', 'Outflow', 'Reusable', 'Inflow', 'Outflow', 'Reusable'])

