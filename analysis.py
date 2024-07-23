#!/usr/bin/env python
# coding: utf-8

"""
Script to perform configurable analysis of the strictness of textual EU legislative documents.
In this preliminary analysis, we focus on the distribution of regulatory statements in these documents
by legal policy area and over time. There is a Jupyter notebook version of this script located here: 
https://github.com/nature-of-eu-rules/eu-legislation-strictness-analysis/blob/main/analysis.ipynb
"""
import os
import json
import argparse
import sys
from os.path import exists
import numpy as np
# Setting up argument parser

argParser = argparse.ArgumentParser(description='Strictness analysis of EU legislation')
required = argParser.add_argument_group('required arguments')
required.add_argument("-in", "--input", required=True, help="Path to input CSV file with at least these columns: 'sent', 'word_count', 'year', 'dc_string', 'reg_count' see the following repos for more information: https://github.com/nature-of-eu-rules/regulatory-statement-classification and https://github.com/nature-of-eu-rules/data-preprocessing")
required.add_argument("-sm", "--strictm", choices=['count', 'mean'], required=True, help="Strictness metric used for the analysis which must be a value in ['count','mean']. 'count' just measures the number (density) of regulatory statements. 'mean' aggregates the mean number of regulatory statements by year and / or policy area.")
required.add_argument("-out", "--output", required=True, help="Path to a directory in which to store all generated plots, figures and information about the analysis e.g. 'output/' ")
argParser.add_argument("-noz", "--nozeros", action="store_true", help="Remove documents that contain NO legal obligations from the analysis")
argParser.add_argument("-t", "--time", choices=['year', 'date'], help="Time period resolution for analysis. Must be in ['year', 'date']. 'year' analyses strictness in each year and 'date' by day")
args = argParser.parse_args()

IN_FNAME = str(args.input) # Input filename
STRICTNESS_METRIC = str(args.strictm) # Strictness metric

NOZEROS = False
TIME_COLUMN = 'year'

if args.time:
    if args.time == 'date':
       TIME_COLUMN = 'date'

if args.nozeros:
    NOZEROS = True

STRICTNESS_METRIC = str(args.strictm) # Strictness metric
OUT_DIR = str(args.output) # Output directory
# Create directory if not exist
if not os.path.exists(OUT_DIR):
    # If it doesn't exist, create it
    os.makedirs(OUT_DIR)

import pandas as pd
import plotly.express as plotly

# Import data
metadata_df = pd.read_csv(IN_FNAME)
metadata_df['word_count'] = metadata_df['word_count'].replace(1,0)
if 'year' not in metadata_df.columns:
    metadata_df['date_adoption'] = pd.to_datetime(metadata_df['date_adoption'])
    # Generate a new 'Year' column
    metadata_df['year'] = metadata_df['date_adoption'].dt.year

if 'dc_string' not in metadata_df.columns:
    # Get one topic (to make analysis easier)
    metadata_df['dc_string'] = metadata_df['directory_code'].str.split('|').str[0]

print('Total number of documents: ', len(metadata_df))

#################################
# DESCRIPTIVE STATISTICS SECTION
#################################

# Check correlation of reg_count with other variables
import scipy
from sklearn import datasets

print()
print('Global correlation:')
print()
reg_has_variance = metadata_df['reg_count'].nunique() > 1
word_has_variance = metadata_df['word_count'].nunique() > 1

if reg_has_variance and word_has_variance:
    corr1, p_value1 = scipy.stats.spearmanr(metadata_df['reg_count'], metadata_df['word_count'])
    print('Spearman correlation (reg_count | word_count):', corr1, ' - ', p_value1)
    corr2, p_value2 = scipy.stats.pearsonr(metadata_df['reg_count'], metadata_df['word_count'])
    print('Pearson correlation (reg_count | word_count):', corr2, ' - ', p_value2)

word_count_test = metadata_df.groupby([TIME_COLUMN, 'dc_string'])['word_count'].sum()
word_count_test_ind = word_count_test.reset_index(drop=False)
word_count_new_frame = pd.DataFrame(word_count_test_ind.values.tolist(), columns=[TIME_COLUMN, 'policy_area', 'word_count'])
doc_count_test = metadata_df.groupby([TIME_COLUMN, 'dc_string'])['celex'].count()
doc_count_test_ind = doc_count_test.reset_index(drop=False)
doc_count_new_frame = pd.DataFrame(doc_count_test_ind.values.tolist(), columns=[TIME_COLUMN, 'policy_area', 'doc_count'])
policy_area_test = metadata_df.groupby([TIME_COLUMN, 'dc_string'])['reg_count'].sum()
policy_area_test_ind = policy_area_test.reset_index(drop=False)
policy_area_new_frame = pd.DataFrame(policy_area_test_ind.values.tolist(), columns=[TIME_COLUMN, 'policy_area', 'reg_count'])
policy_area_new_frame['doc_count'] = doc_count_new_frame['doc_count'].tolist()
policy_area_new_frame['word_count'] = word_count_new_frame['word_count'].tolist()

print()
print('Correlation relative to policy area and {}:'.format(TIME_COLUMN))
print()
corr_pearson1, p_valuea = scipy.stats.pearsonr(policy_area_new_frame['reg_count'], policy_area_new_frame['word_count'])
corr_spearman1, p_valueb = scipy.stats.spearmanr(policy_area_new_frame['reg_count'], policy_area_new_frame['word_count'])
print('Pearson correlation (reg_count | word_count):', corr_pearson1, ' - ', format(p_valuea, 'f'))
print('Spearman correlation (reg_count | word_count):', corr_spearman1, ' - ', format(p_valueb, 'f'))
corr_pearson, p_value1 = scipy.stats.pearsonr(policy_area_new_frame['reg_count'], policy_area_new_frame['doc_count'])
corr_spearman, p_value2 = scipy.stats.spearmanr(policy_area_new_frame['reg_count'], policy_area_new_frame['doc_count'])
print('Pearson correlation (reg_count | doc_count):', corr_pearson, ' - ', format(p_value1, 'f'))
print('Spearman correlation (reg_count | doc_count):', corr_spearman, ' - ', format(p_value2, 'f'))
print()

# Filter out other types of legislation other than 'R', 'L', and 'D' (Regulations, Directives, Decisions)
metadata_df = metadata_df[metadata_df['form'].isin(['R','L','D'])]
print('Total number of (R, L, D) documents: ', len(metadata_df))

# Get descriptive statistics about number of sentences (```sent_count```), words (```word_count```) and classified legal obligations (```reg_count```) in each document in the dataset:
desc_cols = set(metadata_df.columns) - {TIME_COLUMN, 'month', 'day', 'procedure_code'} # specify irrelevant columns
describe_df = metadata_df[list(desc_cols)] # remove irrelevant columns
describe_df.describe() # generate and display descriptive statistics

# How many documents do not contain any legal obligations:
no_legal_obligations = metadata_df[metadata_df['reg_count'] == 0]
print('Number of documents with no legal obligations: ', len(no_legal_obligations))

# The document with the most number of legal obligations is...
print('The document(s) with the most number of legal obligations: ')
most_legal_obligations = metadata_df[metadata_df['reg_count'] == metadata_df['reg_count'].max()]
print(most_legal_obligations['celex'].tolist())

# calculate the global mean of strictness (i.e. on average across all documents from all years):
global_mean = metadata_df['reg_count'].mean()
nonzero_df = metadata_df[metadata_df['reg_count'] > 0]
mean_for_nonzero = nonzero_df['reg_count'].mean()
describe_nonzero_df = nonzero_df[list(desc_cols)] # remove irrelevant columns
describe_nonzero_df.describe()

print('Global mean number of regulations per document: ', global_mean)
print('Global mean number of regulations per document (only for docs that contain at least one): ', mean_for_nonzero)

# Lets see the distribution of legislative documents by legal form (R, L, D). First we will prepare the preferences for plotting the graphs in this section:
import matplotlib.pyplot as plt
import random

metadata_df = metadata_df.astype({'year':'int'})
nonzero_df = nonzero_df.astype({'year':'int'})

SMALL_SIZE = 8
MEDIUM_SIZE = 12
MEDIUML_SIZE = 14
BIGGER_SIZE = 18

custom_palette = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
    '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
    '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
]

plt.rcParams['legend.title_fontsize'] = 'x-large'
# plt.rc('title', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUML_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Now plot the distribution by legal form:
# group the rows by 'year' and 'category', and count the number of occurrences of each category
counts = metadata_df.groupby(['year', 'form']).size()
# unstack the multi-level index and fill any missing values with 0
counts = counts.unstack().fillna(0)
# sort the counts in descending order
top_counts = counts.sum().sort_values(ascending=False)
# filter the counts DataFrame to only include the top categories
counts = counts[top_counts.index]
# create a stacked bar plot of the counts by year, set the plot title and labels
fig = plotly.bar(counts, title='Yearly distribution of EU legislation by legal form', labels={
                     "value" : "# of documents",
                     "form" : "Legal form"
                 }, color_discrete_sequence=custom_palette)

fig.update_xaxes(tickmode='linear', tickangle=60)
fig.write_html(os.path.join(OUT_DIR, 'eu_legislation_distribution_over_legal_forms.html'))
fig.show()

# Now we will plot how many documents there are across year and policy area. 

# group the rows by 'year' and 'category', and count the number of occurrences of each category
counts = metadata_df.groupby(['year', 'dc_string']).size()
# unstack the multi-level index and fill any missing values with 0
counts = counts.unstack().fillna(0)
# sort the counts in descending order
top_counts = counts.sum().sort_values(ascending=False)
# filter the counts DataFrame to only include the top categories
counts = counts[top_counts.index]
# create a stacked bar plot of the counts by year, set the plot title and labels
fig = plotly.bar(counts, title='Yearly distribution of EU legislation over major policy areas', labels={
                     "value" : "# of documents",
                     "dc_string" : "Policy Area"
                 }, color_discrete_sequence=custom_palette)
fig.update_xaxes(tickmode='linear', tickangle=60)
fig.write_html(os.path.join(OUT_DIR, 'eu_legislation_distribution_over_policy_areas.html'))
fig.show()

# Apart from Agriculture (the most prevalent policy area by far) what is the distribution of documents by policy areas and year?
no_agri_df = metadata_df[~metadata_df['dc_string'].isin(['Agriculture'])]
counts = no_agri_df.groupby(['year', 'dc_string']).size()
counts = counts.unstack().fillna(0)
top_counts = counts.sum().sort_values(ascending=False)
counts = counts[top_counts.index]

fig = plotly.bar(counts, title='Yearly distribution of EU legislation over major policy areas <br><sup>(Excludes Agriculture policy area)</sup>', labels={
                     "value" : "# of documents",
                     "dc_string" : "Policy Area"
                 }, color_discrete_sequence=custom_palette)
fig.update_xaxes(tickmode='linear', tickangle=60)
fig.write_html(os.path.join(OUT_DIR, 'eu_legislation_distribution_over_policy_areas_noagri.html'))
fig.show()

# Let us repeat the two plots above by limiting our focus to only those documents which contain at least one legal obligation for a specific agent:
counts = nonzero_df.groupby(['year', 'dc_string']).size()
counts = counts.unstack().fillna(0)
top_counts = counts.sum().sort_values(ascending=False)
counts = counts[top_counts.index]

fig = plotly.bar(counts, title='Yearly distribution of EU legislation over major policy areas <br><sup>(Excludes documents that do not contain at least one legal obligation)</sup>', labels={
                     "value" : "# of documents",
                     "dc_string" : "Policy Area"
                 }, color_discrete_sequence=custom_palette)
fig.update_xaxes(tickmode='linear', tickangle=60)
fig.write_html(os.path.join(OUT_DIR, 'eu_legislation_distribution_over_policy_areas_onlyregdocs.html'))
fig.show()

# And similarly removing Agriculture from consideration:
no_agri_nonzero_df = nonzero_df[~nonzero_df['dc_string'].isin(['Agriculture'])]
counts = no_agri_nonzero_df.groupby(['year', 'dc_string']).size()
counts = counts.unstack().fillna(0)
top_counts = counts.sum().sort_values(ascending=False)
counts = counts[top_counts.index]

fig = plotly.bar(counts, title='Yearly distribution of EU legislation over major policy areas <br><sup>(Excludes Agriculture policy area and documents which do not contain any legal obligations)</sup>', labels={
                     "value" : "# of documents",
                     "dc_string" : "Policy Area"
                 }, color_discrete_sequence=custom_palette)
fig.update_xaxes(tickmode='linear', tickangle=60)
fig.write_html(os.path.join(OUT_DIR, 'eu_legislation_distribution_over_policy_areas_onlyregdocs_noagri.html'))
fig.show()

#############################
# STRICTNESS ANALYSIS SECTION
#############################

# ## 2. Strictness analysis over time
# In this section we plot the mean strictness per year or date (day). That is, the average number of legal obligations per document in that year (or day).
# group the average number of legal obligations by year (regardless of legal form for now):

metric_col = 'total_reg_count'
if STRICTNESS_METRIC != 'count':
    metric_col = 'avg_reg_count'
    
df = None

if NOZEROS:
    df = metadata_df[metadata_df['reg_count'] > 0]
else:
    df = metadata_df
    
mean_years_noform = None

if STRICTNESS_METRIC == 'count':
    mean_years_noform = df.groupby([TIME_COLUMN])['reg_count'].sum()
    no_ind_noform = mean_years_noform.reset_index(drop=False)
else:
    mean_years_noform = df.groupby([TIME_COLUMN])['reg_count'].mean(numeric_only=True)
    no_ind_noform = mean_years_noform.reset_index(drop=False)
        
avg_reg_count_by_year_noform = pd.DataFrame(no_ind_noform.values.tolist(), columns=[TIME_COLUMN, metric_col])
max_time = avg_reg_count_by_year_noform[avg_reg_count_by_year_noform[metric_col] == avg_reg_count_by_year_noform[metric_col].max()]

# Out of interest, what is the year or date with the maximum value for total number of legal obligations?
print("The {} with the maximum {} regulatory statements is: ".format(TIME_COLUMN, STRICTNESS_METRIC),  max_time[TIME_COLUMN].tolist())

# Generic function to plot line graphs:
def plot_line_graph(df, xcol, ycol, labels, w, h, title, filepath, color='', color_discrete_sequence=''):    
    if color == '':
        if color_discrete_sequence == '':
            myFigure = plotly.line(df, x=xcol, y=ycol, labels=labels, title=title)
        else:
            myFigure = plotly.line(df, x=xcol, y=ycol, labels=labels, title=title, color_discrete_sequence=color_discrete_sequence)
    else:
        if color_discrete_sequence == '':
            myFigure = plotly.line(df, x=xcol, y=ycol, labels=labels, title=title, color=color)
        else:
            myFigure = plotly.line(df, x=xcol, y=ycol, labels=labels, title=title, color=color, color_discrete_sequence=color_discrete_sequence)
    
    if xcol == 'date':
        # Sort data by the selected time variable (date)
        df.sort_values(by=[xcol], inplace=True)
        # get the start and end points for the time period (first and last values in the column)
        start = pd.to_datetime(df[xcol].tolist()[0])
        end = pd.to_datetime(df[xcol].tolist()[len(df)-1])
        # Generate full complete range of values in the time period
        new_dates = pd.date_range(start=start,end=end,freq='D')
        # Create a DataFrame with the complete date range
        complete_df = pd.DataFrame({'date': new_dates})
        # Merge old dataframe with this new one
        df = pd.merge(complete_df, df, on='date', how='left')

        n = 10
        total_rows = df.shape[0]
        indices = np.linspace(0, total_rows - 1, n, dtype=int)
        selected_rows = df.iloc[indices]
        datetimelist = []
        for item in selected_rows['date'].tolist():
            datetimelist.append(pd.to_datetime(item))
                                
        myFigure.update_xaxes(
            tickvals=datetimelist,
            tickformat="%d-%m-%Y",
            tickangle=60
        )
    else:
        myFigure.update_xaxes(tickmode='linear',tickangle=60)        
    
    myFigure.write_html(filepath)
    myFigure.show()

# plot and save graph which measures mean strictness over the years (regardless of other criteria):
plot_line_graph(avg_reg_count_by_year_noform, xcol=TIME_COLUMN, ycol=metric_col, w=1200, h=400, labels={
                     "year": "Year",
                     "date" : "Day",
                     "avg_reg_count" : "Mean # of Legal Obligations",
                     "total_reg_count" : "Total # of Legal Obligations"
                 }, title="{} of EU Legal Obligations by {}".format(STRICTNESS_METRIC.upper(), TIME_COLUMN), filepath=os.path.join(OUT_DIR, "{}_legal_obligations_by_{}.html".format(STRICTNESS_METRIC, TIME_COLUMN)))

# ## 3. Strictness analysis by year and legal form
# ...now we group results by legal form as well.
mean_years = None
if STRICTNESS_METRIC == 'count':
    mean_years = df.groupby([TIME_COLUMN, 'form'])['reg_count'].sum()
    no_ind = mean_years.reset_index(drop=False)
else:
    mean_years = df.groupby([TIME_COLUMN, 'form'])['reg_count'].mean(numeric_only=True)
    no_ind = mean_years.reset_index(drop=False)
        
avg_reg_count_by_year = pd.DataFrame(no_ind.values.tolist(), columns=[TIME_COLUMN, 'form', metric_col])

# plot the graph...
# avg_reg_count_by_year['date'] = pd.to_datetime(avg_reg_count_by_year_noform['date'])
plot_line_graph(avg_reg_count_by_year, xcol=TIME_COLUMN, ycol=metric_col, w=1200, h=400, labels={
                     "year": "Year",
                     "date" : "Day",
                     "avg_reg_count" : "Mean # of Legal Obligations",
                     "total_reg_count" : "Total # of Legal Obligations"
                 }, title="{} of EU Legal Obligations by {} and Legal Form".format(STRICTNESS_METRIC.upper(), TIME_COLUMN), color="form", filepath=os.path.join(OUT_DIR,"{}_legal_obligations_by_{}_and_form.html".format(STRICTNESS_METRIC, TIME_COLUMN)))


# ## 4. Strictness analysis by year and policy area
# Now we analyse the strictness over time grouped by policy area only.
# First let's prepare or group the dataframe legal topic (policy area e.g. agriculture, taxation etc.). ```dc_string``` is the column which holds the (most general) legal topic:

mean_years_policy = None
if STRICTNESS_METRIC == 'count':
    mean_years_policy = df.groupby([TIME_COLUMN, 'dc_string'])['reg_count'].sum()
    no_ind_policy = mean_years_policy.reset_index(drop=False)
else:
    mean_years_policy = df.groupby([TIME_COLUMN, 'dc_string'])['reg_count'].mean(numeric_only=True)
    no_ind_policy = mean_years_policy.reset_index(drop=False)
        
avg_reg_count_by_year_policy = pd.DataFrame(no_ind_policy.values.tolist(), columns=[TIME_COLUMN, 'dc_string', metric_col])

# plot this graph...
# Custom colours for the lines (one for each legal topic)
# custom_palette = [
#     '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
#     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', 
#     '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5', 
#     '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
# ]

plot_line_graph(avg_reg_count_by_year_policy, xcol=TIME_COLUMN, ycol=metric_col, w=1300, h=400, labels={
                     "year": "Year",
                     "avg_reg_count": "Mean # of Legal Obligations",
                     "total_reg_count": "Total # of Legal Obligations",
                     "dc_string": "Policy Area"
                 }, title="{} of EU Legal Obligations by {} and Policy Area".format(STRICTNESS_METRIC.upper(), TIME_COLUMN), color="dc_string", filepath=os.path.join(OUT_DIR, "{}_legal_obligations_by_{}_and_policyarea.html".format(STRICTNESS_METRIC.upper(), TIME_COLUMN)), color_discrete_sequence=custom_palette)


# Plot separate graphs for each policy area:
import matplotlib.pyplot as plt

fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8), (ax9, ax10, ax11, ax12), (ax13, ax14, ax15, ax16), (ax17, ax18, ax19, ax20), (ax21, ax22, ax23, ax24)) = plt.subplots(nrows=6, ncols=4, sharex=True, sharey=True, figsize=(14,12))

# Doing each of these manually (ugh)
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Environment, consumers and health protection'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax1)
ax1.set_title("Env, consumers, health")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Competition policy'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax2)
ax2.set_title("Competition policy")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'External relations'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax3)
ax3.set_title("External relations")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Energy'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax4)
ax4.set_title("Energy")

avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Transport policy'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax5)
ax5.set_title("Transport policy")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Taxation'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax6)
ax6.set_title("Taxation")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Fisheries'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax7)
ax7.set_title("Fisheries")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == "People's Europe"].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax8)
ax8.set_title("People's Europe")

avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Customs Union and free movement of goods'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax9)
ax9.set_title("Customs Union")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Industrial policy and internal market'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax10)
ax10.set_title("Industrial policy")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Science, information, education and culture'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax11)
ax11.set_title("Science")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == "Law relating to undertakings"].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax12)
ax12.set_title("Undertakings")

avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Area of freedom, security and justice'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax13)
ax13.set_title("Security")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Regional policy and coordination of structural instruments'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax14)
ax14.set_title("Regional policy")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Freedom of movement for workers and social policy'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax15)
ax15.set_title("Freedom of movement")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == "Common Foreign and Security Policy"].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax16)
ax16.set_title("Common Foreign")

avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Provisional data'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax17)
ax17.set_title("Provisional data")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Economic and monetary policy and free movement of capital'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax18)
ax18.set_title("Economic policy")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Right of establishment and freedom to provide services'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax19)
ax19.set_title("Right of establishment")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == "General, financial and institutional matters"].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax20)
ax20.set_title("General, financial")

avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Agriculture'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax21)
# avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == 'Environment, consumers and health protection'].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax21)
ax21.set_title("Agriculture")
avg_reg_count_by_year_policy[avg_reg_count_by_year_policy['dc_string'] == "unknown topic"].plot(x=TIME_COLUMN, y=metric_col, legend=False, ax=ax22)
ax22.set_title("Unknown")
# If we don't do tight_layout() there can be strange rendering issues such as overlapping diagrams or text
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "legal_obligations_by_{}_separate_plots_for_policyareas.svg".format(TIME_COLUMN)), format="svg")

# ## 5. Strictness analysis by legal form and policy area
# Now we analyse the strictness over based on legal form and policy are using a heatmap. This view can give insight into, for example, if Agriculture regulations are more strict than External relations decisions or directives.
# First prepare the dataframe with only the data we need to plot the heatmap (only need form, year, policy area and mean number of regulatory statements):

mean_years_policy_form = None
if STRICTNESS_METRIC == 'count':
    mean_years_policy_form = df.groupby([TIME_COLUMN, 'form', 'dc_string'])['reg_count'].sum()
    no_ind_policy_form = mean_years_policy_form.reset_index(drop=False)
else:
    mean_years_policy_form = df.groupby([TIME_COLUMN, 'form', 'dc_string'])['reg_count'].mean(numeric_only=True)
    no_ind_policy_form = mean_years_policy_form.reset_index(drop=False)
        
avg_reg_count_by_year_policy_form = pd.DataFrame(no_ind_policy_form.values.tolist(), columns=[TIME_COLUMN, 'form', 'dc_string', metric_col])

# Now prepare the data (pivot the dataframe) to put it in a format for rendering in a heatmap (2D matrix):
df_heatmap = avg_reg_count_by_year_policy_form.pivot_table(values=metric_col,index='dc_string',columns='form')

# Now plot the heatmap:
import seaborn as sns
plt.subplots(figsize=(50,15))
s = sns.heatmap(df_heatmap, cmap ='rocket_r', linewidths = 0.50, annot = True)
s.set_ylabel('Policy Area', fontsize=25)
s.set_xlabel('Legal Form', fontsize=25)
fig = s.get_figure()
fig.savefig(os.path.join(OUT_DIR, "legal_obligations_heatmap_by_policyarea_and_form.svg"), format="svg")


