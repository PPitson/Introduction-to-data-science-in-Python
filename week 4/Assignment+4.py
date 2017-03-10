
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# In[1]:

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind


# # Assignment 4 - Hypothesis Testing
# This assignment requires more individual learning than previous assignments - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.
# 
# Definitions:
# * A _quarter_ is a specific three month period, Q1 is January through March, Q2 is April through June, Q3 is July through September, Q4 is October through December.
# * A _recession_ is defined as starting with two consecutive quarters of GDP decline, and ending with two consecutive quarters of GDP growth.
# * A _recession bottom_ is the quarter within a recession which had the lowest GDP.
# * A _university town_ is a city which has a high percentage of university students compared to the total population of the city.
# 
# **Hypothesis**: University towns have their mean housing prices less effected by recessions. Run a t-test to compare the ratio of the mean price of houses in university towns the quarter before the recession starts compared to the recession bottom. (`price_ratio=quarter_before_recession/recession_bottom`)
# 
# The following data files are available for this assignment:
# * From the [Zillow research data site](http://www.zillow.com/research/data/) there is housing data for the United States. In particular the datafile for [all homes at a city level](http://files.zillowstatic.com/research/public/City/City_Zhvi_AllHomes.csv), ```City_Zhvi_AllHomes.csv```, has median home sale prices at a fine grained level.
# * From the Wikipedia page on college towns is a list of [university towns in the United States](https://en.wikipedia.org/wiki/List_of_college_towns#College_towns_in_the_United_States) which has been copy and pasted into the file ```university_towns.txt```.
# * From Bureau of Economic Analysis, US Department of Commerce, the [GDP over time](http://www.bea.gov/national/index.htm#gdp) of the United States in current dollars (use the chained value in 2009 dollars), in quarterly intervals, in the file ```gdplev.xls```. For this assignment, only look at GDP data from the first quarter of 2000 onward.
# 
# Each function in this assignment below is worth 10%, with the exception of ```run_ttest()```, which is worth 50%.

# In[2]:

# Use this dictionary to map state names to two letter acronyms
states = {'OH': 'Ohio', 'KY': 'Kentucky', 'AS': 'American Samoa', 'NV': 'Nevada', 'WY': 'Wyoming', 'NA': 'National', 'AL': 'Alabama', 'MD': 'Maryland', 'AK': 'Alaska', 'UT': 'Utah', 'OR': 'Oregon', 'MT': 'Montana', 'IL': 'Illinois', 'TN': 'Tennessee', 'DC': 'District of Columbia', 'VT': 'Vermont', 'ID': 'Idaho', 'AR': 'Arkansas', 'ME': 'Maine', 'WA': 'Washington', 'HI': 'Hawaii', 'WI': 'Wisconsin', 'MI': 'Michigan', 'IN': 'Indiana', 'NJ': 'New Jersey', 'AZ': 'Arizona', 'GU': 'Guam', 'MS': 'Mississippi', 'PR': 'Puerto Rico', 'NC': 'North Carolina', 'TX': 'Texas', 'SD': 'South Dakota', 'MP': 'Northern Mariana Islands', 'IA': 'Iowa', 'MO': 'Missouri', 'CT': 'Connecticut', 'WV': 'West Virginia', 'SC': 'South Carolina', 'LA': 'Louisiana', 'KS': 'Kansas', 'NY': 'New York', 'NE': 'Nebraska', 'OK': 'Oklahoma', 'FL': 'Florida', 'CA': 'California', 'CO': 'Colorado', 'PA': 'Pennsylvania', 'DE': 'Delaware', 'NM': 'New Mexico', 'RI': 'Rhode Island', 'MN': 'Minnesota', 'VI': 'Virgin Islands', 'NH': 'New Hampshire', 'MA': 'Massachusetts', 'GA': 'Georgia', 'ND': 'North Dakota', 'VA': 'Virginia'}


# In[3]:

def get_list_of_university_towns():
    '''Returns a DataFrame of towns and the states they are in from the 
    university_towns.txt list. The format of the DataFrame should be:
    DataFrame( [ ["Michigan", "Ann Arbor"], ["Michigan", "Yipsilanti"] ], 
    columns=["State", "RegionName"]  )
    
    The following cleaning needs to be done:

    1. For "State", removing characters from "[" to the end.
    2. For "RegionName", when applicable, removing every character from " (" to the end.
    3. Depending on how you read the data, you may need to remove newline character '\n'. '''
    
    with open('university_towns.txt', 'r') as file:
        data = []
        state_name = ''
        for line in file:
            if '[edit]' in line:
                state_name = line[:line.index('[')]
            else:
                try:
                    para_index = line.index('(')
                    uni_town = line[:para_index][:-1]
                except ValueError:
                    uni_town = line
                data.append([state_name, uni_town.replace('\n', '')]) 
     
    df = pd.DataFrame(data=data, columns=['State', 'RegionName'])
    return df

get_list_of_university_towns()


# In[4]:

def get_gdps():
    df = pd.read_excel('gdplev.xls')
    df.drop(df.columns[[0, 1, 2, 3, 5, 7]], axis=1, inplace=True)
    df.drop(np.arange(218), axis=0, inplace=True)
    df.columns = ['Quarter', 'GDP']
    df = df.set_index('Quarter')
    df['diff'] = df['GDP'].diff()
    return df

def search_start(diffs):
    for i, diff in enumerate(diffs):
        if diff < 0 and i + 1 < len(diffs) and diffs[i + 1] < 0:
            return i

def get_recession_start():
    '''Returns the year and quarter of the recession start time as a 
    string value in a format such as 2005q3'''
    df = get_gdps()
    ind = search_start(df['diff'])
    return df.iloc[ind].name
    
get_recession_start()


# In[5]:

def search_end(diffs):
    for i, diff in enumerate(diffs):
        if diff > 0 and i + 1 < len(diffs) and diffs[i + 1] > 0:
            return i + 1

def get_recession_end():
    '''Returns the year and quarter of the recession end time as a 
    string value in a format such as 2005q3'''
    df = get_gdps()
    start_index = search_start(df['diff'])
    recession_length = search_end(df.iloc[start_index:]['diff'])
    return df.iloc[start_index + recession_length].name

get_recession_end()


# In[6]:

def get_recession_bottom():
    '''Returns the year and quarter of the recession bottom time as a 
    string value in a format such as 2005q3'''
    df = get_gdps()
    start_index = get_recession_start()
    end_index = get_recession_end()
    df = df[start_index <= df.index]
    df = df[df.index <= end_index]
    return df['GDP'].idxmin()
    
get_recession_bottom()


# In[7]:

from collections import OrderedDict


def chunks(data, length_of_chunk):
    for i in range(0, len(data), length_of_chunk):
        yield data[i:i + length_of_chunk]

def convert(df):
    year = 2000
    quarter = 1
    column_names_dict = OrderedDict()
    for chunk in chunks(df.columns, 3):
        column_names_dict[str(year) + 'q' + str(quarter)] = list(chunk)
        quarter += 1
        if quarter == 5:
            quarter = 1
            year += 1
    
    for quarter, column_names in column_names_dict.items():
        chunk = df[column_names]
        df[quarter] = chunk.mean(axis=1)
        df.drop(df.columns[:len(column_names)], axis=1, inplace=True)
    return df

def convert_housing_data_to_quarters():
    '''Converts the housing data to quarters and returns it as mean 
    values in a dataframe. This dataframe should be a dataframe with
    columns for 2000q1 through 2016q3, and should have a multi-index
    in the shape of ["State","RegionName"].
    
    Note: Quarters are defined in the assignment description, they are
    not arbitrary three month periods.
    
    The resulting dataframe should have 67 columns, and 10,730 rows.
    '''
    df = pd.read_csv('City_Zhvi_AllHomes.csv')
    df = df.drop(df.columns[np.append(np.arange(3,51), 0)], axis=1)
    df['State'] = df['State'].apply(lambda state: states[state])
    df = df.set_index(['State', 'RegionName'])
    return convert(df)

convert_housing_data_to_quarters()


# In[30]:

def decrement_quarter(quarter):
    year, quarter = int(quarter[:4]), int(quarter[5])
    return str(year - 1) + 'q4' if quarter == 1 else str(year) + 'q' + str(quarter - 1)

def run_ttest():
    '''First creates new data showing the decline or growth of housing prices
    between the recession start and the recession bottom. Then runs a ttest
    comparing the university town values to the non-university towns values, 
    return whether the alternative hypothesis (that the two groups are the same)
    is true or not as well as the p-value of the confidence. 
    
    Return the tuple (different, p, better) where different=True if the t-test is
    True at a p<0.01 (we reject the null hypothesis), or different=False if 
    otherwise (we cannot reject the null hypothesis). The variable p should
    be equal to the exact p value returned from scipy.stats.ttest_ind(). The
    value for better should be either "university town" or "non-university town"
    depending on which has a lower mean price ratio (which is equivilent to a
    reduced market loss).'''
    
    df = convert_housing_data_to_quarters()
    uni_towns = list(map(tuple, get_list_of_university_towns().values))
    recession_start = get_recession_start()
    recession_bottom = get_recession_bottom()
    quarter_before_recession = decrement_quarter(recession_start)
    df = df[[quarter_before_recession, recession_bottom]]
    df['ratio'] = df[quarter_before_recession].div(df[recession_bottom])
    uni_df = df.loc[uni_towns].dropna(how='any')
    non_uni_df = df[~df.index.isin(uni_towns)].dropna(how='any')
    ttest_result = ttest_ind(uni_df['ratio'], non_uni_df['ratio'])
    uni_ratio, non_uni_ratio = uni_df['ratio'].mean(), non_uni_df['ratio'].mean()
    return (ttest_result.pvalue < 0.01, ttest_result.pvalue, 
            'university town' if uni_ratio < non_uni_ratio else 'non-university town')


run_ttest()



