
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-data-analysis/resources/0dhYG) course resource._
# 
# ---

# # Assignment 3 - More Pandas
# This assignment requires more individual learning then the last one did - you are encouraged to check out the [pandas documentation](http://pandas.pydata.org/pandas-docs/stable/) to find functions or methods you might not have used yet, or ask questions on [Stack Overflow](http://stackoverflow.com/) and tag them as pandas and python related. And of course, the discussion forums are open for interaction with your peers and the course staff.

# ### Question 1 (20%)
# Load the energy data from the file `Energy Indicators.xls`, which is a list of indicators of [energy supply and renewable electricity production](Energy%20Indicators.xls) from the [United Nations](http://unstats.un.org/unsd/environment/excel_file_tables/2013/Energy%20Indicators.xls) for the year 2013, and should be put into a DataFrame with the variable name of **energy**.
# 
# Keep in mind that this is an Excel file, and not a comma separated values file. Also, make sure to exclude the footer and header information from the datafile. The first two columns are unneccessary, so you should get rid of them, and you should change the column labels so that the columns are:
# 
# `['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']`
# 
# Convert `Energy Supply` to gigajoules (there are 1,000,000 gigajoules in a petajoule). For all countries which have missing data (e.g. data with "...") make sure this is reflected as `np.NaN` values.
# 
# Rename the following list of countries (for use in later questions):
# 
# ```"Republic of Korea": "South Korea",
# "United States of America": "United States",
# "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
# "China, Hong Kong Special Administrative Region": "Hong Kong"```
# 
# There are also several countries with numbers and/or parenthesis in their name. Be sure to remove these, 
# 
# e.g. 
# 
# `'Bolivia (Plurinational State of)'` should be `'Bolivia'`, 
# 
# `'Switzerland17'` should be `'Switzerland'`.
# 
# <br>
# 
# Next, load the GDP data from the file `world_bank.csv`, which is a csv containing countries' GDP from 1960 to 2015 from [World Bank](http://data.worldbank.org/indicator/NY.GDP.MKTP.CD). Call this DataFrame **GDP**. 
# 
# Make sure to skip the header, and rename the following list of countries:
# 
# ```"Korea, Rep.": "South Korea", 
# "Iran, Islamic Rep.": "Iran",
# "Hong Kong SAR, China": "Hong Kong"```
# 
# <br>
# 
# Finally, load the [Sciamgo Journal and Country Rank data for Energy Engineering and Power Technology](http://www.scimagojr.com/countryrank.php?category=2102) from the file `scimagojr-3.xlsx`, which ranks countries based on their journal contributions in the aforementioned area. Call this DataFrame **ScimEn**.
# 
# Join the three datasets: GDP, Energy, and ScimEn into a new dataset (using the intersection of country names). Use only the last 10 years (2006-2015) of GDP data and only the top 15 countries by Scimagojr 'Rank' (Rank 1 through 15). 
# 
# The index of this DataFrame should be the name of the country, and the columns should be ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations',
#        'Citations per document', 'H index', 'Energy Supply',
#        'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008',
#        '2009', '2010', '2011', '2012', '2013', '2014', '2015'].
# 
# *This function should return a DataFrame with 20 columns and 15 entries.*

# In[35]:

import pandas as pd
import numpy as np
from string import digits

def insert_nans(row):
    if row['Energy Supply'] == '...':
        row['Energy Supply'] = np.NaN
    if row['Energy Supply per Capita'] == '...':
        row['Energy Supply per Capita'] = np.NaN
    return row

def convert_to_giga(row):
    row['Energy Supply'] *= 1000000
    return row

def remove_digits(s):
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    return res

def remove_part_in_paranthesis(s):
    para_index = s.find('(')
    return s[:(para_index - 1)] if para_index != -1 else s

def rename_countries(row):
    countries_to_rename = {"Republic of Korea": "South Korea",
                        "United States of America": "United States",
                        "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
                        "China, Hong Kong Special Administrative Region": "Hong Kong"}
    row['Country'] = remove_digits(row['Country'])
    row['Country'] = remove_part_in_paranthesis(row['Country'])
    row['Country'] = countries_to_rename.get(row['Country'], row['Country'])
    return row
    
def make_changes(row):
    row = insert_nans(row)
    row = convert_to_giga(row)
    row = rename_countries(row)
    return row

def rename_countries_gdp(row):
    countries_to_rename = {"Korea, Rep.": "South Korea", "Iran, Islamic Rep.": "Iran", "Hong Kong SAR, China": "Hong Kong"}
    row['Country'] = countries_to_rename.get(row['Country'], row['Country'])
    return row

def get_energy():
    col_names = ['Country', 'Energy Supply', 'Energy Supply per Capita', '% Renewable']
    energy = pd.read_excel('Energy Indicators.xls')
    energy = energy.drop(energy.index[:16]).drop(energy.index[243:]).drop(energy.columns[:2], axis=1)
    energy.columns = col_names
    energy = energy.apply(make_changes, axis=1)
    return energy

def get_GDP():
    GDP = pd.read_csv('world_bank.csv')
    GDP = GDP.drop(GDP.index[:4]).rename(columns={'Data Source': 'Country'})
    GDP = GDP.apply(rename_countries_gdp, axis=1)
    return GDP
    
def get_ScimEn():
    ScimEn = pd.read_excel('scimagojr-3.xlsx')
    return ScimEn
    
def get_data():
    return get_energy(), get_GDP(), get_ScimEn()

def answer_one():
    energy, GDP, ScimEn = get_data()
    GDP_columns = ['Country']
    GDP_columns.extend(list(GDP.columns[-10:]))
    cols = ['Rank', 'Documents', 'Citable documents', 'Citations', 'Self-citations', 'Citations per document', 'H index', 
            'Energy Supply', 'Energy Supply per Capita', '% Renewable', '2006', '2007', '2008', '2009', '2010', '2011',
            '2012', '2013', '2014', '2015']
    merged = pd.merge(ScimEn[ScimEn['Rank'] < 16], energy, how='inner', left_on='Country', right_on='Country')
    final = pd.merge(merged, GDP[GDP_columns], how='inner', left_on ='Country', right_on='Country')
    final = final.set_index('Country')
    final.columns = cols
    return final

answer_one()


# ### Question 2 (6.6%)
# The previous question joined three datasets then reduced this to just the top 15 entries. When you joined the datasets, but before you reduced this to the top 15 items, how many entries did you lose?
# 
# *This function should return a single number.*

# In[129]:

get_ipython().run_cell_magic('HTML', '', '<svg width="800" height="300">\n  <circle cx="150" cy="180" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="blue" />\n  <circle cx="200" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="red" />\n  <circle cx="100" cy="100" r="80" fill-opacity="0.2" stroke="black" stroke-width="2" fill="green" />\n  <line x1="150" y1="125" x2="300" y2="150" stroke="black" stroke-width="2" fill="black" stroke-dasharray="5,3"/>\n  <text  x="300" y="165" font-family="Verdana" font-size="35">Everything but this!</text>\n</svg>')


# In[36]:

def answer_two():
    energy, GDP, ScimEn = get_data()
    Top15 = answer_one()
    GDP_columns = ['Country']
    GDP_columns.extend(list(GDP.columns[-10:]))
    merged = pd.merge(ScimEn, energy, how='outer', left_on='Country', right_on='Country')
    final = pd.merge(merged, GDP[GDP_columns], how='outer', left_on ='Country', right_on='Country')
    final = final.set_index('Country')
    return len(final) - len(Top15)

answer_two()


# <br>
# 
# Answer the following questions in the context of only the top 15 countries by Scimagojr Rank (aka the DataFrame returned by `answer_one()`)

# ### Question 3 (6.6%)
# What is the average GDP over the last 10 years for each country? (exclude missing values from this calculation.)
# 
# *This function should return a Series named `avgGDP` with 15 countries and their average GDP sorted in descending order.*

# In[5]:

import numpy as np

def avg(values):
    sum = 0
    length = 0
    for val in values:
        if np.isnan(val): 
            continue
        sum += val
        length += 1
    return sum / length
    
def fun(row):
    columns = list(map(str, range(2006, 2016)))
    gdps = row[columns]
    avgs = avg(gdps)
    return avgs

def answer_three():
    Top15 = answer_one()
    series = Top15.apply(fun, axis=1)
    series.name = 'avgGDP'
    series.sort(ascending=False, inplace=True)
    return series
    
answer_three()


# ### Question 4 (6.6%)
# By how much had the GDP changed over the 10 year span for the country with the 6th largest average GDP?
# 
# *This function should return a single number.*

# In[6]:

def answer_four():
    Top15 = answer_one()
    ans3 = answer_three()
    country = ans3.index[5]
    return Top15.loc[country]['2015'] - Top15.loc[country]['2006']

answer_four()


# ### Question 5 (6.6%)
# What is the mean `Energy Supply per Capita`?
# 
# *This function should return a single number.*

# In[7]:

def answer_five():
    Top15 = answer_one()
    return np.mean(Top15['Energy Supply per Capita'])

answer_five()


# ### Question 6 (6.6%)
# What country has the maximum % Renewable and what is the percentage?
# 
# *This function should return a tuple with the name of the country and the percentage.*

# In[8]:

def answer_six():
    Top15 = answer_one()
    max_renewable = np.max(Top15['% Renewable'])
    country = Top15[Top15['% Renewable'] == max_renewable].index[0]
    return (country, max_renewable)

answer_six()


# ### Question 7 (6.6%)
# Create a new column that is the ratio of Self-Citations to Total Citations. 
# What is the maximum value for this new column, and what country has the highest ratio?
# 
# *This function should return a tuple with the name of the country and the ratio.*

# In[9]:

def answer_seven():
    Top15 = answer_one()
    Top15['Ratio'] = Top15['Self-citations'] / Top15['Citations']
    max_ratio = np.max(Top15['Ratio'])
    country = Top15[Top15['Ratio'] == max_ratio].index[0]
    return (country, max_ratio)

answer_seven()


# ### Question 8 (6.6%)
# 
# Create a column that estimates the population using Energy Supply and Energy Supply per capita. 
# What is the third most populous country according to this estimate?
# 
# *This function should return a single string value.*

# In[10]:

def answer_eight():
    Top15 = answer_one()
    Top15['Population'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    pops = Top15['Population'].copy()
    pops.sort(ascending=False, inplace=True)
    return pops.index[2]
    
answer_eight()


# ### Question 9 (6.6%)
# Create a column that estimates the number of citable documents per person. 
# What is the correlation between the number of citable documents per capita and the energy supply per capita? Use the `.corr()` method, (Pearson's correlation).
# 
# *This function should return a single number.*
# 
# *(Optional: Use the built-in function `plot9()` to visualize the relationship between Energy Supply per Capita vs. Citable docs per Capita)*

# In[11]:

def answer_nine():
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    return Top15['Citable docs per Capita'].corr(Top15['Energy Supply per Capita'])

answer_nine()


# In[12]:

def plot9():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    Top15['Citable docs per Capita'] = Top15['Citable documents'] / Top15['PopEst']
    Top15.plot(x='Citable docs per Capita', y='Energy Supply per Capita', kind='scatter', xlim=[0, 0.0006])


# ### Question 10 (6.6%)
# Create a new column with a 1 if the country's % Renewable value is at or above the median for all countries in the top 15, and a 0 if the country's % Renewable value is below the median.
# 
# *This function should return a series named `HighRenew` whose index is the country name sorted in ascending order of rank.*

# In[13]:

def is_above_median(row, median):
    row['% Renewable above median'] = int(row['% Renewable'] >= median)
    return row

def answer_ten():
    Top15 = answer_one()
    median = np.median(Top15['% Renewable'])
    Top15 = Top15.apply(is_above_median, axis=1, args=(median, ))
    result = Top15['% Renewable above median'].copy()
    result.name = 'HighRenew'
    return result

answer_ten()


# ### Question 11 (6.6%)
# Use the following dictionary to group the Countries by Continent, then create a dateframe that displays the sample size (the number of countries in each continent bin), and the sum, mean, and std deviation for the estimated population of each country.
# 
# ```python
# ContinentDict  = {'China':'Asia', 
#                   'United States':'North America', 
#                   'Japan':'Asia', 
#                   'United Kingdom':'Europe', 
#                   'Russian Federation':'Europe', 
#                   'Canada':'North America', 
#                   'Germany':'Europe', 
#                   'India':'Asia',
#                   'France':'Europe', 
#                   'South Korea':'Asia', 
#                   'Italy':'Europe', 
#                   'Spain':'Europe', 
#                   'Iran':'Asia',
#                   'Australia':'Australia', 
#                   'Brazil':'South America'}
# ```
# 
# *This function should return a DataFrame with index named Continent `['Asia', 'Australia', 'Europe', 'North America', 'South America']` and columns `['size', 'sum', 'mean', 'std']`*

# In[14]:

def answer_eleven():
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    return Top15.groupby(ContinentDict.get)['PopEst'].agg({'size': np.count_nonzero, 'sum': np.sum, 'mean': np.mean, 'std': np.std})

answer_eleven()


# ### Question 12 (6.6%)
# Cut % Renewable into 5 bins. Group Top15 by the Continent, as well as these new % Renewable bins. How many countries are in each of these groups?
# 
# *This function should return a __Series__ with a MultiIndex of `Continent`, then the bins for `% Renewable`. Do not include groups with no countries.*

# In[33]:

def answer_twelve():
    ContinentDict  = {'China':'Asia', 
                  'United States':'North America', 
                  'Japan':'Asia', 
                  'United Kingdom':'Europe', 
                  'Russian Federation':'Europe', 
                  'Canada':'North America', 
                  'Germany':'Europe', 
                  'India':'Asia',
                  'France':'Europe', 
                  'South Korea':'Asia', 
                  'Italy':'Europe', 
                  'Spain':'Europe', 
                  'Iran':'Asia',
                  'Australia':'Australia', 
                  'Brazil':'South America'}
    Top15 = answer_one()
    Top15['Bin'] = pd.cut(Top15['% Renewable'], 5)
    Top15['Continent'] = [ContinentDict[index] for index in Top15.index]
    index_tuples = []
    data = []
    for v, frame in Top15.groupby(['Continent', 'Bin']):
        index_tuples.append(v)
        data.append(len(frame))
    result = pd.Series(data, index=pd.MultiIndex.from_tuples(index_tuples, names=['Continent', '% Renewable']))
    return result

answer_twelve()


# ### Question 13 (6.6%)
# Convert the Population Estimate series to a string with thousands separator (using commas). Do not round the results.
# 
# e.g. 317615384.61538464 -> 317,615,384.61538464
# 
# *This function should return a Series `PopEst` whose index is the country name and whose values are the population estimate string.*

# In[17]:

def answer_thirteen():
    Top15 = answer_one()
    Top15['PopEst'] = Top15['Energy Supply'] / Top15['Energy Supply per Capita']
    return Top15['PopEst'].map('{:,}'.format)

answer_thirteen()


# ### Optional
# 
# Use the built in function `plot_optional()` to see an example visualization.

# In[ ]:

def plot_optional():
    import matplotlib as plt
    get_ipython().magic('matplotlib inline')
    Top15 = answer_one()
    ax = Top15.plot(x='Rank', y='% Renewable', kind='scatter', 
                    c=['#e41a1c','#377eb8','#e41a1c','#4daf4a','#4daf4a','#377eb8','#4daf4a','#e41a1c',
                       '#4daf4a','#e41a1c','#4daf4a','#4daf4a','#e41a1c','#dede00','#ff7f00'], 
                    xticks=range(1,16), s=6*Top15['2014']/10**10, alpha=.75, figsize=[16,6]);

    for i, txt in enumerate(Top15.index):
        ax.annotate(txt, [Top15['Rank'][i], Top15['% Renewable'][i]], ha='center')

    print("This is an example of a visualization that can be created to help understand the data. This is a bubble chart showing % Renewable vs. Rank. The size of the bubble corresponds to the countries' 2014 GDP, and the color corresponds to the continent.")


# In[ ]:

#plot_optional() # Be sure to comment out plot_optional() before submitting the assignment!

