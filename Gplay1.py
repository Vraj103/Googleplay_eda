# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:32:09 2021

@author: Lenovo
"""

"""This is my second Exploratory Data Analysis Project which consists of 2 datasets.
First dataset being 'apps' consisting of  9596 non-null apps.
Second and last being user-reviews consisting of real reviews from the apps used for
sentiment analysis which is a part of NLP (so we'll cover that some other time)."""

#Importing libraries for use
import pandas as pd
import numpy as np
import seaborn as sns
import plotly
import matplotlib.pyplot as plt
                                         
# Read in dataset
apps_with_duplicates = pd.read_csv("apps.csv")

# Drop duplicates from apps_with_duplicates
apps = apps_with_duplicates.drop_duplicates()

#Print the total number of apps
print('Total number of apps in the dataset = ', len(apps))

#random sample of 5 rows
n=5
print(apps.sample(n))

"""To clean the data here, we remove the characters like ',' '$' and '+' 
which makes it impossible to do further analysis as the data can't be
converted into float from string. """

# List of characters to remove
chars_to_remove = ['+', ',', '$'," "]
# List of column names to clean
cols_to_clean = ['Installs', 'Price']

# Loop for each column in cols_to_clean
for col in cols_to_clean:
    # Loop for each char in chars_to_remove
    for char in chars_to_remove:
        # Replace the character with an empty string
        apps[col] = apps[col].apply(lambda x: x.replace(char, ''))  

# Print a summary of the apps dataframe
print(apps.info())

"""Thus, after cleaning the data we convert the columns required to the float
data type from string data type which further makes it easy to analyze.""" 

# Convert Installs to float data type
apps['Installs'] = apps["Installs"].astype(float)

# Convert Price to float data type
apps['Price'] = apps["Price"].astype(float)

# Checking dtypes of the apps dataframe
print(apps.dtypes)

"""Now, we discover the types of app categories by a graph using plotly.
And in conclusion we conclude that "Family" type category acquire most number 
of applications."""

plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

# Print the total number of unique categories
num_categories = len(apps['Category'].unique())
print('Number of categories = ', num_categories)

# Count the number of apps in each 'Category'. 
num_apps_in_category = apps['Category'].value_counts()

# Sort num_apps_in_category in descending order based on the count of apps in each category
sorted_num_apps_in_category = num_apps_in_category.sort_values(ascending = False)

data = [go.Bar(
        x = num_apps_in_category.index, # index = category name
        y = num_apps_in_category.values, # value = count
)]

plotly.offline.plot(data)

"""Now, we find out the average rating of each app and in conclusion make a 
histogram of it using plotly."""


# Average rating of apps
avg_app_rating = apps["Rating"].mean()
print('Average app rating = ', avg_app_rating)

# Distribution of apps according to their ratings
data = [go.Histogram(
        x = apps['Rating']
)]

# Vertical dashed line to indicate the average app rating
layout = {'shapes': [{
              'type' :'line',
              'x0': avg_app_rating,
              'y0': 0,
              'x1': avg_app_rating,
              'y1': 1000,
              'line': { 'dash': 'dashdot'}
          }]
          }

plotly.offline.iplot({'data': data, 'layout': layout})

"""Now, in this segment we take analysis of Size vs Rating of those apps with 
the with size more than or equal to 250 MB and later Price vs Rating of the same 
apps."""

"""And when we conclude, the Following results come out :
    1 : The average rating lies between 4.1-4.2 and most apps are 10-20 MB.
    2 : The average price lies between 0-10 MB."""

sns.set_style("darkgrid")
import warnings
warnings.filterwarnings("ignore")

# Select rows where both 'Rating' and 'Size' values are present (ie. the two values are not null)
apps_with_size_and_rating_present = apps[(~apps["Rating"].isnull()) & (~apps["Size"].isnull())]

# Subset for categories with at least 250 apps
large_categories = apps_with_size_and_rating_present.groupby(["Category"]).filter(lambda x: len(x) >= 250)

# Plot size vs. rating
plt1 = sns.jointplot(x = large_categories["Size"], y = large_categories["Rating"])

# Select apps whose 'Type' is 'Paid'
paid_apps = apps_with_size_and_rating_present[apps_with_size_and_rating_present["Type"] == 'Paid']

# Plot price vs. rating
plt2 = sns.jointplot(x = paid_apps["Price"], y = paid_apps["Rating"])


"""Now, we see the relation between app category and app price.
And based on that we filter out the "JUNK apps" which are useless 
and unnessecerily over-priced."""

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Select a few popular app categories
popular_app_cats = apps[apps.Category.isin(['GAME', 'FAMILY', 'PHOTOGRAPHY',
                                            'MEDICAL', 'TOOLS', 'FINANCE',
                                            'LIFESTYLE','BUSINESS'])]

# Examine the price trend by plotting Price vs Category
ax = sns.stripplot(x = popular_app_cats["Price"], y = popular_app_cats["Category"], jitter=True, linewidth=1)
ax.set_title('App pricing trend across categories')

# Apps whose Price is greater than 200
apps_above_200 = apps[apps["Price"] > 200]
apps_above_200[['Category', 'App', 'Price']]

"""Filtering out junk apps (apps<100$) """

# Select apps priced below $100
apps_under_100 = popular_app_cats[popular_app_cats["Price"] < 100]

fig, ax = plt.subplots()
fig.set_size_inches(15, 8)

# Examine price vs category with the authentic apps (apps_under_100)
ax = sns.stripplot(x = "Price", y = "Category", data = apps_under_100, jitter = True, linewidth = 1)
ax.set_title('App pricing trend across categories after filtering for junk apps')

"""Now, we see the popularity of Paid apps vs Free apps.
In conclusion :
    1 : Paid apps have 10M downloads till date.
    2 : Free apps have 1B download till date."""
    
trace0 = go.Box(
    # Data for paid apps
    y = apps[apps['Type'] == "Paid"]['Installs'],
    name = 'Paid'
)

trace1 = go.Box(
    # Data for free apps
    y = apps[apps['Type'] == "Free"]['Installs'],
    name = 'Free'
)

layout = go.Layout(
    title = "Number of downloads of paid apps vs. free apps",
    yaxis = dict(title = "Log number of downloads",
                type = 'log',
                autorange = True)
)

# Add trace0 and trace1 to a list for plotting
data = [trace0, trace1]
plotly.offline.plot({'data': data, 'layout': layout})

"""For the last segment, we do this sentiment review which ranges between y-axis 
on 1 to -1 where 1 being good comments while -1 being extremely harsh.""" 


# Load user_reviews.csv
reviews_df = pd.read_csv("user_reviews.csv")

# Join the two dataframes
merged_df = pd.merge(apps, reviews_df, on = "App")

# Drop NA values from Sentiment and Review columns
merged_df = merged_df.dropna(subset = ['Sentiment', 'Review'])

sns.set_style('ticks')
fig, ax = plt.subplots()
fig.set_size_inches(11, 8)

# User review sentiment polarity for paid vs. free apps
ax = sns.boxplot(x = "Type" , y = "Sentiment_Polarity"  , data = merged_df )
ax.set_title('Sentiment Polarity Distribution')



 