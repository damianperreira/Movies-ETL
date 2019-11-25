#!/usr/bin/env python
# coding: utf-8

# In[76]:


import json
import pandas as pd
import numpy as np
import re
import sys
import os


# In[77]:


wikifile = json.load(open('wikipedia-movies.json', mode='r'))
kaggle1 = pd.read_csv('the-movies-dataset/movies_metadata.csv',low_memory=False)
kaggle2 = pd.read_csv('the-movies-dataset/ratings.csv',low_memory=False)

def etl_get_files(wikifile,kaggle1,kaggle2):
    try:
        wiki_movies = wikifile
    except IOError:
        print("file not found")
    kaggle_data = kaggle1
    rating = kaggle2
    
    return wiki_movies, kaggle_data, rating


# In[78]:


wiki_movies_raw, kaggle_metadata, ratings = etl_get_files(wikifile,kaggle1,kaggle2)


# In[79]:


# Wiki Movies raw create a filter expression for only movies with a director and an IMDb link...etc Applying filters
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]


# In[80]:


# Create our function to clean our movie data
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie


# In[81]:


# Handling Alternative Titles

# Step 1: Make an empty dict to hold all of the alternative titles.
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    return movie


# In[82]:


# Step 2: Loop through a list of all alternative title keys.
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    return movie


# In[83]:


### Create a Function to Clean the Data, Parts 1-2.

# Columns with slightly different names, consolidate columns with the same data into one column.
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie


# In[84]:


# We can make a list of cleaned movies with a list comprehension and recreate wiki_movies_df
clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)


# In[85]:


# Set wiki_movies_df to be the DataFrame created from clean_movies, and print out a list of the columns.
wiki_movies_df = pd.DataFrame(clean_movies)
#sorted(wiki_movies_df.columns.tolist()) #for inspection


# In[86]:


# Extract IMDb ID using RegEx
# Create a new field named imdb_id in the wiki_movies_df dataframe 
# from a group of characters in the imdb_link field that start with “tt” and has seven digits.
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')


# In[87]:


#Drop any duplicates of IMDb IDs by using the drop_duplicates() method

wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
#print(len(wiki_movies_df)) #for inspection

wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
#print(len(wiki_movies_df)) #for inspection
#wiki_movies_df.head() #for inspection


# In[88]:


# Use the list of columns we want to keep and select from the wiki_movies_df DataFrame
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[89]:


# start on the box office data, which should give us code that we can reuse and tweak for the budget data
# Get only rows where box office data is defined
box_office = wiki_movies_df['Box office'].dropna() 

# check the number of data points that exist after you drop data
#len(box_office) #for inspection


# In[90]:


# Use a simple space as our joining character 
# Apply the join() function only when our data points are lists
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[91]:


# Since box office data is written in: “$123.4 million” (or billion), and “$123,456,789.” 
# We’re going to build a regular expression for each form, and then see what forms are left over.

# Form 1
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
#box_office.str.contains(form_one, flags=re.IGNORECASE, na=False)


# In[92]:


# Form 2
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
#box_office.str.contains(form_two, flags=re.IGNORECASE, na=False).sum()


# In[93]:


# Compare values in above forms
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)


# In[94]:


#box_office[~matches_form_one & ~matches_form_two] #inspect


# In[95]:


# box_office.str.extract(f'(({form_one}|{form_two}))') #inspect


# In[96]:


#  function to turn the extracted values into a numeric value. We’ll call it parse_dollars
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[97]:


# extract the values from box_office using str.extract. 
# Then apply parse_dollars to the first column in the DataFrame returned by str.extract
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[98]:


# We no longer need the Box Office column, so drop it
wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[99]:


# Parse budget data
# Create a budget variable 
budget = wiki_movies_df['Budget'].dropna()


# In[100]:


# Convert any lists to strings:
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[101]:


# remove any values between a dollar sign and a hyphen (for budgets given in ranges)
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[102]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
#budget[~matches_form_one & ~matches_form_two] #inspection


# In[103]:


# Next step is to remove the citation references
budget = budget.str.replace(r'\[\d+\]\s*', '')
#budget[~matches_form_one & ~matches_form_two] #inspect


# In[104]:


# parse the budget values
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[105]:


# drop the original Budget column
wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[106]:


# Parse release date
# make a variable that holds the non-null values of Release date in the DataFrame, converting lists to strings
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[107]:


# 4 forms to parse date
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[108]:


# use the built-in to_datetime() method in Pandas. 
# Since there are different date formats, set the infer_datetime_format option to True
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[109]:


# Parse Running Time
# make a variable that holds the non-null values of Release date in the DataFrame, converting lists to strings
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[110]:


# match all of the hour + minute patterns
# extract digits, and we want to allow for both possible patterns

running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[111]:


# turn the empty strings into Not a Number (NaN)
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[112]:


# convert the hour capture groups and minute capture groups to minutes if the pure minutes capture group is zero
# save the output to wiki_movies_df
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[113]:


# drop running time from dataset 
wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[114]:


#Clean kaggle data
# remove the bad data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]
# keep rows where the adult column is False, and then drop the adult column
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
#kaggle_metadata #for inspection


# In[115]:


# convert
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[116]:


# if there’s any data that can’t be converted to numbers
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[117]:


# convert release date to datetime
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[118]:


# Assign to timestamp column
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[119]:


# Merging the data from Wikipedia and Kaggle
# Print out a list of the columns so we can identify which ones are redundant
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
#movies_df #inspect the merged dataframe


# In[120]:


# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle             Drop Wikipedia
# running_time             runtime                  Keep Kaggle; fill in zeros with Wikipedia data
# budget_wiki              budget_kaggle            Keep Kaggle; fill in zeros with Wikipedia data.
# box_office               revenue                  Keep Kaggle; fill in zeros with Wikipedia data.
# release_date_wiki        release_date_kaggle      Drop Wikipedia
# Language                 original_language        Drop Wikipedia
# Production company(s)    production_companies     Drop Wikipedia


# In[121]:


# drop title_wiki 
movies_df.drop('title_wiki', axis=1, inplace=True)


# In[122]:


# drop outlier using index
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[123]:


# Resolution: drop wikipedia

movies_df.drop('release_date_wiki', axis=1, inplace=True)


# In[124]:


# Resolution: drop wikipedia
movies_df.drop('Language', axis=1, inplace=True)


# In[125]:


# Resolution: drop wikipedia since Kaggle data is more consistent
movies_df.drop('Production company(s)', axis=1, inplace=True)


# In[126]:


# Putting it all together
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[127]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
#movies_df.head() #for inspection


# In[128]:


# reorder columns
movies_df = movies_df[['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


# In[129]:


# rename columns for consistency
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# In[130]:


# on the “movieId” and “rating” columns and take the count for each group
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()


# In[131]:


# rename userid column to count
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1) 


# In[132]:


# pivot this data so that movieId is the index, 
# the columns will be all the rating values, and the rows will be the counts for each rating value
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[133]:


# prepend rating_ to each column with a list comprehension
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[134]:


# use a left merge
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[135]:


# fix missing values with zero
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[136]:


# Import Modules
from sqlalchemy import create_engine
import psycopg2


# In[137]:


# create database engine
# Use the following connection string for Postgresql:
# "postgres://[user]:[password]@[location]:[port]/[database]"

from config import db_password


# In[138]:


# connection string for local server
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5432/movie_data"


# In[139]:


# creating the database engine
engine = create_engine(db_string)


# In[140]:


# Challenge: save the movies_df DataFrame to a SQL table, but first truncate existing data,  
# then If table exists, insert data. Create if does not exist.
engine.execute("TRUNCATE TABLE movies") 
#movies_df.to_sql(name='movies', con=engine)
movies_df.to_sql(name='movies', con=engine, if_exists='append')


# In[71]:


#Same with ratings table, truncate and load data, create table if does not exist
engine.execute("TRUNCATE TABLE ratings") 
# create a variable for the number of rows imported
rows_imported = 0

# make a for loop and append the chunks of data to the new rows to the target SQL table
for data in pd.read_csv(f'the-movies-dataset/ratings.csv', chunksize=1000000):

    # print out the range of rows that are being imported
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')

    data.to_sql(name='ratings', con=engine, if_exists='append')

    # increment the number of rows imported by the size of 'data'
    rows_imported += len(data)

    # print that the rows have finished importing
    print('Done.')


# In[ ]:




