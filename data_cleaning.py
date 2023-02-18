# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 12:48:43 2023

@author: Dell
"""

import pandas as pd

df = pd.read_csv('C:/Users/Dell/OneDrive/Desktop/Udemy Course/Projects/DS Project - Salary Prediction/glassdoor_jobs.csv')


# Cleaning Salary
# remove rows with '-1' value
df = df[ df['Salary Estimate'] != '-1' ]

# getting only numbers from salary range
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
remove_KD = salary.apply(lambda x: x.replace('K','').replace('$',''))

df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0 )
df['employer_provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0 )

min_hr = remove_KD.apply(lambda x: x.lower().replace('per hour', '').replace('employer provided salary:', ''))

# finding mean for range
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2


# Cleaning Company Name

# removing rating from company name
df['Company_text'] = df.apply( lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis = 1)


# State Field
df['jobs_state'] = df['Location'].apply(lambda x: x.split(',')[1])
df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis = 1)


# Comapany age
df['comapny_age'] = df['Founded'].apply(lambda x: x if x < 1 else 2023-x)


# Cleaning Job Description --> For keywords like, python, etc.

# Python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)

# R Studio
df['r_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)

# Excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)

# AWS
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)

# Spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)

# Dropping first unnamed index column
df_out = df.drop(['Unnamed: 0'], axis = 1)

# Covert final df into csv file
df_out.to_csv('Salary_Data_Cleaned.csv', index = False)
