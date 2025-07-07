import pandas as pd

def clean_categorical_typos(df):
    job_title_mapping = {
        'data scntist': 'data scientist',
        'data scienist': 'data scientist',
        'dt scientist': 'data scientist',
        'ml engr': 'ml engineer',
        'ml enginer': 'ml engineer',
        'machine learning engr': 'ml engineer',
        'software engr': 'software engineer',
        'softwre engineer': 'software engineer',
        'sofware engneer': 'software engineer'
    }

    df['job_title'] = df['job_title'].str.lower().str.strip().replace(job_title_mapping)
    return df
