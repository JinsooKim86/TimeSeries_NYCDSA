import pandas as pd
import numpy as np
import re

def clean_data(df,s='',close_flag=True,volume_flag=True):
    local_copy = df.copy()
    col = local_copy.columns.tolist()
    col[0]='Date'
    local_copy.columns = col
    if volume_flag and close_flag:
        col[-2] = s+'_close'
        col[-1] = s+'_vol'
        local_copy.columns = col
        local_copy = local_copy.iloc[:,[0,5,6]]
    elif close_flag:
        col[-1] = s+'_close'
        local_copy.columns = col
        local_copy = local_copy.iloc[:,[0,4]]
    try:
        local_copy['Date'] = pd.to_datetime(local_copy['Date'],format="%Y-%m-%d")
        local_copy = local_copy.set_index('Date')
    except:
        local_copy['Date'] = pd.to_datetime(local_copy['Date'],format="%m/%d/%y")
        local_copy = local_copy.set_index('Date')
    return local_copy

def clean_texts(df,l):
    local = df.copy()
    for s in l:
        # Replace np.nan by empty string.
        local[s] = local[s].apply(lambda t:'' if pd.isnull(t) else t)
        # Change all words to lower_case.
        local[s] = local[s].apply(lambda t: " ".join(word.lower() for word in t.strip().split()))
        # Remove noncharacter/nonspace.
        local[s] = local[s].apply(lambda t: re.sub('[^\w\s]','', t))
        # Remove extra spaces.
        local[s] = local[s].apply(lambda t: re.sub('\s+',' ', t.strip()))
        # Now each title will be a string of words separated by exactly one space.
    try:
        local['Date'] = pd.to_datetime(local['Date'].apply(lambda s : s[:10]),format= "%Y-%m-%d")
    except:
        local['Date'] = pd.to_datetime(local['Date'])

    local = local.sort_values(by='Date')
    local = local.reset_index(drop=True)

    return local

def create2d(vec,n,nrow,ncol,sep=1,gap=1):
    '''
    vec is a list of length n
    create a 2d array of the given size nrow*ncol which looks like
    [[ vec[0], vec[sep],vec[2*sep],...]
    [ vec[gap], vec[gap+sep],vec[gap+2*sep],...]
    ...]
    '''
    if (nrow-1)*gap+(ncol-1)*sep>=n:
        raise ValueError('sep and gap too large')
    M = max(vec)
    m = min(vec)
    if M == m:
        raise ValueError('nothing happened today')
    vec = [x/(M-m) for x in vec]
    result = []
    for i in range(nrow):
        result_inn = []
        for j in range(ncol):
            result_inn.append(vec[i*gap+j*sep])
        result.append(result_inn)

    return result
