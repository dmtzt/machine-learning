import pandas as pd
import constant

header = ['tconst', 'titleType', 'primaryTitle', 'originalTitle',
          'isAdult', 'startYear', 'runtimeMinutes', 'genres']

data = pd.read_csv(constant.BASICS_INPUT_FILE, sep='\t', header=0)

data.loc[(data['titleType'] == 'movie')].loc[(data['startYear'] != '"\"N')].loc[(data['runtimeMinutes'] != '"\"N')].to_csv(
    constant.BASICS_PRE_OUTPUT_FILE, columns=header, index=False)
