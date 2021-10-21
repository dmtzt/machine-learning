import pandas as pd
import constant

header = ['tconst', 'averageRating', 'numVotes']

data_basics = pd.read_csv(constant.BASICS_OUTPUT_FILE, header=0)
data_ratings = pd.read_csv(constant.RATINGS_INPUT_FILE, header=0)

data_ratings.loc[data_ratings['tconst'].isin(
    list(data_basics['tconst']))].to_csv(constant.RATINGS_OUTPUT_FILE, columns=header, index=False)
