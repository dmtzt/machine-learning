import pandas as pd
import constant

data_basics = pd.read_csv(constant.BASICS_OUTPUT_FILE, header=0)
data_ratings = pd.read_csv(constant.RATINGS_INPUT_FILE, header=0)

basics_ratings = pd.merge(data_basics, data_ratings,
                          how='inner').to_csv('test2.csv', index=False)
