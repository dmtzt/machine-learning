import pandas as pd
import constant

data = pd.read_csv(constant.BASICS_PRE_OUTPUT_FILE, header=0)

data['startYear'] = pd.to_numeric(
    data['startYear'], errors='coerce', downcast='integer')

data.sort_values(by=['startYear']).loc[data['startYear'] >=
                                       constant.YEAR].to_csv(constant.BASICS_OUTPUT_FILE, index=False)
