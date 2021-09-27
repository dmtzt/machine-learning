import pandas as pd
import constant

data = pd.read_csv('dataset.csv', header=0)

with open('analysis.txt', 'w', encoding='utf-8') as f:
    f.write(data.groupby(['startYear']).size().to_string())
    f.write('\n\n')
    f.write(data.groupby(['averageRating']).size().to_string())

# print(data.groupby(['startYear']).size())
# print(data.groupby(['averageRating']).size())
