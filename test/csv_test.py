import csv
from itertools import zip_longest

data = {}
data['attention'] = [3,4,5]
data['bttention'] = [6,7,8,9]
csv_rows = zip_longest(*data.values())
with open('hello.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(data.keys())
    writer.writerows(csv_rows)