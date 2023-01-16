# import pandas as pd
from statistics import mean

with open('strategy_out.txt', 'r') as file:
    ent_file = file.read()
with open('worker_strategy_out.txt', 'r') as file:
    ent_file2 = file.read()
# df = pd.read_csv('strategy_out.txt', delimiter=',')
# print(df.head())
throughputs = [float(line.split()[3]) for line in ent_file.split(',') if 'THROUGHPUT' in line]
throughputs += [float(line.split()[3]) for line in ent_file2.split(',') if 'THROUGHPUT' in line]
print('mean throughput with custom splits: {}'.format(mean(throughputs)))

with open('no_strategy_out.txt', 'r') as file:
    ent_file = file.read()
with open('worker_no_strategy_out.txt', 'r') as file:
    ent_file2 = file.read()
# df = pd.read_csv('strategy_out.txt', delimiter=',')
# print(df.head())
temp = [line.split()[3] for line in ent_file.split(',') if 'THROUGHPUT' in line]
temp.remove('samples/s')
throughputs = list(map(lambda x: float(x), temp))
throughputs += [float(line.split()[3]) for line in ent_file2.split(',') if 'THROUGHPUT' in line]
print('mean throughput with vanilla data parallel: {}'.format(mean(throughputs)))
