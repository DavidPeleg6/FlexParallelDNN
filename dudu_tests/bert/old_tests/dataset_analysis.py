import codecs

batch = 512
seq_length = 20
with codecs.open('../europarl30k.fr.txt', mode='r', encoding='utf-8') as file:
    lines = file.read()

words = lines.split()
print(f'word count:{len(words)}, sequence length: {seq_length}, batch: {batch} '
      f'\n---> words in batch: {seq_length * batch}'
      f'\n---> iterations in epoch: {len(words) / (seq_length * batch)}')

