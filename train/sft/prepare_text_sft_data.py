# this script prepares data for text data sft


import pandas as pd
import json
from random import shuffle, random, seed
from glob import glob


total_volume = pow(2,30) # 1G
train_volume = total_volume / 16 / pow(2,11) # 64M
val_volume = total_volume / 64 / pow(2,11) # 16M
test_volume = (total_volume - train_volume - val_volume) / pow(2,11)

medal_data_dir = '../../../../dataset/medal/medal.csv'
eurlex_data_dir = '../../../../dataset/eurlex/train.eurlex.jsonl'

medal_data = ''
eurlex_data = ''


# read
df = pd.read_csv(medal_data_dir)
for idx, row in df.iterrows():
    medal_data += row['TEXT']
    if len(medal_data) >= total_volume:
        break


with open('../../../../dataset/eurlex/train.eurlex.jsonl') as f:
    for line in f.readlines():
        try:
           line = json.loads(line, strict=False)
           pass
        except:
            continue
        eurlex_data += line['text']
        if len(eurlex_data) >= total_volume:
            break


def cut_and_prepare(s):
    return ['<s>' + s[i : min(i+2048, len(s))] for i in range(0, len(s), 2048)]


dataset = 'medal' # medal or legal

if dataset == 'medal':
    text = medal_data
elif dataset == 'legal':
    text = eurlex_data

seed = 42
r = random
print(len(text))
text = cut_and_prepare(text)[:-1]
print(len(text))
shuffle(text, random=r)

train_data = text[:train_volume]
val_data = text[train_volume:val_volume+train_volume]
test_data = text[val_volume+train_volume:test_volume+val_volume+train_volume]

pd.DataFrame(train_data, columns=['text']).to_csv('./train.csv', index=False)
pd.DataFrame(val_data, columns=['text']).to_csv('./val.csv', index=False)
pd.DataFrame(test_data, columns=['text']).to_csv('./test.csv', index=False)

