import pandas as pd
import json
from sklearn.model_selection import train_test_split


# 'short' and 'long' contain the same date
with open(f'persona_chat_short.json') as f:
    data = json.load(f)

instances = data['instances']
# transform instances into a dataframe
dicts = []
for instance in instances:
    inst = instance['instance']
    d = {
        'output': inst['response'],
        'context': inst['context'],
        'fact': inst['fact'],
    }
    for k, v in instance['annotations'].items():
        if 'mean_human' in v:
            d[k] = round(v['mean_human'])
        else:
            d[k] = v['majority_human']
    dicts.append(d)

df = pd.DataFrame(dicts)
df.to_csv(f'annotations.csv', index=False)

train, test = train_test_split(df, train_size=25, random_state=42)
train.to_csv(f'annotations-train.csv', index=False)
test.to_csv(f'annotations-test.csv', index=False)
print(f"train size: {len(train)}, test size: {len(test)}")
