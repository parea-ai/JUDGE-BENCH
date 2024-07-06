import pandas as pd
import json
from sklearn.model_selection import train_test_split

with open('dices_350_expert.json') as f:
    data = json.load(f)

instances = data['instances']
# transform instances into a dataframe
dicts = []
for instance in instances:
    d = {
        'output': instance['instance']['response'],
        'user_prompt': instance['instance']['user_prompt'],
    }
    for k, v in instance['annotations'].items():
        d[k] = v['majority_human']
    dicts.append(d)

df = pd.DataFrame(dicts)
df.to_csv('annotations.csv', index=False)

train, test = train_test_split(df, train_size=25, random_state=42)
train.to_csv('annotations-train.csv', index=False)
test.to_csv('annotations-test.csv', index=False)
print(f"train size: {len(train)}, test size: {len(test)}")
