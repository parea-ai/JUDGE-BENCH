import pandas as pd
import json
from sklearn.model_selection import train_test_split

with open('newsroom.json') as f:
    data = json.load(f)

instances = data['instances']
# transform instances into a dataframe
dicts = []
train_dicts: dict[str, dict[str, str | float]] = dict()  # used to keep track of which summaries were already added
test_dicts = []
for instance in instances:
    instance_string = instance['instance']
    output = instance_string.split('### Generated Summary\n\n')[1].split('\n\n### Source Article\n\n')[0]
    title_article = instance_string.split('\n\n### Source Article\n\n')[1]
    title = title_article.split('\n')[0]
    article = '\n'.join(title_article.split('\n')[1:])

    d = {
        'output': output,
        'title': title,
        'article': article,
    }
    for k, v in instance['annotations'].items():
        d[k] = round(v['mean_human'])
    if article in train_dicts:
        test_dicts.append(d)
    else:
        train_dicts[article] = d
    dicts.append(d)

df = pd.DataFrame(dicts)
train = pd.DataFrame(train_dicts.values())
test = pd.DataFrame(test_dicts)

print(f"train size: {len(train)}, test size: {len(test)}")

train.to_csv('annotations-train.csv', index=False)
test.to_csv('annotations-test.csv', index=False)
df.to_csv('annotations.csv', index=False)
