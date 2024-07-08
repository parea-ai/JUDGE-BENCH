import pandas as pd
import json
from sklearn.model_selection import train_test_split

def create_annotations_for_split(split: str):
    with open(f'toxic_chat_{split}.json') as f:
        data = json.load(f)

    instances = data['instances']
    # transform instances into a dataframe
    dicts = []
    for instance in instances:
        d = {'output': instance['instance']}
        for k, v in instance['annotations'].items():
            d[k] = v['majority_human']
        dicts.append(d)

    df = pd.DataFrame(dicts)
    df.to_csv(f'toxic_chat_{split}-annotations.csv', index=False)

    train, test = train_test_split(df, train_size=25, random_state=42)
    train.to_csv(f'toxic_chat_{split}-annotations-train.csv', index=False)
    test.to_csv(f'toxic_chat_{split}-annotations-test.csv', index=False)
    print(f"train size: {len(train)}, test size: {len(test)}")


if __name__ == '__main__':
    for split in ['train', 'test']:
        create_annotations_for_split(split)
