from itertools import product

import pandas as pd
import json
from sklearn.model_selection import train_test_split

def create_annotation(dataset_name: str):
    with open(f'roscoe-{dataset_name}.json') as f:
        data = json.load(f)

    instances = data['instances']
    # transform instances into a dataframe
    dicts = []
    for instance in instances:
        instance_str = instance['instance']
        output = instance_str.split('GENERATED RESPONSE:')[-1].strip()
        d = {'output': output}
        if dataset_name.startswith('gsm8k'):
            d['question'] = instance_str.split('CONTEXT:\nQuestion: ')[-1].split('Solution: ')[0].strip()
            d['correct_answer'] = instance_str.split('\n\Solution: ')[-1].split('GENERATED RESPONSE:')[0].strip()
        else:
            d['situation_premise'] = instance_str.split('Situation (Premise): ')[-1].split('Claim (Hypothesis): ')[0].strip()
            d['claim_hypothesis'] = instance_str.split('Claim (Hypothesis): ')[-1].split('Is the Claim supported by the Situation?\n\nCorrect Relationship (Yes or No): ')[0].strip()
            d['correct_relationship'] = instance_str.split('Correct Relationship (Yes or No): ')[-1].split('GENERATED RESPONSE:')[0].strip()

        for k, v in instance['annotations'].items():
            if 'mean_human' in v:
                d[k] = round(v['mean_human'])
            else:
                d[k] = v['majority_human']

        dicts.append(d)

    df = pd.DataFrame(dicts)
    df.to_csv(f'roscoe-{dataset_name}-annotations.csv', index=False)

    train, test = train_test_split(df, train_size=25, random_state=42)
    train.to_csv(f'roscoe-{dataset_name}-annotations-train.csv', index=False)
    test.to_csv(f'roscoe-{dataset_name}-annotations-test.csv', index=False)
    print(f"train size: {len(train)}, test size: {len(test)}")


if __name__ == '__main__':
    for ds, split in product(['cosmos', 'drop', 'esnli', 'gsm8k'], ['overall', 'stepwise']):
        create_annotation(f'{ds}-{split}')

