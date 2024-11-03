import json
from pandas import DataFrame

def load_mapping(path: str):
    with open(path) as f:
        data = json.load(f)

    mapping_df = DataFrame.from_dict(data, orient='index', columns=['label', 'description'])
    mapping_df.reset_index(inplace=True)
    mapping_df.columns = ['index', 'label', 'description']
    mapping_df['index'] = mapping_df['index'].astype(int)
    
    return {label: i for (label, i) in zip(mapping_df['label'], mapping_df['index'])}
