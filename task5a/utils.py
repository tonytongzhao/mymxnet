import json
import collections

def load(fi):
    with open(fi) as f:
        data=json.load(f)
    return data



