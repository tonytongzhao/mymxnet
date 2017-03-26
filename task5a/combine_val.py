import sys, collections, json
from utils import read_content, mesh_mapping 
def combine(val, res, mapping):
    data=read_content(val)
    mesh_map, mesh_rev_map=mesh_mapping(mapping)
    print len(mesh_map), len(mesh_rev_map)
    resdict=collections.defaultdict(list)
    res=read_content(res)
    for i in res['documents']:
        resdict[i['pmid']]=[mesh_rev_map[k] for k in i['labels']]

    for i in data['documents']:
        i['meshMajor']=resdict[i['pmid']]

    with open(val.split('.json')[0]+'_res.json', 'w') as outfile:
        json.dump(data, outfile)


if __name__=='__main__':
    val=sys.argv[1]
    res=sys.argv[2]
    mapping=sys.argv[3]
    combine(val, res, mapping)

