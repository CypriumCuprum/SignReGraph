import pickle

def lpkl(pth):
    return pickle.load(open(pth, 'rb'))

def extract_to_only_label(ann,label_only, split):
    assert ann.endswith('.pkl')
    assert label_only.endswith('.pkl')
    data = lpkl(ann)
    split = set(data['split'][split])
    assert 'annos' in data or 'annotations' in data
    annotations = data['annos'] if 'annos' in data else data['annotations']
    key_name = 'frame_dir' if 'frame_dir' in annotations[0] else 'filename'
    data = [x for x in annotations if x[key_name] in split]
    only_label = [x['label'] for x in data]
    with open(label_only, "wb") as f:
        pickle.dump(only_label, f)

if __name__ == "__main__":
    ann_path = "../data/signlanguage_asl_citizen/2731asl_citizen.pkl"
    label_path = "../data/signlanguage_asl_citizen/only_label_test_2731asl_citizen.pkl"
    extract_to_only_label(ann_path, label_path, "test")
    
