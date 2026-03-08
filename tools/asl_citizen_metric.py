from signregraph.smp import *
from mmcv import load
import numpy as np

def pred_position(pred, label):
    pred = [np.argsort(x)[::-1] for x in pred]
    pos_pred = [(l == p).argmax() for l, p in zip(label, pred)]
    return pos_pred

def dcg(pred, label):
    pos_pred = pred_position(pred, label)
    dcg_all = [1/np.log2(1+(i+1)) for i in pos_pred]
    dcg_mean = np.mean(dcg_all)
    return dcg_mean

def mrr(pred, label):
    pos_pred = pred_position(pred, label)
    mrr_all = [1/(i+1) for i in pos_pred]
    mrr_mean = np.mean(mrr_all)
    return mrr_mean


labels = lpkl('./data/signlanguage_asl_citizen/only_label_test_2731asl_citizen.pkl')

joint_path = ""
bone_path = ""

joint = load(joint_path)
bone = load(bone_path)


print(f"Joint-DCG: {dcg(joint, labels)*100}")
print(f"Joint-MRR: {mrr(joint, labels)*100}")
print(f'Joint-Top1: {top1(joint, labels)*100}')
print(f'Joint-Top5: {topk(joint, labels, k=5)*100}')
print(f'Joint-Top10: {topk(joint, labels, k=10)*100}')

print(f"Bone-DCG: {dcg(bone, labels)*100}")
print(f"Bone-MRR: {mrr(bone, labels)*100}")
print(f'Bone-Top1: {top1(bone, labels)*100}')
print(f'Bone-Top5: {topk(bone, labels, k=5)*100}')
print(f'Bone-Top10: {topk(bone, labels, k=10)*100}')

fused = comb([joint, bone], [1, 1])
print(f"J+B-DCG: {dcg(fused, labels)*100}")
print(f"J+B-MRR: {mrr(fused, labels)*100}")
print(f'J+B-Top1: {top1(fused, labels)*100}')
print(f'J+B-Top5: {topk(fused, labels, k=5)*100}')
print(f'J+B-Top10: {topk(fused, labels, k=10)*100}')