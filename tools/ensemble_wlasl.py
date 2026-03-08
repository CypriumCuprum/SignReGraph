from mmcv import load, dump
from signregraph.smp import *
import datetime

joint_path = "" # best_pred.pkl
bone_path = ""


joint = load(joint_path)
bone = load(bone_path)

# label = load_label('./data/signlanguage_wlasl/2000wlasl_focus_hand_w_face.pkl', 'test')
# label = load_label("./data/signlanguage_wlasl/300wlasl_focus_hand_w_face.pkl", 'test')
label = load_label("./data/signlanguage_wlasl/100wlasl_focus_hand_w_face.pkl", 'test')


top_k=5
fused = comb([joint, bone], [1, 1])

print('J-Top-1', top1(joint, label)*100)
print(f'J-Top-{top_k}', topk(joint, label, k=top_k)*100)
print(f'J Mean Class Acc: {mean_acc(joint, label)*100}')

print('B-Top-1', top1(bone, label)*100)
print(f'B-Top-{top_k}', topk(bone, label, k=top_k)*100)
print(f'B Mean Class Acc: {mean_acc(bone, label)*100}')

print('J+B-Top-1', top1(fused, label)*100)
print(f'J+B-Top-{top_k}', topk(fused, label, k=top_k)*100)
print(f'J+B Mean Class Acc: {mean_acc(fused, label)*100}')



