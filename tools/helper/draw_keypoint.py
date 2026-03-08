import matplotlib
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("./figs", exist_ok=True)

def draw_skeleton(ax, keypoints, limbs_definition, title="", kps_color='red', limb_color='blue'):
    kps = np.array(keypoints)
    
    ax.scatter(kps[:, 0], kps[:, 1], c=kps_color, s=50, zorder=2)
    
    for (idx1, idx2) in limbs_definition:
        point1 = kps[idx1]
        point2 = kps[idx2]
        
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 
                color=limb_color, linewidth=2, zorder=1)

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.invert_yaxis()

def compare_skeletons_to_file(predicted_keypoints, real_keypoints, limbs_definition, output_filename="comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    draw_skeleton(axes[0], predicted_keypoints, limbs_definition, 
                  "Predicted Keypoints", kps_color='cyan', limb_color='blue')
    draw_skeleton(axes[1], real_keypoints, limbs_definition, 
                  "Real Keypoints", kps_color='red', limb_color='green')
    
    plt.tight_layout()

    plt.savefig(output_filename, transparent=True)
    print(f"Save to: {output_filename}")
    
    plt.close(fig)
