import matplotlib.pyplot as plt
import numpy as np

lh_roi_alexnet = np.load("lh_mean_roi_correlation_alexnet_features.12.npy")
rh_roi_alexnet = np.load("rh_mean_roi_correlation_alexnet_features.12.npy")
lh_roi_clip = np.load("lh_mean_roi_correlation_clip_visual.npy")
rh_roi_clip = np.load("rh_mean_roi_correlation_clip_visual.npy")
lh_roi_effnet = np.load("lh_mean_roi_correlation_efficientnet_b0_features.8.npy")
rh_roi_effnet = np.load("rh_mean_roi_correlation_efficientnet_b0_features.8.npy")

roi_names = ['V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', 'EBA', 'FBA-1',
             'FBA-2', 'mTL-bodies', 'OFA', 'FFA-1', 'FFA-2', 'mTL-faces',
             'aTL-faces', 'OPA', 'PPA', 'RSC', 'OWFA', 'VWFA-1', 'VWFA-2',
             'mfs-words', 'mTL-words', 'early', 'midventral', 'midlateral',
             'midparietal', 'ventral', 'lateral', 'parietal', 'All vertices']

x = np.arange(len(roi_names))
width = 0.25
plt.bar(x , lh_roi_alexnet, width, label='Alexnet')
plt.bar(x + width, lh_roi_effnet, width, label='EfficientNet')
plt.bar(x + 2 * width, lh_roi_clip, width, label='Clip')
plt.ylim(bottom=0, top=1)
plt.xlabel('ROIs')
plt.xticks(ticks=x, labels=roi_names, rotation=60)
plt.ylabel('Mean Pearson\'s $r$')
plt.legend(frameon=True, loc=1)
plt.title("Left Hemisphere")
plt.show()
plt.bar(x , rh_roi_alexnet, width, label='Alexnet')
plt.bar(x + width, rh_roi_clip, width, label='Clip')
plt.bar(x + 2 * width, rh_roi_effnet, width, label='EfficientNet')
plt.ylim(bottom=0, top=1)
plt.xlabel('ROIs')
plt.xticks(ticks=x, labels=roi_names, rotation=60)
plt.ylabel('Mean Pearson\'s $r$')
plt.legend(frameon=True, loc=1)
plt.title("Right Hemisphere")
plt.show()
