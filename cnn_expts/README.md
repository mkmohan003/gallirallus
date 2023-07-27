
```python feat_extract_correlation.py``` - Does the following: 
  - feature extraction from the conv layers of Alexnet, Efficient-B0 & CLIP by using the Subj01 dataset from Algonaut(which is taken from NSD).
  - Runs a PCA on the extracted features
  - Fits a Voxel encoding model (Linear Regression) on the training data and does predictions on the validation data.
  - Calculates Correlation between the model predictions and the Ground truth fMRI values.

```python plot.py``` - Plots as a bar chart of the Correlation between predictions and Ground truth for the different visual regions of the brain.
