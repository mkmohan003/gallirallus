import clip
import torch
import torchextractor as tx

model = clip.load("ViT-B/32")[0].visual
layer = ['conv1']
model = tx.Extractor(model, layer)
dummy_input = torch.rand(1, 3, 224, 224)
model_output, features = model(dummy_input)
feature_shapes = {name: f.shape for name, f in features.items()}
print(feature_shapes)
