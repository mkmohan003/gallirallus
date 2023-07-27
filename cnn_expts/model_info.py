import torch
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
#model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
#model = clip.load("ViT-B/32")[0].visual

train_nodes, _ = get_graph_node_names(model)
train_nodes = [n for n in train_nodes if "features" in n]
x = torch.rand(1, 3, 224, 224)

for node_name in train_nodes:
    feature_extractor = create_feature_extractor(model,
                                                 return_nodes=[node_name])
    out = feature_extractor(x)
    print(node_name, out[node_name].shape)
