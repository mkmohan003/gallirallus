import clip
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from PIL import Image
from scipy.stats import pearsonr as corr
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LinearRegression
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision import transforms
from tqdm import tqdm

PLATFORM = "local"
BATCH_SIZE = 100
RAND_SEED = 5
FEATURES_TRAIN_TEMPLATE = "features_train_%s_%s.npy"
FEATURES_VAL_TEMPLATE = "features_val_%s_%s.npy"

if PLATFORM == "colab":
    DATA_DIR = '/content/drive/MyDrive/algonauts_2023_tutorial_data/subj01'
    DEVICE = "cuda:0"
    from google.colab import drive
    drive.mount('/content/drive/', force_remount=True)
else:
    DATA_DIR = "subj01"
    DEVICE = "cpu"


class ImageDataset(Dataset):
    def __init__(self, imgs_paths, idxs, transform):
        self.imgs_paths = np.array(imgs_paths)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        # Load the image
        img_path = self.imgs_paths[idx]
        img = Image.open(img_path).convert('RGB')
        # Preprocess the image and send it to the chosen device ('cpu' or 'cuda')
        if self.transform:
            img = self.transform(img).to(DEVICE)
        return img


def getImageDataLoader(data_dir, transform):
    train_img_dir = os.path.join(data_dir, 'training_split', 'training_images')
    test_img_dir = os.path.join(data_dir, 'test_split', 'test_images')
    train_img_list = os.listdir(train_img_dir)
    train_img_list.sort()
    test_img_list = os.listdir(test_img_dir)
    test_img_list.sort()
    print('Training images: ' + str(len(train_img_list)))
    print('Test images: ' + str(len(test_img_list)))
    # 90 - 10 train-val split. Hardcoded to 8800 to get a multiple of 100
    num_train = 8800
    idxs = np.arange(len(train_img_list))
    np.random.shuffle(idxs)
    idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]
    print('Training stimulus images: ' + format(len(idxs_train)))
    print('\nValidation stimulus images: ' + format(len(idxs_val)))
    train_imgs_paths = sorted(list(Path(train_img_dir).iterdir()))
    train_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_train, transform),
        batch_size=BATCH_SIZE
    )
    val_imgs_dataloader = DataLoader(
        ImageDataset(train_imgs_paths, idxs_val, transform),
        batch_size=BATCH_SIZE
    )

    return train_imgs_dataloader, idxs_train, val_imgs_dataloader, idxs_val


def getfMRIData(data_dir, idxs_train, idxs_val):
    fmri_dir = os.path.join(data_dir, 'training_split', 'training_fmri')
    lh_fmri = np.load(os.path.join(fmri_dir, 'lh_training_fmri.npy'))
    rh_fmri = np.load(os.path.join(fmri_dir, 'rh_training_fmri.npy'))
    print(f"LH training fMRI - images * vertices: {lh_fmri.shape}")
    print(f"RH training fMRI - images * vertices: {rh_fmri.shape}")
    lh_fmri_train = lh_fmri[idxs_train]
    lh_fmri_val = lh_fmri[idxs_val]
    rh_fmri_train = rh_fmri[idxs_train]
    rh_fmri_val = rh_fmri[idxs_val]
    # free up RAM
    del lh_fmri, rh_fmri
    return lh_fmri_train, rh_fmri_train, lh_fmri_val, rh_fmri_val


def fitPCA(feature_extractor, dataloader, flatten):

    # Define PCA parameters
    pca = IncrementalPCA(n_components=100, batch_size=BATCH_SIZE)

    # Fit PCA to batch
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        if flatten:
            ft = torch.hstack([torch.flatten(l, start_dim=1)
                               for l in ft.values()])
        # Fit PCA to batch
        pca.partial_fit(ft.detach().cpu().numpy())
    return pca


def extractFeatures(feature_extractor, dataloader, pca, flatten):

    features = []
    for _, d in tqdm(enumerate(dataloader), total=len(dataloader)):
        # Extract features
        ft = feature_extractor(d)
        # Flatten the features
        if flatten:
            ft = torch.hstack([torch.flatten(l, start_dim=1)
                               for l in ft.values()])
        # Apply PCA transform
        ft = pca.transform(ft.cpu().detach().numpy())
        features.append(ft)
    return np.vstack(features)


def getModelExtractor(model_name, model_layer):
    model = None
    feature_extractor = None
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    if model_name == "alexnet":
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet')
        feature_extractor = create_feature_extractor(model,
                                                     return_nodes=[model_layer])
    elif model_name == "efficientnet_b0":
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_extractor = create_feature_extractor(model,
                                                     return_nodes=[model_layer])
    elif model_name == "clip":
        model, transform = clip.load("ViT-B/32")
        feature_extractor = model.encode_image

    return model, feature_extractor, transform


def fitEncodingModel(features_train, lh_fmri_train, rh_fmri_train):
    reg_lh = LinearRegression().fit(features_train, lh_fmri_train)
    reg_rh = LinearRegression().fit(features_train, rh_fmri_train)
    return reg_lh, reg_rh


def predictWithEncodingModel(reg_lh, reg_rh, features_val):
    lh_fmri_val_pred = reg_lh.predict(features_val)
    rh_fmri_val_pred = reg_rh.predict(features_val)
    return lh_fmri_val_pred, rh_fmri_val_pred


def calculateCorrelation(lh_fmri_val, rh_fmri_val, lh_fmri_val_pred,
                         rh_fmri_val_pred):
    lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
    for v in tqdm(range(lh_fmri_val_pred.shape[1])):
        lh_correlation[v] = corr(lh_fmri_val_pred[:, v], lh_fmri_val[:, v])[0]

    rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
    for v in tqdm(range(rh_fmri_val_pred.shape[1])):
        rh_correlation[v] = corr(rh_fmri_val_pred[:, v], rh_fmri_val[:, v])[0]

    return lh_correlation, rh_correlation


def plotCorrelationMap(lh_correlation, rh_correlation, data_dir):
    # Load the ROI classes mapping dictionaries
    roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
        'mapping_floc-faces.npy', 'mapping_floc-places.npy',
        'mapping_floc-words.npy', 'mapping_streams.npy']
    roi_name_maps = []
    for r in roi_mapping_files:
        roi_name_maps.append(np.load(os.path.join(data_dir, 'roi_masks', r),
            allow_pickle=True).item())

    # Load the ROI brain surface maps
    lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
        'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
        'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
        'lh.streams_challenge_space.npy']
    rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
        'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
        'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
        'rh.streams_challenge_space.npy']
    lh_challenge_rois = []
    rh_challenge_rois = []
    for r in range(len(lh_challenge_roi_files)):
        lh_challenge_rois.append(np.load(os.path.join(data_dir, 'roi_masks',
            lh_challenge_roi_files[r])))
        rh_challenge_rois.append(np.load(os.path.join(data_dir, 'roi_masks',
            rh_challenge_roi_files[r])))

    # Select the correlation results vertices of each ROI
    roi_names = []
    lh_roi_correlation = []
    rh_roi_correlation = []
    for r1 in range(len(lh_challenge_rois)):
        for r2 in roi_name_maps[r1].items():
            if r2[0] != 0: # zeros indicate to vertices falling outside the ROI of interest
                roi_names.append(r2[1])
                lh_roi_idx = np.where(lh_challenge_rois[r1] == r2[0])[0]
                rh_roi_idx = np.where(rh_challenge_rois[r1] == r2[0])[0]
                lh_roi_correlation.append(lh_correlation[lh_roi_idx])
                rh_roi_correlation.append(rh_correlation[rh_roi_idx])

    roi_names.append('All vertices')
    lh_roi_correlation.append(lh_correlation)
    rh_roi_correlation.append(rh_correlation)

    lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r])
        for r in range(len(lh_roi_correlation))]
    rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r])
        for r in range(len(rh_roi_correlation))]

    plt.figure(figsize=(18, 6))
    x = np.arange(len(roi_names))
    width = 0.30
    plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
    plt.bar(x + width/2, rh_mean_roi_correlation, width, label='Right Hemishpere')
    plt.xlim(left=min(x)-.5, right=max(x)+.5)
    plt.ylim(bottom=0, top=1)
    plt.xlabel('ROIs')
    plt.xticks(ticks=x, labels=roi_names, rotation=60)
    plt.ylabel('Mean Pearson\'s $r$')
    plt.legend(frameon=True, loc=1)
    plt.show()


if __name__ == "__main__":
    np.random.seed(RAND_SEED)
    model_name = "alexnet"
    model_layer = "features.2"
    flatten = False if model_name == "clip" else True
    model, feature_extractor, transform = getModelExtractor(model_name,
                                                            model_layer)
    if model is not None:
        model.eval()
        model.to(DEVICE)
        train_imgs_loader, ids_train, val_imgs_loader, ids_val =  \
                getImageDataLoader(DATA_DIR, transform)
        lh_fmri_train, rh_fmri_train, lh_fmri_val, rh_fmri_val = \
                getfMRIData(DATA_DIR, ids_train, ids_val)
        FEATURES_TRAIN_FILE = FEATURES_TRAIN_TEMPLATE % (model_name, model_layer)
        FEATURES_VAL_FILE = FEATURES_VAL_TEMPLATE % (model_name, model_layer)
        if os.path.isfile(FEATURES_TRAIN_FILE) and os.path.isfile(FEATURES_VAL_FILE):
            features_train = np.load(FEATURES_TRAIN_FILE)
            features_val = np.load(FEATURES_VAL_FILE)
        else:
            pca = fitPCA(feature_extractor, train_imgs_loader, flatten)
            features_train = extractFeatures(feature_extractor,
                                             train_imgs_loader,
                                             pca, flatten)
            features_val = extractFeatures(feature_extractor,
                                           val_imgs_loader,
                                           pca, flatten)
            np.save(FEATURES_TRAIN_FILE, features_train)
            np.save(FEATURES_VAL_FILE, features_val)

        print(f"training image features: {features_train.shape}")
        print(f"validation image features: {features_val.shape}")
        reg_lh, reg_rh = fitEncodingModel(features_train, lh_fmri_train,
                                          rh_fmri_train)
        lh_fmri_val_pred, rh_fmri_val_pred = \
                predictWithEncodingModel(reg_lh, reg_rh, features_val)
        lh_correlation, rh_correlation = calculateCorrelation(lh_fmri_val,
                                                              rh_fmri_val,
                                                              lh_fmri_val_pred,
                                                              rh_fmri_val_pred)
        print(f"Overall: L:{lh_correlation.mean()},R:{rh_correlation.mean()}")
        plotCorrelationMap(lh_correlation, rh_correlation, DATA_DIR)
