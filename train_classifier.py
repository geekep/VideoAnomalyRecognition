import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from features_loader import FeaturesLoader, FeaturesLoaderVal
from network.anomaly_detector_model import AnomalyDetector, RegularizedLoss, custom_objective
from network.model import static_model
from utils.svm import mysvm, save_model

parser = argparse.ArgumentParser(description="Train SVMs")
parser.add_argument('--features_path', default='out',
                    help="path to features")
parser.add_argument('--annotation_path', default="Train_Annotation.txt",
                    help="path to train annotation")
parser.add_argument('--random-seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', type=str, default="./exps/model",
                    help="set model dir.")

if __name__ == "__main__":
    args = parser.parse_args()

    # load network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    network = AnomalyDetector()
    network.to(device)
    net = static_model(net=network,
                       criterion=RegularizedLoss(network, custom_objective).to(device),
                       model_prefix=args.model_dir)
    model_path = net.get_checkpoint_path(20000)
    net.load_checkpoint(pretrain_path=model_path, epoch=20000)
    net.net.to(device)

    # enable cudnn tune
    cudnn.benchmark = True

    # load train data
    data_loader = FeaturesLoader(features_path=args.features_path,
                                 annotation_path=args.annotation_path)
    data_iter = torch.utils.data.DataLoader(data_loader,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=1,
                                            pin_memory=True)
    # train classifier
    X = []
    y = []
    for features, label in tqdm(data_iter):
        # features is a batch where each item is a tensor of 32 4096D features
        features.to(device)
        with torch.no_grad():
            input_var = torch.autograd.Variable(features)
            outputs = net.predict(input_var)[0]  # (batch_size, 32)
            outputs = outputs.reshape(outputs.shape[0], 32)
            features = features.reshape(32, features.shape[1]).astype('float')
            for i in range(32):
                if outputs[i] > 0.5:
                    X.append(features[i][:])
                    y.append(label)
    # train classifier
    clf = mysvm(X_train=X, y_train=y)
    save_model(clf.train_svm(), './exps/svm.pkl')

'''bash
python train_classifier.py
    --features_path c3d_out
    --annotation_path Anomaly_Detection_splits/Train_Annotation.txt
'''
