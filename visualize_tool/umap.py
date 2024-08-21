import torch
import random
import os
import numpy as np

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import datasets.datasets_minred_empssl

# 特徴量可視化用
from matplotlib import pyplot as plt
import umap.umap_ as umap



class feature_umap_model:
    
    def __init__(self, args):
        
        self.features = None
        
        if args.data_type == 'cifar10':
            trainfname = "/home/kouyou/CIFAR10"
            valfname = "/home/kouyou/CIFAR10"
            data_insize = 32
            class_size = 1000
        else:
            assert False
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        augmentation = transforms.Compose([
            transforms.Resize(size=(data_insize, data_insize)),
            transforms.ToTensor(),
            normalize,
        ])
        
        included_classes = [0, 1, 2, 3, 4]
        self.included_classes = included_classes
        
        
        # データセットの作成
        if args.data_type == "cifar10":
            train_dataset = datasets.datasets_minred_empssl.Cifar10tSNEDataset(
                trainfname,
                included_classes,
                class_size=class_size,
                transform=augmentation,
                train=True,
                download=True
            )

        # train_loaderの作成
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle=True,
            num_workers=5,
            pin_memory=True
        )
        
        self.train_loader = train_loader
    
    
    def visualize(self, model, args, batch_i):
        
        # 可視化結果を保存するパス
        logdir = os.path.join(args.tsne_dir, args.log_name)
        tsne_path = logdir

        # modelをevalモードに変更
        model.eval()
        
        """  リスト作成  """
        # 特徴量格納用リスト
        features = []

        # ラベル格納用リスト
        labels = []
        
        # タスクの決定
        #if args.data_type == "cifar10":
        #    #print("len(train_loader) : ", len(train_loader))  # 総イタレーション数
        #    task_size = len(train_loader) / 10
        #    print("task_size : ", task_size)
        #    if batch_i > 0 and batch_i <= task_size * 1:
        #        included_classes = [0]
        #    elif batch_i > task_size*1 and batch_i <= task_size*2:
        #        included_classes = [0, 1]
        #    elif batch_i > task_size*2 and batch_i <= task_size*3:
        #        included_classes = [0, 1, 2]
        #    elif batch_i > task_size*3 and batch_i <= task_size*4:
        #        included_classes = [0, 1, 2, 3]
        #    elif batch_i > task_size*4:
        #        included_classes = [0, 1, 2, 3, 4]
        #else:
        #    assert False
        
        # 特徴量の可視化を行いたい
        with torch.no_grad():
        
            for idx, out in enumerate(self.train_loader):

                # 画像データとラベルを用意
                images, targets = out['input'], out['target']

                # ラベルの格納
                for label in targets:
                    labels.append(label.item())

                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)

                # 特徴埋め込みの獲得
                embeddings, feature1, feature2 = model(images)

                for feature in embeddings:
                    features.append(feature)

            # 特徴量を連結
            features = torch.stack(list(features), dim=0)
            features = features.detach().cpu().numpy()

            if self.features is None:
                
                # UMAPモデルの作成と次元削減
                umap_model = umap.UMAP(n_components=2, random_state=7)
                X = umap_model.fit_transform(features)
                
                self.features = features
            else:
                # UMAPモデルの作成・訓練と次元削減
                umap_model = umap.UMAP(n_components=2, random_state=7)
                _ = umap_model.fit_transform(self.features)
                X = umap_model.transform(features)
                
                #print("self.features : ", self.features.shape)                     # (5000, 1024)
                #print("features : ", features.shape)                               # (5000, 1024)
                self.features = np.concatenate((self.features, features), axis=0)
                #print("self.features.shape : ", self.features.shape)               # (10000, 1024)
                
        
        color_map = class_color_cifar10()
        for cls in self.included_classes:
            idxs = [i for i, label in enumerate(labels) if label == cls]
            plt.scatter(X[idxs, 0], X[idxs, 1], label=f"Class {cls}", marker="o", s=1, color=color_map[str(cls)])
            
            
        plt.legend()
        plt.title(f"iteration {batch_i}")
        plt.xlabel("UMAP Feature 1")
        plt.ylabel("UMAP Feature 2")
        plt.show()
        plt.legend(loc='upper right')
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i:07d}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        
        model.train()
                
        
        
class feature_umap_model:
    
    def __init__(self, args):
        
        self.features = None
        
        if args.data_type == 'cifar10':
            trainfname = "/home/kouyou/CIFAR10"
            valfname = "/home/kouyou/CIFAR10"
            data_insize = 32
            class_size = 1000
        else:
            assert False
        
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])

        augmentation = transforms.Compose([
            transforms.Resize(size=(data_insize, data_insize)),
            transforms.ToTensor(),
            normalize,
        ])
        
        included_classes = [0, 1, 2, 3, 4]
        self.included_classes = included_classes
        
        self.num_features = len(included_classes) * class_size
        
        
        # データセットの作成
        if args.data_type == "cifar10":
            train_dataset = datasets.datasets_minred_empssl.Cifar10tSNEDataset(
                trainfname,
                included_classes,
                class_size=class_size,
                transform=augmentation,
                train=True,
                download=True
            )

        # train_loaderの作成
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle=False,
            num_workers=5,
            pin_memory=True
        )
        
        self.train_loader = train_loader
    
    
    def store_features(self, model, args, batch_i):

        # modelをevalモードに変更
        model.eval()
        
        """  リスト作成  """
        # 特徴量格納用リスト
        features = []

        # ラベル格納用リスト
        labels = []
        
        # タスクの決定
        #if args.data_type == "cifar10":
        #    #print("len(train_loader) : ", len(train_loader))  # 総イタレーション数
        #    task_size = len(train_loader) / 10
        #    print("task_size : ", task_size)
        #    if batch_i > 0 and batch_i <= task_size * 1:
        #        included_classes = [0]
        #    elif batch_i > task_size*1 and batch_i <= task_size*2:
        #        included_classes = [0, 1]
        #    elif batch_i > task_size*2 and batch_i <= task_size*3:
        #        included_classes = [0, 1, 2]
        #    elif batch_i > task_size*3 and batch_i <= task_size*4:
        #        included_classes = [0, 1, 2, 3]
        #    elif batch_i > task_size*4:
        #        included_classes = [0, 1, 2, 3, 4]
        #else:
        #    assert False
        
        # 特徴量の可視化を行いたい
        with torch.no_grad():
        
            for idx, out in enumerate(self.train_loader):

                # 画像データとラベルを用意
                images, targets = out['input'], out['target']

                # ラベルの格納
                for label in targets:
                    labels.append(label.item())

                if torch.cuda.is_available():
                    images = images.cuda(non_blocking=True)

                # 特徴埋め込みの獲得
                embeddings, feature1, feature2 = model(images)

                for feature in embeddings:
                    features.append(feature)

            # 特徴量を連結
            features = torch.stack(list(features), dim=0)
            features = features.detach().cpu().numpy()

            if self.features is None:                
                self.features = features
                self.labels = labels
            else:
                # UMAPモデルの作成・訓練と次元削減                
                #print("self.features : ", self.features.shape)                     # (5000, 1024)
                #print("features : ", features.shape)                               # (5000, 1024)
                self.features = np.concatenate((self.features, features), axis=0)
                #print("self.features.shape : ", self.features.shape)               # (10000, 1024)
                self.labels += labels
                
                
        model.train()
        
    def visualize(self, batch_i, args):
        
        # 可視化結果を保存するパス
        logdir = os.path.join(args.tsne_dir, args.log_name)
        logdir = os.path.join(logdir, str(batch_i))
        os.makedirs(logdir, exist_ok=True)
        tsne_path = logdir
        
        num_iter = int(self.features.shape[0] / self.num_features)
        
        #umap_model = umap.UMAP(n_components=2, random_state=7)
        umap_model = umap.UMAP(n_components=2)
        _ = umap_model.fit_transform(self.features)
        
        
        #print("num_iter : ", num_iter)
        
        for i in range(num_iter):
            features = self.features[i*self.num_features:(i+1)*self.num_features]
            labels = self.labels[i*self.num_features:(i+1)*self.num_features]
            #labels = self.labels
            #print("features.shape : ", self.features.shape)
            
            X = umap_model.transform(features)
        
            color_map = class_color_cifar10()
            for cls in self.included_classes:
                idxs = [i for i, label in enumerate(labels) if label == cls]
                plt.scatter(X[idxs, 0], X[idxs, 1], label=f"Class {cls}", marker="o", s=1, color=color_map[str(cls)])


            plt.legend()
            plt.title(f"iteration {batch_i}")
            plt.xlabel("UMAP Feature 1")
            plt.ylabel("UMAP Feature 2")
            plt.show()
            plt.legend(loc='upper right')
            plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i:07d}_{i:07d}_umap.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        
        
                
            
# バッファ内のデータのみを可視化する
def feature_umap_v2(model, train_loader, args, batch_i):
    
    # tSNEの可視化結果を保存する用のパスを作成
    logdir = os.path.join(args.tsne_dir, args.log_name)
    #os.makedirs(logdir, exist_ok=True)
    tsne_path = logdir
    
    model.eval()
    
    with torch.no_grad():
        
        # バッファ内のインデックスを獲得
        buff_index = train_loader.batch_sampler.fetch_buffer_data()
    
        if len(buff_index) <= 0:
            return
        
        # 可視化したい特徴量をランダムで選択
        #k = 50   # データ数
        #buff_index = random.sample(buff_index, k=k)
        
        
        
        # バッファ内の画像とラベル，データストリーム内のデータとラベルを獲得
        buff_images = []
        buff_labels = []
        for i in buff_index:
            buff_image, buff_label = train_loader.dataset.get_original_data(i)
            buff_images.append(buff_image)
            buff_labels.append(buff_label)   
        
        
        # データとラベルをそれぞれ連結
        buff_images = torch.stack(buff_images, dim=0)
        buff_labels = torch.stack(buff_labels, dim=0)
        buff_labels_list = buff_labels.tolist()
        buff_labels_set = set(buff_labels_list)
        
        
        # 特徴埋め込みを獲得
        features, _, _ = model(buff_images)
        features = features.detach().cpu().numpy()
        #buff_features = features[:num_buff]
        #stream_features = features[num_buff:]
        
        
        # tSNEモデルの作成と次元削減
        X = umap.UMAP(n_components=2).fit_transform(features)
        
        #print("len(buff_X_tsne) : ", len(buff_X_tsne))
        #print("len(stream_X_tsne) : ", len(stream_X_tsne))
        
        
        # 可視化
        for cls in buff_labels_set:
            idxs = [i for i, label in enumerate(buff_labels_list) if label == cls]
            plt.scatter(X[idxs, 0], X[idxs, 1], label=f"Class {cls}", marker="x", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} buffer Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i}_umap.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    
    model.train()
    
    
    
    
# バッファ内のデータのみを可視化する
# 可視化する特徴量は，データ拡張を加えたデータの平均
def feature_umap_v4(model, train_loader, args, batch_i):
    
    # tSNEの可視化結果を保存する用のパスを作成
    logdir = os.path.join(args.tsne_dir, args.log_name)
    #os.makedirs(logdir, exist_ok=True)
    tsne_path = logdir
    
    model.eval()
    
    with torch.no_grad():
        
        # バッファ内のインデックスを獲得
        buff_index = train_loader.batch_sampler.fetch_buffer_data()
    
        if len(buff_index) <= 0:
            return
        
        # 可視化したい特徴量をランダムで選択
        #k = 50   # データ数
        #buff_index = random.sample(buff_index, k=k)
        
        
        
        # バッファ内の画像とラベル，データストリーム内のデータとラベルを獲得
        buff_images = []
        buff_labels = []
        for i in buff_index:
            buff_image, buff_label = train_loader.dataset.get_buffer_data(i)
            buff_images.append(buff_image)
            buff_labels.append(buff_label)   
        
        
        # データとラベルをそれぞれ連結
        #buff_images = torch.cat(buff_images, dim=0)
        #buff_images = torch.stack(buff_images, dim=0)
        buff_labels = torch.stack(buff_labels, dim=0)
        buff_labels_list = buff_labels.tolist()
        buff_labels_set = set(buff_labels_list)
        
        #print("buff_images.shape : ", buff_images.shape)
        
        # 特徴埋め込みを獲得
        # out of memoryを回避するために複数回に分けて実行
        features = []
        for buff_image in buff_images:
            #print("buff_image.shape : ", buff_image.shape)
            if torch.cuda.is_available():
                buff_image = buff_image.cuda()
                feature, _, _ = model(buff_image)
                features.append(feature)
        
        #features, _, _ = model(buff_images)
        #print("features.shape : ", features.shape)
        
        features = torch.cat(features, dim=0)
        #print("features.shape : ", features.shape)
        features = chunk_avg(features, args.num_patch)
        #print("features.shape : ", features.shape)
        
        
        features = features.detach().cpu().numpy()
        
        # tSNEモデルの作成と次元削減
        X = umap.UMAP(n_components=2).fit_transform(features)
        
        #print("len(buff_X_tsne) : ", len(buff_X_tsne))
        #print("len(stream_X_tsne) : ", len(stream_X_tsne))
        
        
        # 可視化
        for cls in buff_labels_set:
            idxs = [i for i, label in enumerate(buff_labels_list) if label == cls]
            plt.scatter(X[idxs, 0], X[idxs, 1], label=f"Class {cls}", marker="x", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} buffer Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    
    model.train()
    
    
    
    
    
def feature_umap_v5(model, train_loader, args, batch_i):
    
    # 可視化結果を保存するパス
    logdir = os.path.join(args.tsne_dir, args.log_name)
    tsne_path = logdir
    
    if train_loader.batch_sampler.delete_buffer is None:
        return
    
    del_features, del_labels = train_loader.batch_sampler.fetch_delete_feature_label()
    
    # 現在のバッファ内のデータの特徴量とラベル
    features, labels = train_loader.batch_sampler.fetch_feature_label()
    features = torch.stack(features, dim=0)
    features = features.detach().cpu().numpy()
    labels_set = set(labels)
    #print("features.shape : ", features.shape)   # features.shape :  torch.Size([356, 1024])
    
    # バッファから削除したデータの特徴量とラベル
    del_features, del_labels = train_loader.batch_sampler.fetch_delete_feature_label()
    del_features = torch.stack(del_features, dim=0)
    del_features = del_features.detach().cpu().numpy()
    del_labels_set = set(del_labels)
    
    #print("features.shape : ", features.shape)
    #print("del_features.shape : ", del_features.shape)
    
    all_features = np.concatenate((features, del_features), axis=0)
    #print("all_features.shape : ", all_features.shape)
    
    #print(all_features)
    
    
    
    num_features = features.shape[0]
    #print("num_features : ", num_features)
    
    
    # UMAPモデルの作成と次元削減
    umap_model = umap.UMAP(n_components=2)
    X = umap_model.fit_transform(all_features)
    
    X_del = X[num_features:]
    
    color_map = class_color_cifar10()
    
    # バッファ内データの可視化
    for cls in labels_set:
        idxs = [i for i, label in enumerate(labels) if label == cls]
        plt.scatter(X[idxs, 0], X[idxs, 1], label=f"Class {cls}", marker="o", s=1, color=color_map[str(cls)])
    
    
    for cls in del_labels_set:
        idxs = [i for i, label in enumerate(del_labels) if label == cls]
        plt.scatter(X_del[idxs, 0], X_del[idxs, 1], label=f"Del Class {cls}", marker="x", s=23, color=color_map[str(cls)])
    
    
    plt.legend(loc='upper right')
    plt.title(f"UMAP Visualization of {args.data_type} buffer Features")
    plt.xlabel("UMAP Feature 1")
    plt.ylabel("UMAP Feature 2")
    plt.show()
    plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i:07d}_tsne.png', dpi=100, bbox_inches='tight')
    plt.close()

    
    return


# バッファ内データではなく，学習データ全ての特徴量を可視化する
def feature_umap_v6(model, train_loader, args, batch_i):
    
    
    if args.data_type == 'cifar10':
        trainfname = "/home/kouyou/CIFAR10"
        valfname = "/home/kouyou/CIFAR10"
        data_insize = 32
    else:
        assert False
    
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    
    augmentation = transforms.Compose([
        transforms.Resize(size=(data_insize, data_insize)),
        transforms.ToTensor(),
        normalize,
    ])
    
    if args.data_type == 'cifar10':
        trainfname = "/home/kouyou/CIFAR10"
        valfname = "/home/kouyou/CIFAR10"
        class_size = 1000
        data_insize = 32
    else:
        assert False
    
    # タスクの決定
    if args.data_type == "cifar10":
        #print("len(train_loader) : ", len(train_loader))  # 総イタレーション数
        task_size = len(train_loader) / 10
        #print("task_size : ", task_size)
        if batch_i > 0 and batch_i <= task_size * 1:
            included_classes = [0]
        elif batch_i > task_size*1 and batch_i <= task_size*2:
            included_classes = [0, 1]
        elif batch_i > task_size*2 and batch_i <= task_size*3:
            included_classes = [0, 1, 2]
        elif batch_i > task_size*3 and batch_i <= task_size*4:
            included_classes = [0, 1, 2, 3]
        elif batch_i > task_size*4:
            included_classes = [0, 1, 2, 3, 4]
    else:
        assert False
            
    
    # データセットの作成
    if args.data_type == "cifar10":
        train_dataset = datasets.datasets_minred_empssl.Cifar10tSNEDataset(
            trainfname,
            included_classes,
            class_size=class_size,
            transform=augmentation,
            train=True,
            download=True
        )
    
    # train_loaderの作成
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = args.batch_size,
        shuffle=True,
        num_workers=5,
        pin_memory=True
    )
    
    visualize(train_loader, model, args, included_classes, batch_i)
    
    
def visualize(train_loader, model, args, included_classes, batch_i):
    
    # 可視化結果を保存するパス
    logdir = os.path.join(args.tsne_dir, args.log_name)
    tsne_path = logdir
    
    # modelをevalモードに変更
    model.eval()
    
    """  リスト作成  """
    # 特徴量格納用リスト
    features = []
    
    # ラベル格納用リスト
    labels = []
    
    with torch.no_grad():
        
        for idx, out in enumerate(train_loader):
            
            # 画像データとラベルを用意
            images, targets = out['input'], out['target']
            
            # ラベルの格納
            for label in targets:
                labels.append(label.item())
            
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
            
            # 特徴埋め込みの獲得
            embeddings, feature1, feature2 = model(images)
            
            for feature in embeddings:
                features.append(feature)
                
        # 特徴量を連結
        features = torch.stack(list(features), dim=0)
        features = features.detach().cpu().numpy()
    
        # UMAPモデルの作成と次元削減
        umap_model = umap.UMAP(n_components=2)
        X = umap_model.fit_transform(features)
        
        color_map = class_color_cifar10()
        for cls in included_classes:
            idxs = [i for i, label in enumerate(labels) if label == cls]
            plt.scatter(X[idxs, 0], X[idxs, 1], label=f"Class {cls}", marker="o", s=1, color=color_map[str(cls)])
            
            
        plt.legend()
        plt.title(f"iteration {batch_i}")
        plt.xlabel("UMAP Feature 1")
        plt.ylabel("UMAP Feature 2")
        plt.show()
        plt.legend(loc='upper right')
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i:07d}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        model.train()
    
def class_color_cifar10():
    color_map = {
        '0': 'red',      # airplane
        '1': 'blue',     # automobile
        '2': 'pink',     # bird
        '3': 'orange',   # cat
        '4': 'purple',   # deer
        '5': 'brown',    # dog
        '6': 'green',    # frog
        '7': 'gray',     # horse
        '8': 'olive',    # ship
        '9': 'cyan',     # truck
    }
    return color_map
