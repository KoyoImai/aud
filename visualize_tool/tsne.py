import torch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

import random
import os


# データストリームのデータとバッファ内の最新のデータを
def feature_tsne(model, train_loader, args, batch_i):
    
    # tSNEの可視化結果を保存する用のパスを作成
    logdir = os.path.join(args.tsne_dir, args.log_name)
    #os.makedirs(logdir, exist_ok=True)
    tsne_path = logdir
    
    model.eval()
    
    with torch.no_grad():
        
        # バッファ内のインデックスを獲得し新しいデータを除く
        buff_index = train_loader.batch_sampler.fetch_buffer_data()
        buff_index = buff_index[:-args.batch_size]                    # バッファ内の新しいデータ（データストリームのデータ）は除く
        #buff_index = buff_index[-args.batch_size:]                    # 一つ前の反復で使用したデータのみを対象にする
        buff_index = buff_index[-50:]                                  # 最も新しいデータのみを対象にする
        #print("buff_index : ", buff_index)
    
        # データストリームのデータのインデックスを獲得
        stream_index = train_loader.batch_sampler.fetch_stream_data()
    
        if len(buff_index) <= 0:
            return
        
        # 可視化したい特徴量をランダムで選択
        k = 50   # データ数
        buff_index = random.sample(buff_index, k=k)
        stream_index = random.sample(stream_index, k=k)
        
        
        # バッファ内の画像とラベル，データストリーム内のデータとラベルを獲得
        buff_images = []
        buff_labels = []
        for i in buff_index:
            buff_image, buff_label = train_loader.dataset.get_buffer_data(i)
            buff_images.append(buff_image)
            buff_labels.append(buff_label)   
        stream_images = []
        stream_labels = []
        for i in stream_index:
            stream_image, stream_label = train_loader.dataset.get_buffer_data(i)
            stream_images.append(stream_image)
            stream_labels.append(stream_label)
        
        
        # データとラベルをそれぞれ連結
        buff_images = torch.cat(buff_images, dim=0)
        buff_labels = torch.stack(buff_labels, dim=0)
        buff_labels_list = buff_labels.tolist()
        buff_labels_list = [label for label in buff_labels_list for _ in range(args.num_patch)]
        buff_labels_set = set(buff_labels_list)
        #print("buff_images.shape : ", buff_images.shape)
        
        stream_images = torch.cat(stream_images, dim=0)
        stream_labels = torch.stack(stream_labels, dim=0)
        stream_labels_list = stream_labels.tolist()
        stream_labels_list = [label for label in stream_labels_list for _ in range(args.num_patch)]
        stream_labels_set = set(stream_labels_list)
        #print("stream_images.shape : ", stream_images.shape)
        
        
        
        # バッファ内のデータとデータストリームのデータを連結
        num_buff = buff_images.shape[0]
        images = torch.cat((buff_images, stream_images), dim=0)
        #print("num_buff : ", num_buff)
        #print("images.shape : ", images.shape)
        
        
        # 特徴埋め込みを獲得
        features, _, _ = model(images)
        features = features.detach().cpu().numpy()
        #buff_features = features[:num_buff]
        #stream_features = features[num_buff:]
        
        
        # tSNEモデルの作成と次元削減
        perplexity = 50
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=2000)
        X_tsne = tsne.fit_transform(features)
        #print("X_tsne.shape : ", X_tsne.shape)
        num = int(X_tsne.shape[0]/2)
        #print("num : ", num)
        buff_X_tsne = X_tsne[:num]
        stream_X_tsne = X_tsne[num:]
        
        #print("len(buff_X_tsne) : ", len(buff_X_tsne))
        #print("len(stream_X_tsne) : ", len(stream_X_tsne))
        
        
        # 可視化
        for cls in buff_labels_set:
            idxs = [i for i, label in enumerate(buff_labels_list) if label == cls]
            plt.scatter(buff_X_tsne[idxs, 0], buff_X_tsne[idxs, 1], label=f"Class {cls}", marker="x", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} buffer Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buff_{batch_i}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        for cls in stream_labels_set:
            idxs = [i for i, label in enumerate(stream_labels_list) if label == cls]
            plt.scatter(stream_X_tsne[idxs, 0], stream_X_tsne[idxs, 1], label=f"Class {cls}", marker="*", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_stream_{batch_i}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    
    model.train()
    
    
    
# バッファ内のデータのみを可視化する
def feature_tsne_v2(model, train_loader, args, batch_i):
    
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
        perplexity = 50
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=2000)
        X_tsne = tsne.fit_transform(features)
        
        #print("len(buff_X_tsne) : ", len(buff_X_tsne))
        #print("len(stream_X_tsne) : ", len(stream_X_tsne))
        
        
        # 可視化
        for cls in buff_labels_set:
            idxs = [i for i, label in enumerate(buff_labels_list) if label == cls]
            plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], label=f"Class {cls}", marker="x", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} buffer Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    
    model.train()


# バッファ内のデータのみを可視化する
# tSNEのモデルは全て同じものを使用する．
def feature_tsne_v3(model, train_loader, args, batch_i, tsne):
    
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
        perplexity = 50
        X_tsne = tsne.fit_transform(features)
        
        #print("len(buff_X_tsne) : ", len(buff_X_tsne))
        #print("len(stream_X_tsne) : ", len(stream_X_tsne))
        
        
        # 可視化
        for cls in buff_labels_set:
            idxs = [i for i, label in enumerate(buff_labels_list) if label == cls]
            plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], label=f"Class {cls}", marker="x", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} buffer Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    
    model.train()


    
    
# バッファ内のデータのみを可視化する
# 可視化する特徴量は，データ拡張を加えたデータの平均
def feature_tsne_v4(model, train_loader, args, batch_i):
    
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
        perplexity = 20
        #tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_tsne = tsne.fit_transform(features)
        
        #print("len(buff_X_tsne) : ", len(buff_X_tsne))
        #print("len(stream_X_tsne) : ", len(stream_X_tsne))
        
        
        # 可視化
        for cls in buff_labels_set:
            idxs = [i for i, label in enumerate(buff_labels_list) if label == cls]
            plt.scatter(X_tsne[idxs, 0], X_tsne[idxs, 1], label=f"Class {cls}", marker="x", s=10)
        plt.legend()
        plt.title(f"t-SNE Visualization of {args.data_type} buffer Features")
        plt.xlabel("t-SNE Feature 1")
        plt.ylabel("t-SNE Feature 2")
        plt.show()
        plt.savefig(f'{tsne_path}/empssl_{args.data_type}_buffonly_{batch_i}_tsne.png', dpi=100, bbox_inches='tight')
        plt.close()
        
    
    model.train()
    
    
    
def chunk_avg(x, n_chunks=2, normalize=False):
    x_list = x.chunk(n_chunks, dim=0)
    x = torch.stack(x_list, dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0), dim=1)
    