import logging
import os
import pathlib
import pickle
import zipfile
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.preprocessing import OneHotEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataloader import load_graph_adj_mtx, load_graph_node_features
from model_v2 import (gen_loc_graph, gen_nei_graph, GCN, GCNWithDistMatrix, NodeAttnMap, UserEmbeddings, POIEmbeddings, AdjEmbeddings, Time2Vec, CategoryEmbeddings,
                        TransformerModel, Poi_representation, FuseBlock, Task1Model, Pretask1)
from param_parser import parameter_parser
from utils import (increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, recall_at_k,
                   mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss)


def train(args):
    args.save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok, sep='-')
    if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

    # Setup logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=os.path.join(args.save_dir, f"log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    logging.getLogger('matplotlib.font_manager').disabled = True

    # Save run settings
    logging.info(args)
    with open(os.path.join(args.save_dir, 'args.yaml'), 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # Save python code
    zipf = zipfile.ZipFile(os.path.join(args.save_dir, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
    zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
    zipf.close()

    # %% ====================== Load data ======================
    # Read check-in train data
    train_df = pd.read_csv(args.data_train, encoding='GBK')
    val_df = pd.read_csv(args.data_val, encoding='GBK')

    # Build POI graph (built from train_df)
    print('Loading POI graph...')
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)
    logging.info(
        f"raw_X.shape: {raw_X.shape}; "
        f"Four features: {args.feature1}, {args.feature2}, {args.feature3}, {args.feature4}.")
    logging.info(f"raw_A.shape: {raw_A.shape}; Edge from row_index to col_index with weight (frequency).")
    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    logging.info('One-hot encoding poi categories id')
    one_hot_encoder = OneHotEncoder()
    cat_list = list(raw_X[:, 1])
    one_hot_encoder.fit(list(map(lambda x: [x], cat_list)))
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]
    logging.info(f"After one hot encoding poi cat, X.shape: {X.shape}")
    logging.info(f'POI categories: {list(one_hot_encoder.categories_[0])}')
    # Save ont-hot encoder
    with open(os.path.join(args.save_dir, 'one-hot-encoder.pkl'), "wb") as f:
        pickle.dump(one_hot_encoder, f)

    # Normalization
    print('Laplician matrix...')
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')

    # POI id to index
    nodes_df = pd.read_csv(args.data_node_feats, encoding='GBK')
    poi_ids = list(set(nodes_df['node_name/poi_id'].tolist()))
    poi_id2idx_dict = dict(zip(poi_ids, range(len(poi_ids))))

    # Cat id to index
    cat_ids = list(set(nodes_df[args.feature2].tolist()))
    cat_id2idx_dict = dict(zip(cat_ids, range(len(cat_ids))))

    # Poi idx to cat idx
    poi_idx2cat_idx_dict = {}
    for i, row in nodes_df.iterrows():
        poi_idx2cat_idx_dict[poi_id2idx_dict[row['node_name/poi_id']]] = \
            cat_id2idx_dict[row[args.feature2]]

    # User id to index
    user_ids = [str(each) for each in list(set(train_df['user_id'].to_list()))]
    user_id2idx_dict = dict(zip(user_ids, range(len(user_ids))))

    # 生成 user-poi 邻接图
    logging.info('Generating user-poi adjacency graph...')
    user_poi_edges = gen_nei_graph(train_df, len(user_id2idx_dict), poi_id2idx_dict)
    user_poi_edges = user_poi_edges.to(device=args.device)
    #print(user_poi_edges)

    # 生成 POI 距离图
    logging.info('Generating poi distance graph...')
    poi_loc = {poi_id2idx_dict[row['node_name/poi_id']]: (row[args.feature3], row[args.feature4]) for i, row in nodes_df.iterrows()}
    dist_mat, poi_distance_edges, poi_nei_dict = gen_loc_graph(poi_loc, len(poi_id2idx_dict), thre=args.poi_dist_thre)
    dist_mat = torch.tensor(dist_mat).to(device=args.device, dtype=torch.float)
    #print(dist_mat)

    # %% ====================== Define Dataset ======================
    class TrajectoryDatasetTrain(Dataset):
        def __init__(self, train_df):
            self.traj_seqs = []  # traj id: user id + traj no.
            self.input_seqs = []
            self.label_seqs = []
            self.user_seqs = []
            self.pos_seqs = []
            self.neg_seqs = []

            user_pos_neg_seqs = {}

            for user_id in tqdm(set(train_df['user_id'].to_list())):
                user_df = train_df[train_df['user_id'] == user_id]
                poi_ids = user_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]

                # 确保POI序列长度足够大以支持移除一个元素
                if len(poi_idxs) > 1:
                    # 随机选择一个索引进行移除
                    remove_idx = np.random.randint(0, len(poi_idxs))

                    # 负样本：被移除的那个POI
                    neg_seq = poi_idxs[remove_idx]

                    # 正样本：移除所有与负样本相同的POI
                    pos_seq = [poi for poi in poi_idxs if poi != neg_seq]

                    user_pos_neg_seqs[user_id] = (pos_seq, neg_seq)

            for traj_id in tqdm(set(train_df['trajectory_id'].tolist())):
                traj_df = train_df[train_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = [poi_id2idx_dict[each] for each in poi_ids]
                time_feature = traj_df[args.time_feature].to_list()

                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                if len(input_seq) < args.short_traj_thres:
                    continue
                if len(set(poi_idxs)) == 1:
                    continue

                self.traj_seqs.append(traj_id)
                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)

                # 获取该轨迹的用户ID，并扩充对应的正负样本
                user_id = traj_df['user_id'].iloc[0]
                pos_seq, neg_seq = user_pos_neg_seqs[user_id]
                self.user_seqs.append(str(user_id))
                self.pos_seqs.append(pos_seq)
                self.neg_seqs.append(neg_seq)

            #print(self.traj_seqs, self.input_seqs, self.label_seqs)
            #print(self.user_seqs, self.pos_seqs, self.neg_seqs)


        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index], self.user_seqs[index], self.pos_seqs[index], self.neg_seqs[index])

    class TrajectoryDatasetVal(Dataset):
        def __init__(self, val_df):
            self.traj_seqs = []
            self.input_seqs = []
            self.label_seqs = []
            self.user_seqs = []
            self.pos_seqs = []
            self.neg_seqs = []
            user_pos_neg_seqs = {}
            for user_id in tqdm(set(val_df['user_id'].to_list())):
                # Ignore user if not in training set
                if str(user_id) not in user_id2idx_dict.keys():
                    continue
                user_df = val_df[val_df['user_id'] == user_id]
                poi_ids = user_df['POI_id'].to_list()
                poi_idxs = []
                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                if len(poi_idxs) > 1:
                    neg_seq = poi_idxs[-1]

                    pos_seq = [poi for poi in poi_idxs if poi != neg_seq]

                    user_pos_neg_seqs[user_id] = (pos_seq, neg_seq)

            for traj_id in tqdm(set(val_df['trajectory_id'].tolist())):
                user_id = traj_id.split('_')[0]

                # Ignore user if not in training set
                if user_id not in user_id2idx_dict.keys():
                    continue



                # Ger POIs idx in this trajectory
                traj_df = val_df[val_df['trajectory_id'] == traj_id]
                poi_ids = traj_df['POI_id'].to_list()
                poi_idxs = []
                time_feature = traj_df[args.time_feature].to_list()
                for each in poi_ids:
                    if each in poi_id2idx_dict.keys():
                        poi_idxs.append(poi_id2idx_dict[each])
                    else:
                        # Ignore poi if not in training set
                        continue

                # Construct input seq and label seq
                input_seq = []
                label_seq = []
                for i in range(len(poi_idxs) - 1):
                    input_seq.append((poi_idxs[i], time_feature[i]))
                    label_seq.append((poi_idxs[i + 1], time_feature[i + 1]))

                # Ignore seq if too short
                if len(input_seq) < args.short_traj_thres:
                    continue
                if len(set(poi_idxs)) == 1:
                    continue

                self.input_seqs.append(input_seq)
                self.label_seqs.append(label_seq)
                self.traj_seqs.append(traj_id)

                pos_seq, neg_seq = user_pos_neg_seqs[int(user_id)]
                self.user_seqs.append(user_id)
                self.pos_seqs.append(pos_seq)
                self.neg_seqs.append(neg_seq)

        def __len__(self):
            assert len(self.input_seqs) == len(self.label_seqs) == len(self.traj_seqs)
            return len(self.traj_seqs)

        def __getitem__(self, index):
            return (self.traj_seqs[index], self.input_seqs[index], self.label_seqs[index], self.user_seqs[index], self.pos_seqs[index], self.neg_seqs[index])

    # %% ====================== Define dataloader ======================
    print('Prepare dataloader...')
    train_dataset = TrajectoryDatasetTrain(train_df)
    val_dataset = TrajectoryDatasetVal(val_df)

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch,
                              shuffle=True, drop_last=False,
                              pin_memory=True, num_workers=args.workers,
                              collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch,
                            shuffle=False, drop_last=False,
                            pin_memory=True, num_workers=args.workers,
                            collate_fn=lambda x: x)

    # %% ====================== Build Models ======================
    # Model1: POI embedding model
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
        A = torch.from_numpy(A)
    X = X.to(device=args.device, dtype=torch.float)
    A = A.to(device=args.device, dtype=torch.float)

    args.gcn_nfeat = X.shape[1]
    
    #poi_embed_model = GCN(ninput=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.poi_embed_dim, dropout=args.gcn_dropout)
    poi_embed_model = GCNWithDistMatrix(in_features=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.poi_embed_dim, adj_matrix=A, dist_matrix=dist_mat, dropout=args.gcn_dropout).to(device=args.device)
    # Node Attn Model
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    num_pois = len(poi_id2idx_dict)
    #poi_embed_model = POIEmbeddings(num_pois, args.poi_embed_dim)

    # %% Model2: User embedding model Adj embedding model, nn.embedding
    num_users = len(user_id2idx_dict)
    user_embed_model = UserEmbeddings(num_users, embedding_dim=args.user_embed_dim)
    adj_embed_model = AdjEmbeddings(num_users, embedding_dim=args.user_embed_dim, user_poi_edges=user_poi_edges)

    # %% Model3: Time Model
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)

    # %% Model4: Category embedding model
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    # %% Model5: Embedding fusion models
    args.seq_input_embed = args.poi_embed_dim + args.user_embed_dim + args.time_embed_dim + args.cat_embed_dim
    args.task1_input_embed = args.poi_embed_dim + args.user_embed_dim + args.cat_embed_dim
    poi_rep_model = Poi_representation(args.poi_embed_dim, args.cat_embed_dim)
    time_focused_model = FuseBlock(args.user_embed_dim, args.time_embed_dim)
    adj_focused_model = FuseBlock(args.user_embed_dim, args.poi_embed_dim + args.cat_embed_dim)
    #pre_task1_model = Pretask1(num_pois, args.task1_input_embed)
    # %% Model6: Sequence model
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)
    task1_model = Task1Model(num_pois, args.task1_input_embed , d_ffn=256, num_layers=2)
    # Define overall loss and optimizer
    optimizer = optim.Adam(params=list(poi_embed_model.parameters()) +
                                  list(node_attn_model.parameters()) +
                                  list(user_embed_model.parameters()) +
                                  list(adj_embed_model.parameters()) +
                                  list(time_embed_model.parameters()) +
                                  list(cat_embed_model.parameters()) +
                                  list(poi_rep_model.parameters()) +
                                  list(time_focused_model.parameters()) +
                                  list(adj_focused_model.parameters()) +
                                  #list(pre_task1_model.parameters()) +
                                  list(seq_model.parameters())
                                  + list(task1_model.parameters())
                           , lr=args.lr, weight_decay=args.weight_decay)

    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss
    criterion_task1 = nn.CrossEntropyLoss(ignore_index=-1)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', verbose=True, factor=args.lr_scheduler_factor)

    # %% Tool functions for training
    def input_traj_to_embeddings(sample, poi_embeddings, task1_state):
        # Parse sample 0: traj_id, 1: traj_input, 2: traj_true; 3: user_id, 4: task1_pos, 5: task1_neg; [4098, 128]
        traj_id = sample[0]
        input_seq = [each[0] for each in sample[1]]
        input_seq_time = [each[1] for each in sample[1]]
        input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]
        pos_seq = [each for each in sample[4]]
        pos_seq_cat = [poi_idx2cat_idx_dict[each] for each in pos_seq]
        input_seq_embed_1, input_seq_embed_2 = [], []
        # User to embedding
        user_id = traj_id.split('_')[0]
        user_idx = user_id2idx_dict[user_id]
        input = torch.LongTensor([user_idx]).to(device=args.device)
        user_embedding = user_embed_model(input)
        user_embedding = torch.squeeze(user_embedding)
        adj_embedding = adj_embed_model(input, poi_embeddings)
        adj_embedding = torch.squeeze(adj_embedding)

        if task1_state == 'run':
            for idx in range(len(pos_seq)):
                poi_embedding = poi_embeddings[pos_seq[idx]]
                poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)
                cat_idx = torch.LongTensor([pos_seq_cat[idx]]).to(device=args.device)
                cat_embedding = cat_embed_model(cat_idx)
                cat_embedding = torch.squeeze(cat_embedding)
                poi_rep_embedding = poi_rep_model(poi_embedding, cat_embedding)
                concat_embedding_1 = adj_focused_model(adj_embedding, poi_rep_embedding)
                #concat_embedding_1 = adj_focused_model(user_embedding, poi_rep_embedding)
                input_seq_embed_1.append(concat_embedding_1)

        # POI to embedding and fuse embeddings

        for idx in range(len(input_seq)):
            poi_embedding = poi_embeddings[input_seq[idx]]
            poi_embedding = torch.squeeze(poi_embedding).to(device=args.device)

            # Time to vector
            time_embedding = time_embed_model(
                torch.tensor([input_seq_time[idx]], dtype=torch.float).to(device=args.device))
            time_embedding = torch.squeeze(time_embedding).to(device=args.device)

            # Categroy to embedding
            cat_idx = torch.LongTensor([input_seq_cat[idx]]).to(device=args.device)
            cat_embedding = cat_embed_model(cat_idx)
            cat_embedding = torch.squeeze(cat_embedding)

            # Fuse poi+time embeds
            poi_rep_embedding = poi_rep_model(poi_embedding, cat_embedding)
            time_rep_embedding = time_focused_model(user_embedding, time_embedding)

            # Concat
            concat_embedding_2 = torch.cat((poi_rep_embedding, time_rep_embedding), dim=-1)
            # Save final embed
            input_seq_embed_2.append(concat_embedding_2)

        return input_seq_embed_1, input_seq_embed_2

    def adjust_pred_prob_by_graph(y_pred_poi):
        y_pred_poi_adjusted = torch.zeros_like(y_pred_poi)
        attn_map = node_attn_model(X, A)

        for i in range(len(batch_seq_lens)):
            traj_i_input = batch_input_seqs[i]  # list of input check-in pois
            for j in range(len(traj_i_input)):
                y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

        return y_pred_poi_adjusted

    # %% ====================== Train ======================
    poi_embed_model = poi_embed_model.to(device=args.device)
    node_attn_model = node_attn_model.to(device=args.device)
    user_embed_model = user_embed_model.to(device=args.device)
    adj_embed_model = adj_embed_model.to(device=args.device)
    time_embed_model = time_embed_model.to(device=args.device)
    cat_embed_model = cat_embed_model.to(device=args.device)
    poi_rep_model = poi_rep_model.to(device=args.device)
    time_focused_model = time_focused_model.to(device=args.device)
    adj_focused_model = adj_focused_model.to(device=args.device)
    #pre_task1_model = pre_task1_model.to(device=args.device)
    seq_model = seq_model.to(device=args.device)
    task1_model = task1_model.to(device=args.device)

    # %% Loop epoch
    # For plotting
    train_epochs_top1_acc_list = []
    train_epochs_top5_acc_list = []
    train_epochs_top10_acc_list = []
    train_epochs_top20_acc_list = []
    train_epochs_mAP20_list = []
    train_epochs_mrr_list = []
    train_epochs_task1_top1_acc_list = []
    train_epochs_task1_top5_acc_list = []
    train_epochs_task1_top10_acc_list = []
    train_epochs_task1_top20_acc_list = []
    train_epochs_loss_list = []
    train_epochs_task1_loss_list = []
    train_epochs_poi_loss_list = []
    train_epochs_time_loss_list = []
    train_epochs_cat_loss_list = []
    val_epochs_top1_acc_list = []
    val_epochs_top5_acc_list = []
    val_epochs_top10_acc_list = []
    val_epochs_top20_acc_list = []
    val_epochs_mAP20_list = []
    val_epochs_mrr_list = []
    val_epochs_task1_top1_acc_list = []
    val_epochs_task1_top5_acc_list = []
    val_epochs_task1_top10_acc_list = []
    val_epochs_task1_top20_acc_list = []
    val_epochs_loss_list = []
    val_epochs_task1_loss_list = []
    val_epochs_poi_loss_list = []
    val_epochs_time_loss_list = []
    val_epochs_cat_loss_list = []
    # For saving ckpt
    max_val_score = -np.inf

    for epoch in range(args.epochs):
        logging.info(f"{'*' * 50}Epoch:{epoch:03d}{'*' * 50}\n")
        poi_embed_model.train()
        node_attn_model.train()
        user_embed_model.train()
        adj_embed_model.train()
        time_embed_model.train()
        cat_embed_model.train()
        poi_rep_model.train()
        time_focused_model.train()
        adj_focused_model.train()
        #pre_task1_model.train()
        seq_model.train()
        task1_model.train()

        train_batches_top1_acc_list = []
        train_batches_top5_acc_list = []
        train_batches_top10_acc_list = []
        train_batches_top20_acc_list = []
        train_batches_task1_top1_acc_list, train_batches_task1_top5_acc_list, train_batches_task1_top10_acc_list, train_batches_task1_top20_acc_list = [], [], [], []
        train_batches_mAP20_list = []
        train_batches_mrr_list = []
        train_batches_loss_list = []
        train_batches_task1_loss_list = []
        train_batches_poi_loss_list = []
        train_batches_time_loss_list = []
        train_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        task1_usersave = []
        # Loop batch
        for b_idx, batch in enumerate(train_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_task1_seq_lens = []
            batch_seq_embeds_1 = []
            batch_seq_embeds_2 = []
            batch_seq_labels_task1_poi = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A)
            #poi_embeddings = poi_embed_model(torch.LongTensor(list(range(num_pois))).to(device=args.device))
            
            # Convert input seq to embeddings
            for sample in batch:
                task1_state = 'stop'
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq， 3: user_id, 4: task1_pos, 5: task1_neg;
                user_id = sample[3]
                if user_id not in task1_usersave:
                    task1_usersave.append(user_id)
                    task1_state = 'run'
                    pos_seq = [each for each in sample[4]]
                    neg_seq = [sample[5]]
                    input_seq_embed_1, input_seq_embed_2 = input_traj_to_embeddings(sample, poi_embeddings, task1_state)
                    input_seq_embed_1 = torch.stack(input_seq_embed_1)
                    batch_seq_embeds_1.append(input_seq_embed_1)
                    batch_task1_seq_lens.append(len(pos_seq))
                    batch_seq_labels_task1_poi.extend(neg_seq)
                else:
                    _, input_seq_embed_2 = input_traj_to_embeddings(sample, poi_embeddings, task1_state)

                input_seq_embed_2 = torch.stack(input_seq_embed_2)
                batch_seq_embeds_2.append(input_seq_embed_2)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            if batch_seq_embeds_1 != []:
                batch_padded_1 = pad_sequence(batch_seq_embeds_1, batch_first=True, padding_value=-1)
                x1 = batch_padded_1.to(device=args.device, dtype=torch.float)
                y_task1_poi = torch.tensor(batch_seq_labels_task1_poi).to(device=args.device, dtype=torch.long)
                #y_pred_task1 = pre_task1_model(x1)
                y_pred_task1 = task1_model(x1)
                # [batch_size, id_num]; [batch_size]
                loss_task1 = criterion_task1(y_pred_task1, y_task1_poi)
            else: loss_task1 = torch.tensor(0, dtype=torch.float)

            batch_padded_2 = pad_sequence(batch_seq_embeds_2, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            # Feedforward
            x2 = batch_padded_2.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            # [batch_size, seq_len, id_num]
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x2, src_mask)
            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)
            # [batch_size, id_num, seq_len]; [batch_size, seq_len]
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)

            # Final loss
            loss = loss_task1*0.5 + (loss_poi + loss_time * args.time_loss_weight + loss_cat)*0.5
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            task1_top1_acc = 0
            task1_top5_acc = 0
            task1_top10_acc = 0
            task1_top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_task1_pois = y_task1_poi.detach().cpu().numpy()
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            batch_task1_pred = y_pred_task1.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for task1_pois, task1_pred in zip(batch_task1_pois, batch_task1_pred):
                task1_top1_acc += recall_at_k(task1_pois, task1_pred, k=1)
                task1_top5_acc += recall_at_k(task1_pois, task1_pred, k=5)
                task1_top10_acc += recall_at_k(task1_pois, task1_pred, k=10)
                task1_top20_acc += recall_at_k(task1_pois, task1_pred, k=20)

            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):

                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            train_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            train_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            train_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            train_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            train_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            train_batches_mrr_list.append(mrr / len(batch_label_pois))

            train_batches_task1_top1_acc_list.append(task1_top1_acc / len(batch_task1_pois))
            train_batches_task1_top5_acc_list.append(task1_top5_acc / len(batch_task1_pois))
            train_batches_task1_top10_acc_list.append(task1_top10_acc / len(batch_task1_pois))
            train_batches_task1_top20_acc_list.append(task1_top20_acc / len(batch_task1_pois))

            train_batches_loss_list.append(loss.detach().cpu().numpy())

            train_batches_task1_loss_list.append(loss_task1.detach().cpu().numpy())

            train_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            train_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            train_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report training progress
            if (b_idx % (args.batch * 5)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{b_idx}, '
                             f'train_batch_loss:{loss.item():.2f}, '
                             f'train_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'train_move_loss:{np.mean(train_batches_loss_list):.2f}\n'
                             f'train_move_task1_loss:{np.mean(train_batches_task1_loss_list):.2f}\n'
                             f'train_move_poi_loss:{np.mean(train_batches_poi_loss_list):.2f}\n'
                             f'train_move_time_loss:{np.mean(train_batches_time_loss_list):.2f}\n'
                             f'train_move_cat_loss:{np.mean(train_batches_cat_loss_list):.2f}\n'
                             f'train_move_top1_acc:{np.mean(train_batches_top1_acc_list):.4f}\n'
                             f'train_move_top5_acc:{np.mean(train_batches_top5_acc_list):.4f}\n'
                             f'train_move_top10_acc:{np.mean(train_batches_top10_acc_list):.4f}\n'
                             f'train_move_top20_acc:{np.mean(train_batches_top20_acc_list):.4f}\n'
                             f'train_move_mAP20:{np.mean(train_batches_mAP20_list):.4f}\n'
                             f'train_move_MRR:{np.mean(train_batches_mrr_list):.4f}\n'
                             f'train_move_task1_top1_acc:{np.mean(train_batches_task1_top1_acc_list):.4f}\n'
                             f'train_move_task1_top5_acc:{np.mean(train_batches_task1_top5_acc_list):.4f}\n'
                             f'train_move_task1_top10_acc:{np.mean(train_batches_task1_top10_acc_list):.4f}\n'
                             f'train_move_task1_top20_acc:{np.mean(train_batches_task1_top20_acc_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq: {batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'user_id:{batch[sample_idx][3]}\n'
                             f'pos_seq: {batch[sample_idx][4]}\n'
                             f'neg_seq:{batch[sample_idx][5]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             + '=' * 100)

        # train end --------------------------------------------------------------------------------------------------------
        poi_embed_model.eval()
        node_attn_model.eval()
        user_embed_model.eval()
        adj_embed_model.eval()
        time_embed_model.eval()
        cat_embed_model.eval()
        poi_rep_model.eval()
        time_focused_model.eval()
        adj_focused_model.eval()
        seq_model.eval()
        #pre_task1_model.eval()
        task1_model.eval()

        val_batches_top1_acc_list = []
        val_batches_top5_acc_list = []
        val_batches_top10_acc_list = []
        val_batches_top20_acc_list = []
        val_batches_task1_top1_acc_list = []
        val_batches_task1_top5_acc_list = []
        val_batches_task1_top10_acc_list = []
        val_batches_task1_top20_acc_list = []
        val_batches_mAP20_list = []
        val_batches_mrr_list = []
        val_batches_loss_list = []
        val_batches_task1_loss_list = []
        val_batches_poi_loss_list = []
        val_batches_time_loss_list = []
        val_batches_cat_loss_list = []
        src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(args.device)
        task1_usersave = []
        for vb_idx, batch in enumerate(val_loader):
            if len(batch) != args.batch:
                src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(args.device)

            # For padding
            batch_input_seqs = []
            batch_seq_lens = []
            batch_task1_seq_lens = []
            batch_seq_embeds_1 = []
            batch_seq_embeds_2 = []
            batch_seq_labels_task1_poi = []
            batch_seq_labels_poi = []
            batch_seq_labels_time = []
            batch_seq_labels_cat = []

            poi_embeddings = poi_embed_model(X, A)
            #poi_embeddings = poi_embed_model(torch.LongTensor(list(range(num_pois))).to(device=args.device))
            
            # Convert input seq to embeddings
            for sample in batch:
                task1_state = 'stop'
                traj_id = sample[0]
                input_seq = [each[0] for each in sample[1]]
                label_seq = [each[0] for each in sample[2]]
                label_seq_time = [each[1] for each in sample[2]]
                label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
                # sample[0]: traj_id, sample[1]: input_seq, sample[2]: label_seq， 3: user_id, 4: task1_pos, 5: task1_neg;
                user_id = sample[3]
                if user_id not in task1_usersave:
                    task1_usersave.append(user_id)
                    task1_state = 'run'
                    pos_seq = [each for each in sample[4]]
                    neg_seq = [sample[5]]
                    input_seq_embed_1, input_seq_embed_2 = input_traj_to_embeddings(sample, poi_embeddings, task1_state)
                    input_seq_embed_1 = torch.stack(input_seq_embed_1)
                    batch_seq_embeds_1.append(input_seq_embed_1)
                    batch_task1_seq_lens.append(len(pos_seq))
                    batch_seq_labels_task1_poi.extend(neg_seq)
                else:
                    _, input_seq_embed_2 = input_traj_to_embeddings(sample, poi_embeddings, task1_state)

                input_seq_embed_2 = torch.stack(input_seq_embed_2)
                batch_seq_embeds_2.append(input_seq_embed_2)
                batch_seq_lens.append(len(input_seq))
                batch_input_seqs.append(input_seq)
                batch_seq_labels_poi.append(torch.LongTensor(label_seq))
                batch_seq_labels_time.append(torch.FloatTensor(label_seq_time))
                batch_seq_labels_cat.append(torch.LongTensor(label_seq_cats))

            # Pad seqs for batch training
            if batch_seq_embeds_1 != []:
                batch_padded_1 = pad_sequence(batch_seq_embeds_1, batch_first=True, padding_value=-1)
                x1 = batch_padded_1.to(device=args.device, dtype=torch.float)
                y_task1_poi = torch.tensor(batch_seq_labels_task1_poi).to(device=args.device, dtype=torch.long)
                #y_pred_task1 = pre_task1_model(x1)
                y_pred_task1 = task1_model(x1)
                # [batch_size, id_num]; [batch_size]
                loss_task1 = criterion_task1(y_pred_task1, y_task1_poi)
            else:
                loss_task1 = torch.tensor(0, dtype=torch.float)

            batch_padded_2 = pad_sequence(batch_seq_embeds_2, batch_first=True, padding_value=-1)
            label_padded_poi = pad_sequence(batch_seq_labels_poi, batch_first=True, padding_value=-1)
            label_padded_time = pad_sequence(batch_seq_labels_time, batch_first=True, padding_value=-1)
            label_padded_cat = pad_sequence(batch_seq_labels_cat, batch_first=True, padding_value=-1)
            # Feedforward
            x2 = batch_padded_2.to(device=args.device, dtype=torch.float)
            y_poi = label_padded_poi.to(device=args.device, dtype=torch.long)
            y_time = label_padded_time.to(device=args.device, dtype=torch.float)
            y_cat = label_padded_cat.to(device=args.device, dtype=torch.long)
            # [batch_size, seq_len, id_num]
            y_pred_poi, y_pred_time, y_pred_cat = seq_model(x2, src_mask)
            # Graph Attention adjusted prob
            y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi)
            # [batch_size, id_num, seq_len]; [batch_size, seq_len]
            loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
            loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
            loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
            loss = loss_task1*0.5 + (loss_poi + loss_time * args.time_loss_weight + loss_cat)*0.5

            # Performance measurement
            top1_acc = 0
            top5_acc = 0
            top10_acc = 0
            top20_acc = 0
            task1_top1_acc = 0
            task1_top5_acc = 0
            task1_top10_acc = 0
            task1_top20_acc = 0
            mAP20 = 0
            mrr = 0
            batch_task1_pois = y_task1_poi.detach().cpu().numpy()
            batch_label_pois = y_poi.detach().cpu().numpy()
            batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()
            batch_task1_pred = y_pred_task1.detach().cpu().numpy()
            batch_pred_times = y_pred_time.detach().cpu().numpy()
            batch_pred_cats = y_pred_cat.detach().cpu().numpy()
            for task1_pois, task1_pred in zip(batch_task1_pois, batch_task1_pred):
                task1_top1_acc += recall_at_k(task1_pois, task1_pred, k=1)
                task1_top5_acc += recall_at_k(task1_pois, task1_pred, k=5)
                task1_top10_acc += recall_at_k(task1_pois, task1_pred, k=10)
                task1_top20_acc += recall_at_k(task1_pois, task1_pred, k=20)

            for label_pois, pred_pois, seq_len in zip(batch_label_pois, batch_pred_pois, batch_seq_lens):
                label_pois = label_pois[:seq_len]  # shape: (seq_len, )
                pred_pois = pred_pois[:seq_len, :]  # shape: (seq_len, num_poi)
                top1_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=1)
                top5_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=5)
                top10_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=10)
                top20_acc += top_k_acc_last_timestep(label_pois, pred_pois, k=20)
                mAP20 += mAP_metric_last_timestep(label_pois, pred_pois, k=20)
                mrr += MRR_metric_last_timestep(label_pois, pred_pois)

            val_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
            val_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
            val_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
            val_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
            val_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
            val_batches_mrr_list.append(mrr / len(batch_label_pois))

            val_batches_task1_top1_acc_list.append(task1_top1_acc / len(batch_task1_pois))
            val_batches_task1_top5_acc_list.append(task1_top5_acc / len(batch_task1_pois))
            val_batches_task1_top10_acc_list.append(task1_top10_acc / len(batch_task1_pois))
            val_batches_task1_top20_acc_list.append(task1_top20_acc / len(batch_task1_pois))

            val_batches_loss_list.append(loss.detach().cpu().numpy())
            val_batches_task1_loss_list.append(loss_task1.detach().cpu().numpy())
            val_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
            val_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
            val_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

            # Report validation progress
            if (vb_idx % (args.batch * 2)) == 0:
                sample_idx = 0
                batch_pred_pois_wo_attn = y_pred_poi.detach().cpu().numpy()
                logging.info(f'Epoch:{epoch}, batch:{vb_idx}, '
                             f'val_batch_loss:{loss.item():.2f}, '
                             f'val_batch_top1_acc:{top1_acc / len(batch_label_pois):.2f}, '
                             f'val_move_loss:{np.mean(val_batches_loss_list):.2f} \n'
                             f'val_move_task1_loss:{np.mean(val_batches_task1_loss_list):.2f}\n'
                             f'val_move_poi_loss:{np.mean(val_batches_poi_loss_list):.2f} \n'
                             f'val_move_time_loss:{np.mean(val_batches_time_loss_list):.2f} \n'
                             f'val_move_cat_loss:{np.mean(val_batches_cat_loss_list):.2f} \n'
                             f'val_move_top1_acc:{np.mean(val_batches_top1_acc_list):.4f} \n'
                             f'val_move_top5_acc:{np.mean(val_batches_top5_acc_list):.4f} \n'
                             f'val_move_top10_acc:{np.mean(val_batches_top10_acc_list):.4f} \n'
                             f'val_move_top20_acc:{np.mean(val_batches_top20_acc_list):.4f} \n'
                             f'val_move_mAP20:{np.mean(val_batches_mAP20_list):.4f} \n'
                             f'val_move_MRR:{np.mean(val_batches_mrr_list):.4f} \n'
                             f'val_move_task1_top1_rec:{np.mean(val_batches_task1_top1_acc_list):.4f}\n'
                             f'val_move_task1_top5_rec:{np.mean(val_batches_task1_top5_acc_list):.4f}\n'
                             f'val_move_task1_top10_rec:{np.mean(val_batches_task1_top10_acc_list):.4f}\n'
                             f'val_move_task1_top20_rec:{np.mean(val_batches_task1_top20_acc_list):.4f}\n'
                             f'traj_id:{batch[sample_idx][0]}\n'
                             f'input_seq:{batch[sample_idx][1]}\n'
                             f'label_seq:{batch[sample_idx][2]}\n'
                             f'user_id:{batch[sample_idx][3]}\n'
                             f'pos_seq: {batch[sample_idx][4]}\n'
                             f'neg_seq:{batch[sample_idx][5]}\n'
                             f'pred_seq_poi_wo_attn:{list(np.argmax(batch_pred_pois_wo_attn, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             f'pred_seq_poi:{list(np.argmax(batch_pred_pois, axis=2)[sample_idx][:batch_seq_lens[sample_idx]])} \n'
                             + '=' * 100)
        # valid end --------------------------------------------------------------------------------------------------------

        # Calculate epoch metrics
        epoch_train_top1_acc = np.mean(train_batches_top1_acc_list)
        epoch_train_top5_acc = np.mean(train_batches_top5_acc_list)
        epoch_train_top10_acc = np.mean(train_batches_top10_acc_list)
        epoch_train_top20_acc = np.mean(train_batches_top20_acc_list)
        epoch_train_mAP20 = np.mean(train_batches_mAP20_list)
        epoch_train_mrr = np.mean(train_batches_mrr_list)

        epoch_train_task1_top1_acc = np.mean(train_batches_task1_top1_acc_list)
        epoch_train_task1_top5_acc = np.mean(train_batches_task1_top5_acc_list)
        epoch_train_task1_top10_acc = np.mean(train_batches_task1_top10_acc_list)
        epoch_train_task1_top20_acc = np.mean(train_batches_task1_top20_acc_list)

        epoch_train_loss = np.mean(train_batches_loss_list)
        epoch_train_task1_loss = np.mean(train_batches_task1_loss_list)
        epoch_train_poi_loss = np.mean(train_batches_poi_loss_list)
        epoch_train_time_loss = np.mean(train_batches_time_loss_list)
        epoch_train_cat_loss = np.mean(train_batches_cat_loss_list)
        epoch_val_top1_acc = np.mean(val_batches_top1_acc_list)
        epoch_val_top5_acc = np.mean(val_batches_top5_acc_list)
        epoch_val_top10_acc = np.mean(val_batches_top10_acc_list)
        epoch_val_top20_acc = np.mean(val_batches_top20_acc_list)
        epoch_val_mAP20 = np.mean(val_batches_mAP20_list)
        epoch_val_mrr = np.mean(val_batches_mrr_list)

        epoch_val_task1_top1_acc = np.mean(val_batches_task1_top1_acc_list)
        epoch_val_task1_top5_acc = np.mean(val_batches_task1_top5_acc_list)
        epoch_val_task1_top10_acc = np.mean(val_batches_task1_top10_acc_list)
        epoch_val_task1_top20_acc = np.mean(val_batches_task1_top20_acc_list)

        epoch_val_loss = np.mean(val_batches_loss_list)
        epoch_val_task1_loss = np.mean(val_batches_task1_loss_list)
        epoch_val_poi_loss = np.mean(val_batches_poi_loss_list)
        epoch_val_time_loss = np.mean(val_batches_time_loss_list)
        epoch_val_cat_loss = np.mean(val_batches_cat_loss_list)

        # Save metrics to list
        train_epochs_loss_list.append(epoch_train_loss)
        train_epochs_task1_loss_list.append(epoch_train_task1_loss)
        train_epochs_poi_loss_list.append(epoch_train_poi_loss)
        train_epochs_time_loss_list.append(epoch_train_time_loss)
        train_epochs_cat_loss_list.append(epoch_train_cat_loss)
        train_epochs_top1_acc_list.append(epoch_train_top1_acc)
        train_epochs_top5_acc_list.append(epoch_train_top5_acc)
        train_epochs_top10_acc_list.append(epoch_train_top10_acc)
        train_epochs_top20_acc_list.append(epoch_train_top20_acc)
        train_epochs_mAP20_list.append(epoch_train_mAP20)
        train_epochs_mrr_list.append(epoch_train_mrr)
        train_epochs_task1_top1_acc_list.append(epoch_train_task1_top1_acc)
        train_epochs_task1_top5_acc_list.append(epoch_train_task1_top5_acc)
        train_epochs_task1_top10_acc_list.append(epoch_train_task1_top10_acc)
        train_epochs_task1_top20_acc_list.append(epoch_train_task1_top20_acc)
        val_epochs_loss_list.append(epoch_val_loss)
        val_epochs_task1_loss_list.append(epoch_val_task1_loss)
        val_epochs_poi_loss_list.append(epoch_val_poi_loss)
        val_epochs_time_loss_list.append(epoch_val_time_loss)
        val_epochs_cat_loss_list.append(epoch_val_cat_loss)
        val_epochs_top1_acc_list.append(epoch_val_top1_acc)
        val_epochs_top5_acc_list.append(epoch_val_top5_acc)
        val_epochs_top10_acc_list.append(epoch_val_top10_acc)
        val_epochs_top20_acc_list.append(epoch_val_top20_acc)
        val_epochs_mAP20_list.append(epoch_val_mAP20)
        val_epochs_mrr_list.append(epoch_val_mrr)
        val_epochs_task1_top1_acc_list.append(epoch_val_task1_top1_acc)
        val_epochs_task1_top5_acc_list.append(epoch_val_task1_top5_acc)
        val_epochs_task1_top10_acc_list.append(epoch_val_task1_top10_acc)
        val_epochs_task1_top20_acc_list.append(epoch_val_task1_top20_acc)

        # Monitor loss and score
        monitor_loss = epoch_val_loss
        monitor_score_1 = np.mean(epoch_val_task1_top1_acc + epoch_val_task1_top20_acc)
        monitor_score = monitor_score_1
        # Learning rate schuduler
        lr_scheduler.step(monitor_loss)

        # Print epoch results
        logging.info(f"Epoch {epoch}/{args.epochs}\n"
                     f"train_loss:{epoch_train_loss:.4f}, "
                     f"train_task1_loss:{epoch_train_task1_loss:.4f}, "
                     f"train_poi_loss:{epoch_train_poi_loss:.4f}, "
                     f"train_time_loss:{epoch_train_time_loss:.4f}, "
                     f"train_cat_loss:{epoch_train_cat_loss:.4f}, "
                     f"train_top1_acc:{epoch_train_top1_acc:.4f}, "
                     f"train_top5_acc:{epoch_train_top5_acc:.4f}, "
                     f"train_top10_acc:{epoch_train_top10_acc:.4f}, "
                     f"train_top20_acc:{epoch_train_top20_acc:.4f}, "
                     f"train_mAP20:{epoch_train_mAP20:.4f}, "
                     f"train_mrr:{epoch_train_mrr:.4f}\n"
                     f"train_task1_top1_rec:{epoch_train_task1_top1_acc:.4f}, "
                     f"train_task1_top5_rec:{epoch_train_task1_top5_acc:.4f}, "
                     f"train_task1_top10_rec:{epoch_train_task1_top10_acc:.4f}, "
                     f"train_task1_top20_rec:{epoch_train_task1_top20_acc:.4f}\n"
                     f"val_loss: {epoch_val_loss:.4f}, "
                     f"val_task1_loss: {epoch_val_task1_loss:.4f}, "
                     f"val_poi_loss: {epoch_val_poi_loss:.4f}, "
                     f"val_time_loss: {epoch_val_time_loss:.4f}, "
                     f"val_cat_loss: {epoch_val_cat_loss:.4f}, "
                     f"val_top1_acc:{epoch_val_top1_acc:.4f}, "
                     f"val_top5_acc:{epoch_val_top5_acc:.4f}, "
                     f"val_top10_acc:{epoch_val_top10_acc:.4f}, "
                     f"val_top20_acc:{epoch_val_top20_acc:.4f}, "
                     f"val_mAP20:{epoch_val_mAP20:.4f}, "
                     f"val_mrr:{epoch_val_mrr:.4f}\n "
                     f"val_task1_top1_rec:{epoch_val_task1_top1_acc:.4f}, "
                     f"val_task1_top5_rec:{epoch_val_task1_top5_acc:.4f}, "
                     f"val_task1_top10_rec:{epoch_val_task1_top10_acc:.4f}, "
                     f"val_task1_top20_rec:{epoch_val_task1_top20_acc:.4f}")

        # Save poi and user embeddings
        if args.save_embeds:
            embeddings_save_dir = os.path.join(args.save_dir, 'embeddings')
            if not os.path.exists(embeddings_save_dir): os.makedirs(embeddings_save_dir)
            # Save best epoch embeddings
            if monitor_score >= max_val_score:
                # Save poi embeddings
                poi_embeddings = poi_embed_model(X, A).detach().cpu().numpy()
                poi_embedding_list= [], []
                for poi_idx in range(len(poi_id2idx_dict)):
                    poi_embedding = poi_embeddings[poi_idx]
                    poi_embedding_list.append(poi_embedding)
                save_poi_embeddings = np.array(poi_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_poi_embeddings'), save_poi_embeddings)
                # Save user embeddings
                user_embedding_list = []
                for user_idx in range(len(user_id2idx_dict)):
                    input = torch.LongTensor([user_idx]).to(device=args.device)
                    user_embedding = user_embed_model(input).detach().cpu().numpy().flatten()
                    user_embedding_list.append(user_embedding)
                user_embeddings = np.array(user_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_user_embeddings'), user_embeddings)
                # Save cat embeddings
                cat_embedding_list = []
                for cat_idx in range(len(cat_id2idx_dict)):
                    input = torch.LongTensor([cat_idx]).to(device=args.device)
                    cat_embedding = cat_embed_model(input).detach().cpu().numpy().flatten()
                    cat_embedding_list.append(cat_embedding)
                cat_embeddings = np.array(cat_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_cat_embeddings'), cat_embeddings)
                # Save time embeddings
                time_embedding_list = []
                for time_idx in range(args.time_units):
                    input = torch.FloatTensor([time_idx]).to(device=args.device)
                    time_embedding = time_embed_model(input).detach().cpu().numpy().flatten()
                    time_embedding_list.append(time_embedding)
                time_embeddings = np.array(time_embedding_list)
                np.save(os.path.join(embeddings_save_dir, 'saved_time_embeddings'), time_embeddings)

        # Save model state dict
        if args.save_weights:
            state_dict = {
                'epoch': epoch,
                'poi_embed_state_dict': poi_embed_model.state_dict(),
                'node_attn_state_dict': node_attn_model.state_dict(),
                'user_embed_state_dict': user_embed_model.state_dict(),
                'adj_embed_state_dict': adj_embed_model.state_dict(),
                'time_embed_state_dict': time_embed_model.state_dict(),
                'cat_embed_state_dict': cat_embed_model.state_dict(),
                'poi_rep_state_dict': poi_rep_model.state_dict(),
                'time_focused_state_dict': time_focused_model.state_dict(),
                'adj_focused_state_dict': adj_focused_model.state_dict(),
                'seq_model_state_dict': seq_model.state_dict(),
                #'pre_task1_model_state_dict': pre_task1_model.state_dict(),
                'task1_model_state_dict': task1_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'user_id2idx_dict': user_id2idx_dict,
                'poi_id2idx_dict': poi_id2idx_dict,
                'cat_id2idx_dict': cat_id2idx_dict,
                'poi_idx2cat_idx_dict': poi_idx2cat_idx_dict,
                'user_poi_edges_dict': user_poi_edges,
                'dist_mat_dict': dist_mat,
                'args': args,
                'epoch_train_metrics': {
                    'epoch_train_loss': epoch_train_loss,
                    'epoch_train_task1_loss': epoch_train_task1_loss,
                    'epoch_train_poi_loss': epoch_train_poi_loss,
                    'epoch_train_time_loss': epoch_train_time_loss,
                    'epoch_train_cat_loss': epoch_train_cat_loss,
                    'epoch_train_top1_acc': epoch_train_top1_acc,
                    'epoch_train_top5_acc': epoch_train_top5_acc,
                    'epoch_train_top10_acc': epoch_train_top10_acc,
                    'epoch_train_top20_acc': epoch_train_top20_acc,
                    'epoch_train_mAP20': epoch_train_mAP20,
                    'epoch_train_mrr': epoch_train_mrr,
                    'epoch_train_task1_top1_acc': epoch_train_task1_top1_acc,
                    'epoch_train_task1_top5_acc': epoch_train_task1_top5_acc,
                    'epoch_train_task1_top10_acc': epoch_train_task1_top10_acc,
                    'epoch_train_task1_top20_acc': epoch_train_task1_top20_acc
                },
                'epoch_val_metrics': {
                    'epoch': epoch,
                    'epoch_val_loss': epoch_val_loss,
                    'epoch_val_task1_loss': epoch_val_task1_loss,
                    'epoch_val_poi_loss': epoch_val_poi_loss,
                    'epoch_val_time_loss': epoch_val_time_loss,
                    'epoch_val_cat_loss': epoch_val_cat_loss,
                    'epoch_val_top1_acc': epoch_val_top1_acc,
                    'epoch_val_top5_acc': epoch_val_top5_acc,
                    'epoch_val_top10_acc': epoch_val_top10_acc,
                    'epoch_val_top20_acc': epoch_val_top20_acc,
                    'epoch_val_mAP20': epoch_val_mAP20,
                    'epoch_val_mrr': epoch_val_mrr,
                    'epoch_val_task1_top1_rec': epoch_val_task1_top1_acc,
                    'epoch_val_task1_top5_rec': epoch_val_task1_top5_acc,
                    'epoch_val_task1_top10_rec': epoch_val_task1_top10_acc,
                    'epoch_val_task1_top20_rec': epoch_val_task1_top20_acc
                }
            }
            model_save_dir = os.path.join(args.save_dir, 'checkpoints')
            # Save best val score epoch
            if monitor_score >= max_val_score:
                if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
                torch.save(state_dict, rf"{model_save_dir}/best_epoch.state.pt")
                with open(rf"{model_save_dir}/best_epoch.txt", 'w') as f:
                    print(state_dict['epoch_val_metrics'], file=f)
                max_val_score = monitor_score

        # Save train/val metrics for plotting purpose
        with open(os.path.join(args.save_dir, 'metrics-train.txt'), "w") as f:
            print(f'train_epochs_loss_list={[float(f"{each:.4f}") for each in train_epochs_loss_list]}', file=f)
            print(f'train_epochs_task1_loss_list={[float(f"{each:.4f}") for each in train_epochs_task1_loss_list]}', file=f)
            print(f'train_epochs_poi_loss_list={[float(f"{each:.4f}") for each in train_epochs_poi_loss_list]}', file=f)
            print(f'train_epochs_time_loss_list={[float(f"{each:.4f}") for each in train_epochs_time_loss_list]}',
                  file=f)
            print(f'train_epochs_cat_loss_list={[float(f"{each:.4f}") for each in train_epochs_cat_loss_list]}', file=f)
            print(f'train_epochs_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_top1_acc_list]}', file=f)
            print(f'train_epochs_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_top5_acc_list]}', file=f)
            print(f'train_epochs_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_top10_acc_list]}',
                  file=f)
            print(f'train_epochs_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_top20_acc_list]}',
                  file=f)
            print(f'train_epochs_mAP20_list={[float(f"{each:.4f}") for each in train_epochs_mAP20_list]}', file=f)
            print(f'train_epochs_mrr_list={[float(f"{each:.4f}") for each in train_epochs_mrr_list]}', file=f)

            print(f'train_epochs_task1_top1_acc_list={[float(f"{each:.4f}") for each in train_epochs_task1_top1_acc_list]}', file=f)
            print(f'train_epochs_task1_top5_acc_list={[float(f"{each:.4f}") for each in train_epochs_task1_top5_acc_list]}', file=f)
            print(f'train_epochs_task1_top10_acc_list={[float(f"{each:.4f}") for each in train_epochs_task1_top10_acc_list]}', file=f)
            print(f'train_epochs_task1_top20_acc_list={[float(f"{each:.4f}") for each in train_epochs_task1_top20_acc_list]}', file=f)
        with open(os.path.join(args.save_dir, 'metrics-val.txt'), "w") as f:
            print(f'val_epochs_loss_list={[float(f"{each:.4f}") for each in val_epochs_loss_list]}', file=f)
            print(f'val_epochs_task1_loss_list={[float(f"{each:.4f}") for each in val_epochs_task1_loss_list]}', file=f)
            print(f'val_epochs_poi_loss_list={[float(f"{each:.4f}") for each in val_epochs_poi_loss_list]}', file=f)
            print(f'val_epochs_time_loss_list={[float(f"{each:.4f}") for each in val_epochs_time_loss_list]}', file=f)
            print(f'val_epochs_cat_loss_list={[float(f"{each:.4f}") for each in val_epochs_cat_loss_list]}', file=f)
            print(f'val_epochs_top1_acc_list={[float(f"{each:.4f}") for each in val_epochs_top1_acc_list]}', file=f)
            print(f'val_epochs_top5_acc_list={[float(f"{each:.4f}") for each in val_epochs_top5_acc_list]}', file=f)
            print(f'val_epochs_top10_acc_list={[float(f"{each:.4f}") for each in val_epochs_top10_acc_list]}', file=f)
            print(f'val_epochs_top20_acc_list={[float(f"{each:.4f}") for each in val_epochs_top20_acc_list]}', file=f)
            print(f'val_epochs_mAP20_list={[float(f"{each:.4f}") for each in val_epochs_mAP20_list]}', file=f)
            print(f'val_epochs_mrr_list={[float(f"{each:.4f}") for each in val_epochs_mrr_list]}', file=f)
            print(f'val_epochs_task1_top1_rec_list={[float(f"{each:.4f}") for each in val_epochs_task1_top1_acc_list]}', file=f)
            print(f'val_epochs_task1_top5_rec_list={[float(f"{each:.4f}") for each in val_epochs_task1_top5_acc_list]}',
                  file=f)
            print(f'val_epochs_task1_top10_rec_list={[float(f"{each:.4f}") for each in val_epochs_task1_top10_acc_list]}',
                  file=f)
            print(f'val_epochs_task1_top20_rec_list={[float(f"{each:.4f}") for each in val_epochs_task1_top20_acc_list]}',
                  file=f)


if __name__ == '__main__':
    args = parameter_parser()
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    #args.feature2 = 'poi_catid_code'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    train(args)
