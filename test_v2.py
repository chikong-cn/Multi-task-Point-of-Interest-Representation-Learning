import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_graph_adj_mtx, load_graph_node_features
from model_v2 import (gen_loc_graph, gen_nei_graph, GCN, GCNWithDistMatrix, NodeAttnMap, UserEmbeddings, POIEmbeddings, AdjEmbeddings, Time2Vec, CategoryEmbeddings,
                        TransformerModel, Poi_representation, FuseBlock, Task1Model, Pretask1)
from param_parser import parameter_parser
from utils import increment_path, calculate_laplacian_matrix, zipdir, top_k_acc_last_timestep, recall_at_k, \
    mAP_metric_last_timestep, MRR_metric_last_timestep, maksed_mse_loss
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class TrajectoryDatasetTest(Dataset):
    def __init__(self, test_df, poi_id2idx_dict, user_id2idx_dict, poi_idx2cat_idx_dict, time_feature, short_traj_thres):
        self.df = test_df
        self.traj_seqs = []
        self.input_seqs = []
        self.label_seqs = []
        self.user_seqs = []
        self.pos_seqs = []
        self.neg_seqs = []
        user_pos_neg_seqs = {}
        for user_id in tqdm(set(test_df['user_id'].to_list())):
            # Ignore user if not in training set
            if str(user_id) not in user_id2idx_dict.keys():
                continue
            user_df = test_df[test_df['user_id'] == user_id]
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

        for traj_id in tqdm(set(test_df['trajectory_id'].tolist())):
            user_id = traj_id.split('_')[0]

            if user_id not in user_id2idx_dict.keys():
                continue

            traj_df = test_df[test_df['trajectory_id'] == traj_id]
            poi_ids = traj_df['POI_id'].to_list()
            poi_idxs = []
            time_feature_vals = traj_df[time_feature].to_list()

            for each in poi_ids:
                if each in poi_id2idx_dict.keys():
                    poi_idxs.append(poi_id2idx_dict[each])
                else:
                    continue

            input_seq = []
            label_seq = []
            for i in range(len(poi_idxs) - 1):
                input_seq.append((poi_idxs[i], time_feature_vals[i]))
                label_seq.append((poi_idxs[i + 1], time_feature_vals[i + 1]))

            # Ignore seq if too short
            if len(input_seq) < short_traj_thres:
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


def input_traj_to_embeddings(sample, poi_embeddings, user_embed_model, adj_embed_model, time_embed_model, cat_embed_model, poi_rep_model, time_focused_model, adj_focused_model, poi_idx2cat_idx_dict, user_id2idx_dict, task1_state, device):
    traj_id = sample[0]
    input_seq = [each[0] for each in sample[1]]
    input_seq_time = [each[1] for each in sample[1]]
    input_seq_cat = [poi_idx2cat_idx_dict[each] for each in input_seq]
    pos_seq = [each for each in sample[4]]
    pos_seq_cat = [poi_idx2cat_idx_dict[each] for each in pos_seq]
    input_seq_embed_1, input_seq_embed_2 = [], []

    user_id = traj_id.split('_')[0]
    user_idx = user_id2idx_dict[user_id]
    input = torch.LongTensor([user_idx]).to(device)
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
            poi_rep_embedding = poi_rep_model(user_embedding, cat_embedding)
            #concat_embedding_1 = adj_focused_model(user_embedding, poi_rep_embedding)
            concat_embedding_1 = adj_focused_model(adj_embedding, poi_rep_embedding)
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

def adjust_pred_prob_by_graph(y_pred_poi, node_attn_model, X, A, batch_input_seqs, batch_seq_lens):
    y_pred_poi_adjusted = torch.zeros_like(y_pred_poi).to(device)
    attn_map = node_attn_model(X, A).to(device)

    for i in range(len(batch_seq_lens)):
        traj_i_input = batch_input_seqs[i]
        for j in range(len(traj_i_input)):
            y_pred_poi_adjusted[i, j, :] = attn_map[traj_i_input[j], :] + y_pred_poi[i, j, :]

    return y_pred_poi_adjusted

def test(args):

    # Load test data
    test_df = pd.read_csv(args.data_test, encoding='GBK')

    # Load graph and node features
    raw_A = load_graph_adj_mtx(args.data_adj_mtx)
    raw_X = load_graph_node_features(args.data_node_feats,
                                     args.feature1,
                                     args.feature2,
                                     args.feature3,
                                     args.feature4)

    num_pois = raw_X.shape[0]

    # One-hot encoding poi categories
    with open(os.path.join(args.checkpoint_dir, 'one-hot-encoder.pkl'), "rb") as f:
        one_hot_encoder = pickle.load(f)

    cat_list = list(raw_X[:, 1])
    one_hot_rlt = one_hot_encoder.transform(list(map(lambda x: [x], cat_list))).toarray()
    num_cats = one_hot_rlt.shape[-1]
    X = np.zeros((num_pois, raw_X.shape[-1] - 1 + num_cats), dtype=np.float32)
    X[:, 0] = raw_X[:, 0]
    X[:, 1:num_cats + 1] = one_hot_rlt
    X[:, num_cats + 1:] = raw_X[:, 2:]

    # Normalization
    A = calculate_laplacian_matrix(raw_A, mat_type='hat_rw_normd_lap_mat')
    X = torch.from_numpy(X).to(device, dtype=torch.float)
    A = torch.from_numpy(A).to(device, dtype=torch.float)

    # Load model
    checkpoint = torch.load(rf"{args.checkpoint_dir}/checkpoints/best_epoch.state.pt", map_location=device)
    args = checkpoint['args']

    user_id2idx_dict = checkpoint['user_id2idx_dict']
    poi_id2idx_dict = checkpoint['poi_id2idx_dict']
    poi_idx2cat_idx_dict = checkpoint['poi_idx2cat_idx_dict']
    user_poi_edges = checkpoint['user_poi_edges_dict']
    dist_mat = checkpoint['dist_mat_dict']

    #poi_embed_model = GCN(ninput=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.poi_embed_dim, dropout=args.gcn_dropout)
    poi_embed_model = GCNWithDistMatrix(in_features=args.gcn_nfeat, nhid=args.gcn_nhid, noutput=args.poi_embed_dim, adj_matrix=A, dist_matrix=dist_mat, dropout=args.gcn_dropout).to(device=args.device)
    node_attn_model = NodeAttnMap(in_features=X.shape[1], nhid=args.node_attn_nhid, use_mask=False)
    num_pois = len(poi_id2idx_dict)
    #poi_embed_model = POIEmbeddings(num_pois, args.poi_embed_dim)
    user_embed_model = UserEmbeddings(len(user_id2idx_dict), args.user_embed_dim)
    adj_embed_model = AdjEmbeddings(len(user_id2idx_dict), args.user_embed_dim, user_poi_edges=user_poi_edges)
    time_embed_model = Time2Vec('sin', out_dim=args.time_embed_dim)
    cat_embed_model = CategoryEmbeddings(num_cats, args.cat_embed_dim)

    poi_rep_model = Poi_representation(args.poi_embed_dim, args.cat_embed_dim)
    time_focused_model = FuseBlock(args.user_embed_dim, args.time_embed_dim)
    adj_focused_model = FuseBlock(args.user_embed_dim, args.poi_embed_dim + args.cat_embed_dim)
    #pre_task1_model = Pretask1(num_pois, args.task1_input_embed)
    task1_model = Task1Model(num_pois, args.task1_input_embed, d_ffn=256, num_layers=2)
    seq_model = TransformerModel(num_pois,
                                 num_cats,
                                 args.seq_input_embed,
                                 args.transformer_nhead,
                                 args.transformer_nhid,
                                 args.transformer_nlayers,
                                 dropout=args.transformer_dropout)

    criterion_task1 = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_poi = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_cat = nn.CrossEntropyLoss(ignore_index=-1)  # -1 is padding
    criterion_time = maksed_mse_loss

    poi_embed_model.load_state_dict(checkpoint['poi_embed_state_dict'])
    node_attn_model.load_state_dict(checkpoint['node_attn_state_dict'])
    user_embed_model.load_state_dict(checkpoint['user_embed_state_dict'])
    adj_embed_model.load_state_dict(checkpoint['adj_embed_state_dict'])
    time_embed_model.load_state_dict(checkpoint['time_embed_state_dict'])
    cat_embed_model.load_state_dict(checkpoint['cat_embed_state_dict'])

    poi_rep_model.load_state_dict(checkpoint['poi_rep_state_dict'])
    time_focused_model.load_state_dict(checkpoint['time_focused_state_dict'])
    adj_focused_model.load_state_dict(checkpoint['adj_focused_state_dict'])

    seq_model.load_state_dict(checkpoint['seq_model_state_dict'])
    #pre_task1_model.load_state_dict(checkpoint['pre_task1_model_state_dict'])
    task1_model.load_state_dict(checkpoint['task1_model_state_dict'])

    poi_embed_model = poi_embed_model.to(device)
    user_embed_model = user_embed_model.to(device)
    adj_embed_model = adj_embed_model.to(device)
    time_embed_model = time_embed_model.to(device)
    cat_embed_model = cat_embed_model.to(device)
    poi_rep_model = poi_rep_model.to(device)
    time_focused_model = time_focused_model.to(device)
    adj_focused_model = adj_focused_model.to(device)

    seq_model = seq_model.to(device)
    #pre_task1_model = pre_task1_model.to(device)
    task1_model = task1_model.to(device)

    # Prepare test dataloader
    test_dataset = TrajectoryDatasetTest(test_df, poi_id2idx_dict, user_id2idx_dict, poi_idx2cat_idx_dict, args.time_feature, args.short_traj_thres)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch,
                             shuffle=False, drop_last=False,
                             pin_memory=True, num_workers=args.workers,
                             collate_fn=lambda x: x)

    # Evaluation
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

    test_batches_top1_acc_list = []
    test_batches_top5_acc_list = []
    test_batches_top10_acc_list = []
    test_batches_top20_acc_list = []
    test_batches_mAP20_list = []
    test_batches_mrr_list = []
    test_batches_task1_top1_acc_list = []
    test_batches_task1_top5_acc_list = []
    test_batches_task1_top10_acc_list = []
    test_batches_task1_top20_acc_list = []
    test_batches_loss_list = []
    test_batches_task1_loss_list = []
    test_batches_poi_loss_list = []
    test_batches_time_loss_list = []
    test_batches_cat_loss_list = []
    src_mask = seq_model.generate_square_subsequent_mask(args.batch).to(device)
    task1_usersave = []
    for batch in test_loader:
        if len(batch) != args.batch:
            src_mask = seq_model.generate_square_subsequent_mask(len(batch)).to(device)

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

        for sample in batch:
            task1_state = 'stop'
            traj_id = sample[0]
            input_seq = [each[0] for each in sample[1]]
            label_seq = [each[0] for each in sample[2]]
            label_seq_time = [each[1] for each in sample[2]]
            label_seq_cats = [poi_idx2cat_idx_dict[each] for each in label_seq]
            user_id = sample[3]
            if user_id not in task1_usersave:
                task1_usersave.append(user_id)
                task1_state = 'run'
                pos_seq = [each for each in sample[4]]
                neg_seq = [sample[5]]
                input_seq_embed_1, input_seq_embed_2 = input_traj_to_embeddings(sample, poi_embeddings, user_embed_model, adj_embed_model,
                                                                                time_embed_model, cat_embed_model, poi_rep_model, time_focused_model, adj_focused_model,
                                                                                poi_idx2cat_idx_dict, user_id2idx_dict,
                                                                                task1_state, device)
                input_seq_embed_1 = torch.stack(input_seq_embed_1)
                batch_seq_embeds_1.append(input_seq_embed_1)
                batch_task1_seq_lens.append(len(pos_seq))
                batch_seq_labels_task1_poi.append(torch.LongTensor(neg_seq))
            else:
                _, input_seq_embed_2 = input_traj_to_embeddings(sample, poi_embeddings, user_embed_model, adj_embed_model, time_embed_model, cat_embed_model, poi_rep_model, time_focused_model, adj_focused_model, poi_idx2cat_idx_dict, user_id2idx_dict, task1_state, device)

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
        y_pred_poi_adjusted = adjust_pred_prob_by_graph(y_pred_poi, node_attn_model, X, A, batch_input_seqs, batch_seq_lens)
        # [batch_size, id_num, seq_len]; [batch_size, seq_len]
        #loss_poi = criterion_poi(y_pred_poi.transpose(1, 2), y_poi)
        loss_poi = criterion_poi(y_pred_poi_adjusted.transpose(1, 2), y_poi)
        #loss_time = criterion_time(torch.squeeze(y_pred_time), y_time)
        loss_cat = criterion_cat(y_pred_cat.transpose(1, 2), y_cat)
        loss = loss_task1 + loss_poi + loss_cat

        top1_acc = 0
        top5_acc = 0
        top10_acc = 0
        top20_acc = 0
        mAP20 = 0
        mrr = 0
        task1_top1_acc = 0
        task1_top5_acc = 0
        task1_top10_acc = 0
        task1_top20_acc = 0
        batch_task1_pois = y_task1_poi.detach().cpu().numpy()
        batch_label_pois = y_poi.detach().cpu().numpy()
        batch_task1_pred = y_pred_task1.detach().cpu().numpy()
        #batch_pred_pois = y_pred_poi.detach().cpu().numpy()
        batch_pred_pois = y_pred_poi_adjusted.detach().cpu().numpy()

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

        test_batches_top1_acc_list.append(top1_acc / len(batch_label_pois))
        test_batches_top5_acc_list.append(top5_acc / len(batch_label_pois))
        test_batches_top10_acc_list.append(top10_acc / len(batch_label_pois))
        test_batches_top20_acc_list.append(top20_acc / len(batch_label_pois))
        test_batches_mAP20_list.append(mAP20 / len(batch_label_pois))
        test_batches_mrr_list.append(mrr / len(batch_label_pois))
        
        test_batches_task1_top1_acc_list.append(task1_top1_acc / len(batch_task1_pois))
        test_batches_task1_top5_acc_list.append(task1_top5_acc / len(batch_task1_pois))
        test_batches_task1_top10_acc_list.append(task1_top10_acc / len(batch_task1_pois))
        test_batches_task1_top20_acc_list.append(task1_top20_acc / len(batch_task1_pois))

        test_batches_loss_list.append(loss.detach().cpu().numpy())
        test_batches_task1_loss_list.append(loss_task1.detach().cpu().numpy())
        test_batches_poi_loss_list.append(loss_poi.detach().cpu().numpy())
        #test_batches_time_loss_list.append(loss_time.detach().cpu().numpy())
        test_batches_cat_loss_list.append(loss_cat.detach().cpu().numpy())

    epoch_test_top1_acc = np.mean(test_batches_top1_acc_list)
    epoch_test_top5_acc = np.mean(test_batches_top5_acc_list)
    epoch_test_top10_acc = np.mean(test_batches_top10_acc_list)
    epoch_test_top20_acc = np.mean(test_batches_top20_acc_list)
    epoch_test_mAP20 = np.mean(test_batches_mAP20_list)
    epoch_test_mrr = np.mean(test_batches_mrr_list)
    epoch_test_task1_top1_acc = np.mean(test_batches_task1_top1_acc_list)
    epoch_test_task1_top5_acc = np.mean(test_batches_task1_top5_acc_list)
    epoch_test_task1_top10_acc = np.mean(test_batches_task1_top10_acc_list)
    epoch_test_task1_top20_acc = np.mean(test_batches_task1_top20_acc_list)
    epoch_test_loss = np.mean(test_batches_loss_list)
    epoch_test_task1_loss = np.mean(test_batches_task1_loss_list)
    epoch_test_poi_loss = np.mean(test_batches_poi_loss_list)
    #epoch_test_time_loss = np.mean(test_batches_time_loss_list)
    epoch_test_cat_loss = np.mean(test_batches_cat_loss_list)

    results = {
        "test_loss": epoch_test_loss,
        "test_task1_loss": epoch_test_task1_loss,
        "test_poi_loss": epoch_test_poi_loss,
        #"test_time_loss": epoch_test_time_loss,
        "test_cat_loss": epoch_test_cat_loss,
        "test_top1_acc": epoch_test_top1_acc,
        "test_top5_acc": epoch_test_top5_acc,
        "test_top10_acc": epoch_test_top10_acc,
        "test_top20_acc": epoch_test_top20_acc,
        "test_mAP20": epoch_test_mAP20,
        "test_mrr": epoch_test_mrr,
        "test_task1_top1_rec": epoch_test_task1_top1_acc,
        "test_task1_top5_rec": epoch_test_task1_top5_acc,
        "test_task1_top10_rec": epoch_test_task1_top10_acc,
        "test_task1_top20_rec": epoch_test_task1_top20_acc
    }

    return results

def calculate_average(results_list):
    average_results = {}
    for key in results_list[0].keys():
        values = [result[key] for result in results_list]
        values.remove(max(values))
        values.remove(min(values))
        average_results[key] = np.mean(values)
    return average_results

if __name__ == '__main__':
    # Load arguments
    args = parameter_parser()
    # Set test data path
    args.data_test = 'dataset/Foursquare/Foursquare_test.csv'
    args.checkpoint_dir = 'runs/train/Foursquare_a05_gmlp_model'
    # The name of node features in NYC/graph_X.csv
    args.feature1 = 'checkin_cnt'
    args.feature2 = 'poi_catid'
    #args.feature2 = 'poi_catid_code'
    args.feature3 = 'latitude'
    args.feature4 = 'longitude'
    all_results = []

    for i in range(7):
        results = test(args)
        all_results.append(results)

    avg_results = calculate_average(all_results)

    print(f"Average Test Results over 10 runs (excluding max and min):\n"
          f"test_loss: {avg_results['test_loss']:.4f}, "
          f"test_task1_loss: {avg_results['test_task1_loss']:.4f}, "
          f"test_poi_loss: {avg_results['test_poi_loss']:.4f}, "
          #f"test_time_loss: {avg_results['test_time_loss']:.4f}, "
          f"test_cat_loss: {avg_results['test_cat_loss']:.4f}\n"
          f"test_top1_acc: {avg_results['test_top1_acc']:.4f}, "
          f"test_top5_acc: {avg_results['test_top5_acc']:.4f}, "
          f"test_top10_acc: {avg_results['test_top10_acc']:.4f}, "
          f"test_top20_acc: {avg_results['test_top20_acc']:.4f}, "
          f"test_mAP20: {avg_results['test_mAP20']:.4f}, "
          f"test_mrr: {avg_results['test_mrr']:.4f}\n"
          f"test_task1_top1_rec: {avg_results['test_task1_top1_rec']:.4f}, "
          f"test_task1_top5_rec: {avg_results['test_task1_top5_rec']:.4f}, "
          f"test_task1_top10_rec: {avg_results['test_task1_top10_rec']:.4f}, "
          f"test_task1_top20_rec: {avg_results['test_task1_top20_rec']:.4f}, "
          )
