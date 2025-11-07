
import torch
import torch.nn as nn

class GRUWithStatics(nn.Module):
    def __init__(self, f_dyn, hidden=192, layers=1, dropout=0.1,
                 cat_card=None, cat_emb_dim=8, static_cont_dim=2,
                 use_well_emb=True, num_wells=1, well_emb_dim=8, horizon=1):
        super().__init__()
        self.gru = nn.GRU(f_dyn, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout if layers>1 else 0.0)

        self.cat_embs = nn.ModuleDict()
        total_cat_dim = 0
        if cat_card:
            for name, K in cat_card.items():
                self.cat_embs[name] = nn.Embedding(K, cat_emb_dim)
            total_cat_dim = cat_emb_dim * len(cat_card)

        self.use_well_emb = use_well_emb
        self.well_emb = nn.Embedding(num_wells, well_emb_dim) if use_well_emb else None

        static_in = static_cont_dim + total_cat_dim + (well_emb_dim if use_well_emb else 0)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        self.head = nn.Linear(hidden + 64, horizon)

    def forward(self, x_dyn, x_stat_cont, x_stat_cat, well_id):
        h_seq, _ = self.gru(x_dyn)
        h_last = h_seq[:, -1, :]

        cat_vecs = []
        if self.cat_embs:
            for i, name in enumerate(self.cat_embs.keys()):
                cat_vecs.append(self.cat_embs[name](x_stat_cat[:, i]))
        cat_vec = torch.cat(cat_vecs, dim=-1) if cat_vecs else None
        well_vec = self.well_emb(well_id) if self.use_well_emb else None

        parts = [x_stat_cont]
        if cat_vec is not None: parts.append(cat_vec)
        if well_vec is not None: parts.append(well_vec)
        static_all = torch.cat(parts, dim=-1)
        static_feat = self.static_mlp(static_all)

        yhat = self.head(torch.cat([h_last, static_feat], dim=-1))
        return yhat
    


class LSTMWithStatics(nn.Module):
    def __init__(self, f_dyn, hidden=192, layers=1, dropout=0.1,
                 cat_card=None, cat_emb_dim=8, static_cont_dim=2,
                 use_well_emb=True, num_wells=1, well_emb_dim=8, horizon=1,
                 bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(
            input_size=f_dyn,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # categorical embeddings (same as GRU)
        self.cat_embs = nn.ModuleDict()
        total_cat_dim = 0
        if cat_card:
            for name, K in cat_card.items():
                self.cat_embs[name] = nn.Embedding(K, cat_emb_dim)
            total_cat_dim = cat_emb_dim * len(cat_card)

        # well embedding
        self.use_well_emb = use_well_emb
        self.well_emb = nn.Embedding(num_wells, well_emb_dim) if use_well_emb else None

        # static MLP
        static_in = static_cont_dim + total_cat_dim + (well_emb_dim if use_well_emb else 0)
        self.static_mlp = nn.Sequential(
            nn.Linear(static_in, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )

        # head (note: hidden doubles if bidirectional)
        rnn_out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Linear(rnn_out_dim + 64, horizon)

    def forward(self, x_dyn, x_stat_cont, x_stat_cat, well_id):
        # LSTM returns (output, (h_n, c_n)); we’ll use output’s last timestep for consistency
        h_seq, _ = self.rnn(x_dyn)          # (B, T, H*(1 or 2))
        h_last = h_seq[:, -1, :]            # last time step

        # cat embeddings
        cat_vecs = []
        if self.cat_embs:
            for i, name in enumerate(self.cat_embs.keys()):
                cat_vecs.append(self.cat_embs[name](x_stat_cat[:, i]))
        cat_vec = torch.cat(cat_vecs, dim=-1) if cat_vecs else None

        # well embedding
        well_vec = self.well_emb(well_id) if self.use_well_emb else None

        # combine statics
        parts = [x_stat_cont]
        if cat_vec is not None: parts.append(cat_vec)
        if well_vec is not None: parts.append(well_vec)
        static_all = torch.cat(parts, dim=-1)
        static_feat = self.static_mlp(static_all)

        # prediction
        yhat = self.head(torch.cat([h_last, static_feat], dim=-1))
        return yhat

