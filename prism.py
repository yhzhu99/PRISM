import math

import torch
from torch import nn
import torch.nn.functional as F
import einops

class AdjacencyLayer(nn.Module):
    """
    input_dim: num_features
    current: num_features*feat_dim
    """
    def __init__(self):
        super(AdjacencyLayer, self).__init__()
        self.dc_param = nn.Parameter(torch.Tensor([0.05]))

    def forward(
            self, 
            x: torch.Tensor,        # x -> [bs + num_clusters, input_dim]
            dc: torch.Tensor,       # dc -> [bs + num_clusters, input_dim]
        ):
        b, d = x.shape
        x1 = x.unsqueeze(1).expand(-1, b, d)
        x2 = x.unsqueeze(0).expand(b, -1, d)

        dc1 = dc.unsqueeze(1).expand(-1, b, d)
        dc2 = dc.unsqueeze(0).expand(b, -1, d)
        similarity_score = 1 / ((1 - self.dc_param) * (x1 - x2) ** 2 + \
                                self.dc_param * torch.exp(1-dc1) * torch.exp(1-dc2)).mean(2)

        eye = torch.eye(b, device=x.device)
        similarity_score = similarity_score * (1 - eye) + eye

        return similarity_score
    

class GCNLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        bias: bool=True
    ):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight = nn.Parameter(torch.Tensor(input_dim, hidden_dim).float())
        if bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_dim).float())
        else:
            self.register_parameter('bias', None)
        self.initialize_parameters()

    def initialize_parameters(self):
        std = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.uniform_(-std, std)

    def forward(self, x, adj):
        y = torch.mm(x.float(), self.weight.float())
        output = torch.mm(adj.float(), y.float())
        if self.bias is not None:
            return output + self.bias.float()
        else:
            return output   # [bs, hidden_dim]

class SqueezeLayer(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        squeeze_dim: int=16,
    ):
        super(SqueezeLayer, self).__init__()
        self.fc1 = nn.Linear(input_dim, squeeze_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(squeeze_dim, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class GroupPatientLearnerModule(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        centers: torch.Tensor,
        hidden_dim: int=32, 
    ):
        super(GroupPatientLearnerModule, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.centers = centers
        self.first = True
        if self.first:
            self.centers_proj = nn.Linear(61, 19)
        self.adjacency_layer = AdjacencyLayer()
        self.gcn_layer = GCNLayer(input_dim, hidden_dim)

        self.embedding_out_proj = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.center_confidence = nn.Parameter(torch.ones(self.centers.shape[0], 19, dtype=torch.float32))

        self.importance_learner = SqueezeLayer(input_dim, squeeze_dim=16)


    def forward(self, 
        x: torch.Tensor,                # x -> [bs, num_feat, feat_dim]
        dc: torch.Tensor,               # dc -> [bs, num_feat]
    ):
        b, d, f = x.shape
        x_avg = x.mean(dim=-1)
        self.centers = self.centers.to(x.device)
        if self.first:
            self.centers = self.centers_proj(self.centers)
            self.first = False
        x_all = torch.cat([x_avg, self.centers]).detach() # [bs + num_clusters, input_dim]
        dc_all = torch.cat([dc, self.center_confidence]).detach() # [bs + num_clusters, input_dim]
        adj = self.adjacency_layer(x_all, dc_all) # [bs + num_clusters, bs + num_clusters]
        group_patients_adjacency = adj[:b, b:] # [bs, num_clusters]
        group_label = group_patients_adjacency.argmax(dim=1) # [bs]
        embedding = self.gcn_layer(x_all, adj) # [bs + num_clusters, hidden_dim]
        embedding = self.embedding_out_proj(embedding) # [bs + num_clusters, input_dim]

        group_embedding = embedding[b:] # [num_clusters, input_dim]
        group_patients_embedding = group_embedding[group_label] # [bs, input_dim]
        self.centers = group_embedding

        group_scores = self.importance_learner(group_embedding) # [num_clusters, num_features]
        group_scores = group_scores[group_label]  # [bs, num_features]
        return group_scores, group_embedding, group_patients_embedding, group_label

class ConfidenceLearner(nn.Module):
    def __init__(self,feat_num:int = 19):
        super().__init__()
        self.decay_term = nn.Parameter(torch.Tensor([0.8]*feat_num))
        self.global_missing_param = nn.Parameter(torch.Tensor([0.3]))

        self.act = nn.Tanh()
        self.proj = nn.Linear(feat_num, 1)

        self.weight = nn.Linear(feat_num, feat_num)

    def forward(self, attn, gfe, td):
        b, t, d = attn.shape
        td = td.type(attn.dtype)
        td[td > 2] = 2
        gfe = gfe.type(attn.dtype)
        gfe = einops.repeat(gfe, 'd -> b t d', b=b, t=t)
        ## two variables
        divide_term = self.decay_term * torch.log(math.e + (1 - attn) * td)
        dc = self.act(attn / divide_term)
        mask = (td == 2).float()
        updated_values = self.global_missing_param * gfe
        dc = dc * (1 - mask) + updated_values * mask
        score = torch.sigmoid(self.proj(dc))
        return dc, score

class FeatureAttention(nn.Module):
    def __init__(self, num_features, channel, calib=False):
        super().__init__()
        self.query = torch.nn.Linear(channel, channel)
        self.key = torch.nn.Linear(channel, channel)
        self.value = torch.nn.Linear(channel, channel)

        self.calib = calib
        if self.calib:
            self.confidence_param = nn.Parameter(torch.Tensor([0.05]))
            self.confidence_learner = ConfidenceLearner(feat_num=num_features)
        
    def forward(self, x, gfe, td):
        dc=None
        dc_score = None
        batch_size, time_steps, num_features, channel = x.size()

        # Shape: [batch_size, time_steps * num_features, channel]
        Q = self.query(x[:, -1, :, :])
        x = x.view(x.size(0), -1, x.size(3))

        K = self.key(x)
        V = self.value(x)

        attention_weights = F.softmax(Q.bmm(K.transpose(1, 2)) / (K.size(-1) ** 0.5), dim=2) # [bs, time_steps * num_features, time_steps * num_features]
        attention_weights = attention_weights.view(batch_size, num_features, time_steps, num_features)

        if self.calib:
            # calibrate the attention weights
            dc, dc_score = self.confidence_learner(attention_weights.mean(dim=[1]), gfe, td)
            # repeat dc to match the shape of attention_output
            dc = einops.repeat(dc, 'b t f -> b ff t f', ff=num_features)

            calib_attention_weights = self.confidence_param*dc + (1-self.confidence_param)*attention_weights
            dc = dc.mean(dim=[2]) # [bs, time_steps, num_features]

        calib_attention_weights = calib_attention_weights.view(batch_size, num_features, time_steps * num_features).transpose(1, 2)
        calib_attention_weights = einops.repeat(calib_attention_weights, 'b tf f -> b tf (repeat f)', repeat=time_steps)
        attention_output = calib_attention_weights.bmm(V)

        # reverse the shape transformation
        attention_output = attention_output.view(batch_size, time_steps, num_features, channel)
        
        # Current attention weights shape: [batch_size, time_steps * num_features, time_steps * num_features]
        # get the feature level attention weights for each sample
        calib_attention_weights = calib_attention_weights.view(batch_size, time_steps, num_features, time_steps, num_features)
        calib_attention_weights = calib_attention_weights.mean(dim=[3]) # [bs, time_steps, num_features, num_features]

        return attention_output, calib_attention_weights, attention_weights, dc, dc_score


class DemoProj(nn.Module):
    def __init__(self, demo_dim: int, feat_dim: int, **kwargs):
        super().__init__()
        self.feat_dim = feat_dim
        self.demo_projs = nn.ModuleList(
            [
                nn.Linear(1, feat_dim)
                for _ in range(demo_dim)
            ]
        )

    def forward(self, x):
        bs, demo_dim = x.shape
        out = torch.zeros(bs, demo_dim, self.feat_dim).to(x.device)
        for i, proj in enumerate(self.demo_projs):
            out[:, i] = proj(x[:, i:i+1])
        return out

class MCGRU(nn.Module):
    """
    input: x -> [bs, ts, lab_dim]
    output: [bs, ts, n_feature, feat_dim]
    """
    def __init__(self, feat_dim: int=8, **kwargs):
        super().__init__()
        self.num_features = 17 # 12 lab test + 5 categorical features
        self.dim_list = [2,8,12,13,12,1,1,1,1,1,1,1,1,1,1,1,1]
        self.feat_dim = feat_dim
        self.grus = nn.ModuleList(
            [
                nn.GRU(dim, feat_dim, num_layers=1, batch_first=True)
                for dim in self.dim_list
            ]
        )
    def forward(self, x):
        # for each feature, apply gru
        bs, ts, lab_dim = x.shape
        out = torch.zeros(bs, ts, self.num_features, self.feat_dim).to(x.device)
        # each feature's dim is different, as in the dim_list, so we need to iterate over it
        # the dim is the channel dim of each feature
        for i, gru in enumerate(self.grus):
            start_pos = sum(self.dim_list[:i])
            end_pos = sum(self.dim_list[:i+1])
            # print(start_pos, end_pos)
            cur_feat = x[:, :, start_pos:end_pos]
            # print(cur_feat.shape)
            cur_feat = gru(cur_feat)[0]
            out[:, :, i] = cur_feat
        return out

def get_each_feat_missing(gfe, td, dim_list=[1,1,2,8,12,13,12,1,1,1,1,1,1,1,1,1,1,1,1], num_feat=19):
    """
    gfe: [=input_dim,]
    td: [bs, ts, input_dim]
    """
    bs, ts, input_dim = td.shape
    _gfe = torch.zeros(num_feat).to(gfe.device)
    _td = torch.zeros(bs, ts, num_feat).to(td.device)
    for i, dim in enumerate(dim_list):
        start_pos = sum(dim_list[:i])
        # end_pos = sum(dim_list[:i+1])
        _gfe[i] = gfe[start_pos]
        _td[:, :, i] = td[:, :, start_pos]
    return _gfe, _td


class ATCare(nn.Module):
    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        demo_dim: int,
        lab_dim: int,
        centers: torch.Tensor,
        num_lab_feat: int=17,
        num_feat: int=19,
        feat_dim: int=8,
        act_layer=nn.GELU,
        calib: bool=True,
        **kwargs
    ):
        super().__init__()

        assert num_lab_feat+demo_dim == num_feat, "num_lab_feat+demo_dim should equal to num_feat"
        self.demo_proj = DemoProj(demo_dim, feat_dim)
        self.mcgru = MCGRU(feat_dim)
        
        # importance score learning
        self.attention = FeatureAttention(num_feat, feat_dim, calib=calib)
        
        # group patients learning
        self.group_patient_learner = GroupPatientLearnerModule(num_feat, centers, hidden_dim)
        self.group_embed_param = nn.Parameter(torch.Tensor([0.05]))

        # final projection
        self.proj1 = nn.Linear(num_feat*feat_dim, num_feat)
        self.proj2 = nn.Linear(num_feat, hidden_dim)

        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True)


    def forward(self,
        x: torch.Tensor,                # x -> [bs, ts, lab_dim]
        demo: torch.Tensor,             # demo -> [bs, demo_dim]
        gfe: torch.Tensor,              # gfe -> [input_dim]
        td: torch.Tensor,               # td -> [bs, ts, input_dim]
    ):
        b,t,d = x.shape
        gfe = gfe[0]
        demo = self.demo_proj(demo) # [bs, demo_dim, feat_dim]
        lab = self.mcgru(x) # [bs, ts, num_lab_feat, feat_dim]
        # repeat demo to match lab
        demo = einops.repeat(demo, 'b d f -> b t d f', t=x.shape[1])
        # concat lab and demo
        x = torch.cat([demo, lab], dim=2) # [bs, ts, num_lab_feat+demo_dim, feat_dim]
        # get each feature from gfe and td
        gfe, td = get_each_feat_missing(gfe, td)
        # learn feature importance, data confidence
        context, calib_attn, attn, dc, dc_score = self.attention(x, gfe, td) # [bs, ts, num_lab_feat+demo_dim, feat_dim]

        shortcut = x
        x = context
        x = shortcut + x

        # group similar patients
        last_visit, last_visit_dc = x[:, -1, :, :], dc[:, -1, :] # [bs, num_feat, feat_dim], [bs, num_feat]
        group_scores, group_embedding, group_patients_embedding, group_label = self.group_patient_learner(last_visit, last_visit_dc) # [bs, num_feat], [bs, num_feat, hidden_dim], [bs, num_feat, hidden_dim]

        x = x.flatten(2) # [bs, ts, num_features*feat_dim]
        x = self.proj1(x) # [bs, ts, num_feat]

        x = self.group_embed_param * group_patients_embedding.unsqueeze(1) + (1 - self.group_embed_param) * x

        x = self.proj2(x) # [bs, ts, hidden_dim]
        _, out = self.gru(x)
        out = out.mean(dim=0)
        return out, calib_attn, attn, group_label
