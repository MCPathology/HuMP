import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.lhyperbolic import HypLinear
from models.manifolds.lorentz import * 
from models.layers.lhyperbolic import LHypFusion

class Gating(nn.Module):
    def __init__(self, expert_num):
        super(Gating, self).__init__()
        self.expert_num = expert_num
        self.classifier = nn.Linear(expert_num*256*2, expert_num)  

    def forward(self, tokens, g_num, p_num, modality_gate=True):
        # Concatenate tensors along the last dimension
        combined_tensor = torch.cat(tokens, dim=-1)
        if modality_gate:
            gene_part = combined_tensor[:, :g_num, ...].mean(dim=1)  # Take first 6 elements from second dimension
            pathology_part = combined_tensor[:, g_num:g_num+p_num, ...].mean(dim=1)  # Take elements from 6 to end from second dimension
            # table_part = combined_tensor[:, g_num+p_num:, ...].mean(dim=1)
            
            output = self.classifier(torch.cat((gene_part, pathology_part),dim=-1))
        else:
            gene_part = combined_tensor.mean(dim=1)
            output = self.classifier(gene_part)

        weights = F.softmax(output, dim=-1).squeeze(dim=0)
        
        return weights

class KGGenerator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, factor_num=2, l=2):
        super(KGGenerator, self).__init__()
                
        self.l=l
        self.divide=1.0
        self.hidden_dim = hidden_dim
        self.factor_num = factor_num

        self.proto_linear = nn.Linear(input_dim,hidden_dim*factor_num)
        self.all_linear = nn.Linear(input_dim,hidden_dim*factor_num)

        self.fclayer=nn.Linear(hidden_dim,output_dim)
        #self.fclayer2=nn.Linear(hidden_dim,output_dim)
        self.predictor = nn.Linear(output_dim,1)

        nn.init.xavier_normal(self.proto_linear.weight)
        nn.init.xavier_normal(self.all_linear.weight)


    def forward(self, x, n1, n2):
        
        part_g = x[:, :n1, :]
        part_p = x[:, n1:n1+n2, :]
        # print(part_p)
        proto_g = torch.mean(part_g, dim=1)
        proto_p = torch.mean(part_p, dim=1)
        
        c1 = torch.mul(part_g, proto_g.unsqueeze(1))  # [1,1,256] * [1,n1,256] → [1,n1,256]
        c2 = torch.mul(part_p, proto_p.unsqueeze(1))  # [1,n2,256]
    
        # print(c1.shape)
        c = torch.cat([c1, c2], dim=1)  # [1, n1+n2+n3, 256]
        c = self.proto_linear(c)
        x = self.all_linear(x)

        c = c.view(-1, self.hidden_dim, self.factor_num) # [n1+n2+n3, hidden, factor]
        c = torch.squeeze(torch.sum(c, 2)) # [ni+n2+n3, hidden]
        c = F.relu(c)
        c = self.fclayer(c) # [n1+n2+n3, 256]
        c = F.relu(c)
        
        predict = torch.mean(c, dim=0).unsqueeze(0) # [1,256
        predict = self.predictor(predict) # [1, 1]
        
        kappa=torch.randn(predict.shape).cuda()
        kappa=torch.tanh(predict*self.l)/self.divide
        # print(kappa)
        return kappa.contiguous().view(-1)
        
class KPGenerator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, factor_num=1, l=2):
        super(KPGenerator, self).__init__()
                
        self.l=l
        self.divide=1.0
        self.hidden_dim = hidden_dim
        self.factor_num = factor_num

        self.proto_linear = nn.Linear(input_dim,hidden_dim*factor_num)
        self.all_linear = nn.Linear(input_dim,hidden_dim*factor_num)

        self.fclayer=nn.Linear(hidden_dim,output_dim)
        #self.fclayer2=nn.Linear(hidden_dim,output_dim)
        self.predictor = nn.Linear(output_dim,1)

        nn.init.xavier_normal(self.proto_linear.weight)
        nn.init.xavier_normal(self.all_linear.weight)


    def forward(self, x):
        
        proto = torch.mean(x, dim=1)
        x = self.all_linear(x)
        c = torch.mul(x, proto.unsqueeze(1))  # [1,1,256] * [1,n1,256] → [1,n1,256]
    
        # print(c1.shape)
        c = self.proto_linear(c)

        c = c.view(-1, self.hidden_dim, self.factor_num) # [n1+n2+n3, hidden, factor]
        c = torch.squeeze(torch.sum(c, 2)) # [n1, hidden]
        c = F.relu(c)
        c = self.fclayer(c) # [n1, 256]
        c = F.relu(c)
        
        predict = torch.mean(c, dim=0).unsqueeze(0) # [1,256
        predict = self.predictor(predict) # [1, 1]
        
        kappa=torch.randn(predict.shape).cuda()
        kappa=torch.sigmoid(predict*self.l)/self.divide
        # print(kappa)
        return kappa.contiguous().view(-1)  
          
class Experts(nn.Module):
    def __init__(self, kappa,in_dim, dim=256):
        super(Experts, self).__init__()
        self.manifolds = Lorentz(k = kappa)  # Store manifold information (e.g., curvature)

        # Define MLP layers
        self.mlp_pathology = nn.Sequential(
            HypLinear(self.manifolds, in_dim, dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            HypLinear(self.manifolds, dim, dim),
            nn.ReLU(),
            nn.Dropout(0.25)
            
        )
        
        self.mlp_gene = nn.Sequential(
            HypLinear(self.manifolds, 4*dim, dim),
            nn.ELU(),
            nn.Dropout(0.25),
            HypLinear(self.manifolds, dim, dim),
            nn.ELU(),
            nn.Dropout(0.25)
        )

        self.mlp_table = nn.Sequential(
            HypLinear(self.manifolds, 2*dim, dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            HypLinear(self.manifolds, dim, dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        # Cross attention layer
        self.fusion = LHypFusion(self.manifold, dim=256)

    def forward(self, pathology, gene, table):
        pathology_hyper = self.hyperbolic_embedding(pathology)
        gene_hyper = self.hyperbolic_embedding(gene)
        table_hyper = self.hyperbolic_embedding(table)

        pathology_mlp = self.mlp_pathology(pathology_hyper)
        gene_mlp = self.mlp_gene(gene_hyper)
        table_mlp = self.mlp_table(table_hyper)
        
        # Apply cross attention
        attn_output = []# self.fusion(combined_features, combined_features, combined_features)

        return attn_output

    def hyperbolic_embedding(self, x):
        x = torch.cat([torch.ones_like(x)[..., 0:1], x], dim=-1)
        x = self.manifold.expmap0(x)  
        return x
