from models.hyptorch.nn import *
from models.hyptorch.pmath import mobius_matvec
import torch.nn as nn
import torch.nn.functional as F

class PHypCoAttn(torch.nn.Module):
    def __init__(self, dim, c):
        super(PHypCoAttn, self).__init__()
        self.query = HypLinear(dim, dim, c)
        self.key = HypLinear(dim, dim, c)
        self.value = HypLinear(dim, dim, c)
        
        self.softmax = HyperbolicMLR(ball_dim=dim, n_classes=dim, c=c)
        
        self.c = c
        

    def forward(self, q, k, v):
        # Compute query, key and value projections
        q = self.query(q)  # Query from first input
        k = self.key(k)    # Key from second input
        v = self.value(v)  # Value from second input
        
        # Compute attention scores (similarity in hyperbolic space)
        attn_scores = torch.bmm(q, k.transpose(-1, -2)) / (256 ** 0.5)
        
        # Apply softmax to get attention weights
        attn_weights = nn.Softmax(dim=-1)(attn_scores)

        # Compute the weighted sum of values
        attn_output = torch.bmm(attn_weights, v)

        # Project back to the Poincare disk (if necessary)

        return attn_output

class PHSNNBlock(nn.Module):
    def __init__(
        self,
        c,
        in_dim,
        dim=[256, 256],
        dropout=0.25,
    ):
        super().__init__()
        self.c = c
        self.tohyp = ToPoincare(c=self.c, ball_dim = in_dim)
        self.toeuc = FromPoincare(c=self.c, ball_dim = in_dim)

        self.backbone1 = HypLinear(in_dim, dim[0], self.c)
        self.backbone2 = HypLinear(dim[0], dim[1], self.c)
        
        self.elu1 = nn.ELU()
        self.elu2 = nn.ELU()

        self.dropout1 = nn.AlphaDropout(p=dropout, inplace=False)
        self.dropout2 = nn.AlphaDropout(p=dropout, inplace=False)
                                    


    def forward(self, x):
        # print(x)
        x = self.tohyp(x)
        x = self.backbone1(x)
        # x = self.manifold.activation(x, nn.ELU())
        x = self.elu1(x)
        x = self.dropout1(x)
        
        x = self.backbone2(x)
        x = self.elu2(x)
        x = self.dropout2(x)
        
        x = self.toeuc(x)
        
        return x
    

class PHypFusion(nn.Module):
    def __init__(self, dim, c):
        super().__init__()
        self.co_attn_p2g = PHypCoAttn(dim=dim, c=c)
        self.co_attn_g2p = PHypCoAttn(dim=dim, c=c)
        
        self.LN1 = nn.LayerNorm(dim)
        self.LN2 = nn.LayerNorm(dim)
        
        self.p_to_h = ToPoincare(c=c,ball_dim=dim,riemannian=False,clip_r=None)
        self.g_to_h = ToPoincare(c=c,ball_dim=dim,riemannian=False,clip_r=None)
        self.toeuc = FromPoincare(c=c, ball_dim=512)
        
    def forward(self, x): 
        g = x[:,:6,:]
        p = x[:,6:,:]
        
        # normalization 
        g = F.normalize(self.g_to_h(g), dim=-1)
        p = F.normalize(self.p_to_h(p), dim=-1)
        
        g_x = g + self.co_attn_p2g(v=p, k=p, q=g)
        g_x = self.LN1(g_x)
        
        p_x = p + self.co_attn_g2p(v=g, k=g, q=p)
        p_x = self.LN2(p_x)
        
        x = torch.cat((g_x, p_x), dim=1)
        
        x = self.toeuc(x)
        # print(torch.isnan(x).sum().item())
        
        return x

class PMLP(nn.Module):
    def __init__(self, c, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.c = c
        self.tohyp = ToPoincare(c=self.c, ball_dim = in_dim)
        self.toeuc = FromPoincare(c=self.c, ball_dim = out_dim)

        self.backbone1 = nn.Sequential(*[HypLinear(in_dim, hidden_dim, self.c),
                                        nn.ReLU(),
                                        nn.Dropout(0.25)
                                         ])
        self.backbone2 = nn.Sequential(*[HypLinear(hidden_dim, out_dim, self.c),
                                        nn.ReLU(),
                                        nn.Dropout(0.25)
                                         ])
        
    def forward(self, x): 
        x = self.tohyp(x)
        x = self.backbone1(x)
        x = self.backbone2(x)
        x = self.toeuc(x)
        return x