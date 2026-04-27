"""
HuMP: Hyperbolic Unified Multimodal Pathology.

Main model definition. Jointly fuses molecular (genomics, transcriptomics,
proteomics), pathology (WSI patch features), and clinical text features in
a Lorentz hyperbolic manifold under a directional entailment hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.cross_attention import FeedForward
from models.layers.lhyperbolic import (
    SAFusion,
    HypABMIL,
    prototype_kmeans,
    hyperbolic_entailment_completion_strict,
    hyperbolic_entailment_loss_pairwise
)
from models.manifolds.lorentz import Lorentz
from .util import initialize_weights, SNN_Block


class HuMP(nn.Module):
    """Hyperbolic Unified Multimodal Pathology.

    Args:
        genomic_sizes:   list of input dims for the six genomic groups.
        transomic_sizes: list of input dims for the transcriptomic groups.
        n_classes:       number of survival bins (or classification classes).
        fusion:          legacy flag, kept for backward compatibility.
        model_size:      ``"small"`` or ``"large"`` hidden-dim preset.
    """

    def __init__(
        self,
        genomic_sizes=(100, 200, 300, 400, 500, 600),
        transomic_sizes=(),
        n_classes=4,
        fusion="concat",
        model_size="small",
    ):
        super().__init__()
        self.genomic_sizes = list(genomic_sizes)
        self.transomic_sizes = list(transomic_sizes)
        self.n_classes = n_classes
        self.fusion = fusion
        self.num_pathways = len(self.transomic_sizes)

        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics":  {"small": [1024, 256, 256], "large": [1024, 1024, 1024, 256]},
            "protein":   {"small": [1280, 256, 256], "large": [1024, 1024, 1024, 256]},
        }

        # ---------- Pathology branch ----------
        hidden = self.size_dict["pathomics"][model_size]
        p_hidden = self.size_dict["protein"][model_size]

        ffpe_layers = []
        for idx in range(len(hidden) - 1):
            ffpe_layers.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            ffpe_layers.append(nn.ReLU6())
            ffpe_layers.append(nn.Dropout(0.25))
        self.ffpe_fc = nn.Sequential(*ffpe_layers)

        # ---------- Lorentz manifold ----------
        self.k = 1.0
        self.manifold = Lorentz(k=self.k)

        # ---------- Molecular branch ----------
        self._init_per_gene_model(self.genomic_sizes)
        self._init_per_trans_model(self.transomic_sizes)
        self._init_per_protein_model(p_hidden)

        self.gene_agg = self._make_omic_aggregator()
        self.trans_agg = self._make_omic_aggregator()

        # ---------- Clinical branch ----------
        self.clinic_fc = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 256),
        )

        # ---------- Multimodal fusion ----------
        self.attention_fusion = SAFusion(manifold=self.manifold, dim=256)
        self.feed_forward = FeedForward(256, dropout=0.25)
        self.layer_norm = nn.LayerNorm(256)
        self.p_norm = nn.LayerNorm(256)
        self.mil = HypABMIL(Lorentz(k=self.k), 256, 256)

        # ---------- Heads ----------
        self.mm = nn.Sequential(
            nn.Linear(hidden[-1] * 3, hidden[-1]),
            nn.ReLU6(),
        )
        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    # Branch initializers
    # ------------------------------------------------------------------
    def _make_omic_aggregator(self):
        return nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 256),
        )

    def _init_per_gene_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i in range(len(hidden) - 1):
                fc.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc))
        self.gene_sig_networks = nn.ModuleList(sig_networks)

    def _init_per_trans_model(self, omic_sizes):
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i in range(len(hidden) - 1):
                fc.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc))
        self.trans_sig_networks = nn.ModuleList(sig_networks)

    def _init_per_protein_model(self, p_hidden):
        layers = []
        for idx in range(2):
            layers.append(nn.Linear(p_hidden[idx], p_hidden[idx + 1]))
            layers.append(nn.ReLU6())
            layers.append(nn.Dropout(0.25))
        self.protein_networks = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, **kwargs):
        # ---- Missing-modality branches: HGS completion via hyperbolic entailment cones
        if kwargs["x_path"] is None:
            # Pathology missing: complete P from G and C using entailment between them.
            genomics_features, g_loss, t_loss = self._process_multiomics(kwargs)
            x_table = kwargs["report"]
            clinic_features = self.clinic_fc(x_table).unsqueeze(0)
            # Use a genomics prototype to set the target sequence length for P.
            p_proto_shape = prototype_kmeans(genomics_features, 128)
            embed_dict = {
                "P": p_proto_shape,
                "G": genomics_features,
                "C": clinic_features,
            }
            pathology_features = hyperbolic_entailment_completion_strict(
                embed_dict, "P", self.manifold
            )
            genomics_proto = pathology_proto = clinic_proto = None
            g2p_loss = p2c_loss = torch.tensor(0.0, device="cuda")

        elif kwargs["x_genomic1"] is None:
            # Genomics missing: complete G from P and C using entailment intersection.
            x_path = kwargs["x_path"]
            pathology_features = self.ffpe_fc(x_path).to("cuda")
            x_table = kwargs["report"]
            clinic_features = self.clinic_fc(x_table).unsqueeze(0)
            # Use a pathology prototype to set the target sequence length for G.
            g_proto_shape = prototype_kmeans(pathology_features, 32)
            embed_dict = {
                "G": g_proto_shape,
                "P": pathology_features,
                "C": clinic_features,
            }
            genomics_features = hyperbolic_entailment_completion_strict(
                embed_dict, "G", self.manifold
            )
            g_loss = t_loss = torch.tensor(0.0, device="cuda")
            genomics_proto = pathology_proto = clinic_proto = None
            g2p_loss = p2c_loss = torch.tensor(0.0, device="cuda")

        elif kwargs["report"] is None:
            # Clinical missing: complete C from G and P using outer entailment cone.
            genomics_features, g_loss, t_loss = self._process_multiomics(kwargs)
            x_path = kwargs["x_path"]
            pathology_features = self.ffpe_fc(x_path).to("cuda")
            # Use a pathology prototype to set the target sequence length for C.
            c_proto_shape = prototype_kmeans(pathology_features, 10)
            embed_dict = {
                "C": c_proto_shape,
                "G": genomics_features,
                "P": pathology_features,
            }
            clinic_features = hyperbolic_entailment_completion_strict(
                embed_dict, "C", self.manifold
            )
            genomics_proto = pathology_proto = clinic_proto = None
            g2p_loss = p2c_loss = torch.tensor(0.0, device="cuda")

        else:
            # ---- Full three-modality branch ----
            x_path = kwargs["x_path"]
            x_genomic = [kwargs["x_genomic%d" % i].to("cuda") for i in range(1, 7)]
            x_transomic = [kwargs["x_transomic%d" % i].to("cuda")
                           for i in range(1, self.num_pathways + 1)]
            x_table = kwargs["report"]

            if kwargs["protein"] is not None:
                is_protein = True
                protein = kwargs["protein"].to("cuda")
                protein_features = self.protein_networks(protein)
            else:
                is_protein = False
                protein_features = None

            pathology_features = self.ffpe_fc(x_path).to("cuda")
            clinic_features = self.clinic_fc(x_table).unsqueeze(0)

            genomic = [self.gene_sig_networks[idx].forward(sig.float())
                       for idx, sig in enumerate(x_genomic)]
            transomic = [self.trans_sig_networks[idx].forward(sig.float())
                         for idx, sig in enumerate(x_transomic)]
            genomics = self.gene_agg(torch.stack(genomic)).unsqueeze(0)
            transomics = self.trans_agg(torch.stack(transomic)).unsqueeze(0)

            g_loss = hyperbolic_entailment_loss_pairwise(
                genomics, transomics, self.manifold, self.k)
            if is_protein:
                t_loss = hyperbolic_entailment_loss_pairwise(
                    transomics, protein_features, self.manifold, self.k)
                genomics = torch.cat((genomics, transomics, protein_features), dim=-2)
            else:
                t_loss = torch.tensor(0.0, device="cuda")
                genomics = torch.cat((genomics, transomics), dim=-2)
            genomics_features = genomics

            g2p_loss = hyperbolic_entailment_loss_pairwise(
                pathology_features, genomics_features, self.manifold, self.k)
            p2c_loss = hyperbolic_entailment_loss_pairwise(
                clinic_features, pathology_features, self.manifold, self.k)

            pathology_proto = prototype_kmeans(pathology_features, 128)
            genomics_proto = prototype_kmeans(genomics_features, 32)
            clinic_proto = prototype_kmeans(clinic_features, 10)

        # ---- Multimodal fusion ----
        p_total = pathology_features.shape[1]
        g_total = genomics_features.shape[1]
        token_cross = self.attention_fusion(
            genomics_features, pathology_features, clinic_features)
        token_cross = self.feed_forward(token_cross)
        token_cross = self.layer_norm(token_cross)

        gene_embed = token_cross[:, :g_total, :]
        path_embed = token_cross[:, g_total:g_total + p_total, :]
        clinic_embed = token_cross[:, p_total + g_total:, :]

        gene_embed = torch.mean(gene_embed, dim=1)
        path_embed = torch.mean(path_embed, dim=1)
        clinic_embed = torch.mean(clinic_embed, dim=1)
        token_mm = torch.cat((gene_embed, path_embed, clinic_embed), dim=-1)
        fusion = self.mm(token_mm)
        logits = self.classifier(fusion)

        return (
            logits,
            (g_loss + t_loss) + (p2c_loss + g2p_loss),
            {"G": genomics_proto, "P": pathology_proto, "C": clinic_proto},
        )

    # ------------------------------------------------------------------
    # Multi-omics processing for the missing-modality branches
    # ------------------------------------------------------------------
    def _process_multiomics(self, kwargs):
        """Aggregate genomics / transcriptomics / proteomics for missing-mode forward.

        Returns:
            genomics_features (Tensor): aggregated molecular features [1, N, D].
            g_loss (Tensor): genomics -> transcriptomics entailment loss
                (zero in this branch since the Eucliden form is not enforced).
            t_loss (Tensor): transcriptomics -> proteomics entailment loss.
        """
        x_genomic = [kwargs.get(f"x_genomic{i}", None) for i in range(1, 7)]
        x_genomic = [x.to("cuda") for x in x_genomic if x is not None]

        x_transomic = [kwargs.get(f"x_transomic{i}", None)
                       for i in range(1, self.num_pathways + 1)]
        x_transomic = [x.to("cuda") for x in x_transomic if x is not None]

        protein = kwargs.get("protein", None)
        is_protein = protein is not None

        genomic = [self.gene_sig_networks[idx].forward(sig.float())
                   for idx, sig in enumerate(x_genomic)]
        genomics = self.gene_agg(torch.stack(genomic)).unsqueeze(0)

        transomic = [self.trans_sig_networks[idx].forward(sig.float())
                     for idx, sig in enumerate(x_transomic)]
        transomics = self.trans_agg(torch.stack(transomic)).unsqueeze(0)

        if is_protein:
            protein_features = self.protein_networks(protein.to("cuda"))
        else:
            protein_features = None

        g_loss = t_loss = torch.tensor(0.0, device="cuda")

        genomics_features = torch.cat(
            [x for x in [genomics, transomics, protein_features] if x is not None],
            dim=-2,
        )
        return genomics_features, g_loss, t_loss
