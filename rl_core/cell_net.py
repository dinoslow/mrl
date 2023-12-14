import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class CellNet(nn.Module):
    def __init__(self, cell_dim=256, emb_dim=256, num_heads=16):
        super(CellNet, self).__init__()
        self.query_var = torch.nn.Linear(1, emb_dim, bias=False)
        self.proj_cell = torch.nn.Conv1d(cell_dim, emb_dim, 1)
        self.proj_code = torch.nn.Conv1d(12, emb_dim, 1)
        self.multihead_attn = torch.nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads)

    def forward(self, cell, code):
        # Cell Project
        cell_emb = self.proj_cell(cell.permute(0,2,1))
        cell_emb = F.relu(cell_emb).permute(2,0,1)
        # Code Project
        code_emb = self.proj_code(code.permute(0,2,1))
        code_emb = F.relu(code_emb).permute(2,0,1)
        query = self.query_var(torch.ones(cell.shape[0],1).to(cell.get_device())).unsqueeze(0)
        cell_fusion, _ = self.multihead_attn(query, code_emb, cell_emb)
        cell_fusion = cell_fusion.view(cell.shape[0], cell_fusion.shape[-1])
        return cell_fusion


