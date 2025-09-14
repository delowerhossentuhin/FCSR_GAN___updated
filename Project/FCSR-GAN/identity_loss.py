
import torch
import torch.nn as nn
import numpy as np
from insightface.app import FaceAnalysis

class IdentityPreservingLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.app = FaceAnalysis(providers=['CPUExecutionProvider'])  # use CPU if CUDA not available
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.criterion = nn.CosineSimilarity(dim=1)

    def forward(self, recon_img, gt_img):
        recon_emb = []
        gt_emb = []

        for img in recon_img:
            img_np = (img.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            faces = self.app.get(img_np)
            embedding = faces[0].embedding if len(faces) > 0 else torch.zeros(512).to(self.device)
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            recon_emb.append(embedding)

        for img in gt_img:
            img_np = (img.detach().cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            faces = self.app.get(img_np)
            embedding = faces[0].embedding if len(faces) > 0 else torch.zeros(512).to(self.device)
            if not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32).to(self.device)
            gt_emb.append(embedding)

        recon_emb = torch.stack(recon_emb)
        gt_emb = torch.stack(gt_emb)

        return 1 - self.criterion(recon_emb, gt_emb).mean()

