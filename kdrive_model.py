import torch
import models
from timm.models import create_model
from models.modules.mobileone import reparameterize_model


class KDriveModel(torch.nn.Module):
    # extra inputs :
    # past trajectory (x,y)*2 = 20
    # velocity (x,y) = 2
    # rotation (w,x,y,z) = 4
    # acceleration (x,y)=2
    # steering =1
    # throttle =1
    # brake =1
    # input_dim=31
    # output= (x,y)*10 timestamp=20
    def __init__(self, extra_in_feat=31, out_feat=20, bmodel_name="fastvit_t8", bmodel_weight="weights/fastvit_t8.pth.tar"):
        super().__init__()
        self._is_eval = False
        self.base_model = create_model(bmodel_name)
        checkpoint = torch.load(bmodel_weight)
        self.base_model.load_state_dict(checkpoint['state_dict'])
        bmodel_out_feat = self.base_model.head.in_features
        self.base_model.head = torch.nn.Identity()
        self.head = torch.nn.Linear(bmodel_out_feat+extra_in_feat, out_feat)

    def __eval__(self):
        self.base_model.eval()
        self._is_eval = True

    def forward(self, x, extra_inp):
        x = self.base_model(x)
        x = torch.cat([x, extra_inp], dim=1)
        return self.head(x)

    def freeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

    def unfreeze_base_model(self):
        for param in self.base_model.parameters():
            param.requires_grad = True

    def reparametrize(self):
        self.base_model = reparameterize_model(self.base_model)
