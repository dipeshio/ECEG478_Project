import torch.nn as nn
import torchvision
import torchvision.models as models
import torch as pt

# print("PyTorch:", pt.__version__)
# print("Built with CUDA:", pt.version.cuda)
# print("CUDA available:", pt.cuda.is_available())

# check if GPU is available
if pt.cuda.is_available():
    device = pt.device("cuda")
    # print(device)

class AgeGenderNet(nn.Module):
    def __init__(self, n_age=8, n_gen=2, n_race=7):
        super().__init__()
        backbone = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(*(list(backbone.children())[:-1]))
        self.dropout  = nn.Dropout(p=0.3)
        self.age_head = nn.Linear(512, n_age)
        self.gen_head = nn.Linear(512, n_gen)
        self.race_head= nn.Linear(512, n_race)


    def forward(self, x):
        x = self.dropout(self.features(x).flatten(1))   # ← Dropout active only in .train()
        ## using the resNet backbone
        return {"age": self.age_head(x),
                "gender": self.gen_head(x),
                "race": self.race_head(x)}

model = AgeGenderNet().cuda()
loss_age    = nn.CrossEntropyLoss()
loss_gender = nn.CrossEntropyLoss()     # fairly balanced already
loss_race = nn.CrossEntropyLoss()     # fairly balanced already


def criterion(preds, targets, smooth=0.1):
    ce = nn.CrossEntropyLoss(label_smoothing=smooth)
    loss =  ce(preds["age"], targets["age"]) \
          + ce(preds["gender"], targets["gender"]) \
          + ce(preds["race"], targets["race"])
    return loss, None
    
    
## optimizer
optimizer = pt.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)