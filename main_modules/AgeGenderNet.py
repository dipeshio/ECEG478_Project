import torch.nn as nn
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
    def __init__(self, n_age=9, n_gender=2, n_race=7, backbone="resnet18"):
        super().__init__()
        self.backbone = getattr(models, backbone)(weights="IMAGENET1K_V1")
        dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()          # remove final FC

        self.age_head    = nn.Linear(dim, n_age)      # 9 classes
        self.gender_head = nn.Linear(dim, n_gender)   # 2 classes
        self.race_head = nn.Linear(dim, n_race)   # 7 classes


    def forward(self, x):
        feat = self.backbone(x)
        ## using the resNet backbone
        return {
            "age":    self.age_head(feat),
            "gender": self.gender_head(feat),
            "race": self.race_head(feat)

        }

model = AgeGenderNet().cuda()
loss_age    = nn.CrossEntropyLoss()
loss_gender = nn.CrossEntropyLoss()     # fairly balanced already
loss_race = nn.CrossEntropyLoss()     # fairly balanced already


def criterion(pred_dict, target_dict, λ_age=1.0, λ_gender=1.0, λ_race=1.0):
    L_age    = loss_age(pred_dict["age"],    target_dict["age"])
    L_gender = loss_gender(pred_dict["gender"], target_dict["gender"])
    L_race = loss_race(pred_dict["race"], target_dict["race"])

    
    return λ_age*L_age + λ_gender*L_gender + λ_race*L_race , {"age": L_age.item(),
                                            "gender": L_gender.item(), "race": L_race.item()}
    
    
## optimizer
optimizer = pt.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = pt.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)