from torch import nn
from gradient_reversal import GradientReversal
from torch.nn import functional as F
import torch

class DANN(nn.Module):
  def __init__(self):
    super().__init__()  # Important, otherwise will throw an error

    self.cnn = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5),
    nn.ReLU(),

    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5),
    nn.ReLU(),

    nn.MaxPool2d(kernel_size=2, stride=2),
  )

    self.classif = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=48 * 4 * 4, out_features=100),

    nn.ReLU(),
    nn.Linear(in_features=100, out_features=100),

    nn.ReLU(),
    nn.Linear(in_features=100, out_features=10),
    # no softmax because we use cross entropy loss

  )

    self.domain = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=48 * 4 * 4, out_features=100),

    nn.ReLU(),
    nn.Linear(in_features=100, out_features=1)
    # no softmax because we use binary cross entropy loss

  )

  def forward(self, x, factor=-1):
    #TODO
    self.features = self.cnn(x)
    # x.shape -> 128
    # 64 -> Gray 64 -> RGB
    class_pred = self.classif(self.features)
    features_r = GradientReversal.apply(self.features, factor)
    domain_pred = self.domain(features_r)
    
    # domain_pred_s, domain_pred_t = domain_pred.chunk(2, dim=0)
    # domain_pred_tt = GradientReversal.apply(domain_pred_t, factor)
    # domain_pred_all = torch.cat([domain_pred_s, domain_pred_tt], dim=0)

    return class_pred, domain_pred
  
def eval_dann(net, loader, source=True):
  net.eval()

  c_acc, d_acc, cls_loss, d_loss = 0, 0, 0., 0.
  c = 0
  for x, y in loader:
    x = x.cuda()
    if source:
      d = torch.ones(len(x))
    else:
      d = torch.zeros(len(x))

    c += len(x)

    with torch.no_grad():
      cls_logits, domain_logits = net(x.cuda())
      cls_logits, domain_logits = cls_logits.cpu(), domain_logits.cpu()

    cls_loss += F.cross_entropy(cls_logits, y).item()
    d_loss += F.binary_cross_entropy_with_logits(domain_logits[:, 0], d).item()

    c_acc += (cls_logits.argmax(dim=1) == y).sum().item()
    d_acc += ((torch.sigmoid(domain_logits[:, 0]) > 0.5).float() == d).sum().item()

  return round(100 * c_acc / c, 2), round(100 * d_acc / c, 2), round(cls_loss / len(loader), 5), round(d_loss / len(loader), 5)