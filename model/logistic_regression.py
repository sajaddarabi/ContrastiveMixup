import torch
import torch.nn as nn
from base import BaseModel
from .nn_utils import *
import torch.nn.functional as F

class LogisticRegression(BaseModel):
    def __init__(self,
                 input_dim,
                 num_classes,
                 cont_idxs=[],
                 cat_idxs=[],
                 cat_dims=[],
                 model=None,
                 fine_tune=False,
                 mixup_method='',
                 mixup_alpha=1.0,
                 embed=False,
                 **kwargs):
        super().__init__()

        if (len(cat_idxs) == 0) and  (len(cont_idxs) == 0):
            # assume all cont
            cont_idxs = list(range(input_dim))
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.mixup_method = mixup_method
        self.mixup_alpha = mixup_alpha
        self.model = model
        self.fine_tune = fine_tune
        if embed:
            self.embeddings = EmbeddingGenerator(input_dim, cat_dims, cat_idxs)
        else:
            self.embeddings = EmbeddingGenerator(input_dim, [], [])
        self.post_embed_dim = self.embeddings.post_embed_dim

        hidden_dim = [self.post_embed_dim]

        # Encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(self.embeddings)

        self.encoder.append(nn.Linear(hidden_dim[-1], self.num_classes))
        weight = kwargs.get('weight', None)
        if weight is not None:
            weight = torch.Tensor(weight)
        self.criterion_softmax = nn.CrossEntropyLoss(weight=weight)
        self.criterion_mixup = nn.BCELoss(weight=weight)

    def forward(self, data, device, **kwargs):
        # - input
        x, x_u, target, utarget = None, None, None, None
        if type(data) == torch.Tensor:
            x = data
            x = x.to(device)
            target = None
        elif len(data) == 3:
            x, x_u, target = data
        elif len(data) == 4:
            x, x_u, target, utarget = data
            x = torch.cat([x, x_u], dim=0)
            target = torch.cat([target, utarget], dim=0)
        else:
            x, target = data
            x, target = x.to(device), target.to(device)
        x = torch.flatten(x, start_dim=1)
        target_reweighted = None
        if self.model:
            if self.fine_tune:
                x = self.model.encode(x)
            else:
                with torch.no_grad():
                    x = self.model.encode(x)
        if target is not None and self.mixup_method != '':
            target_reweighted = to_one_hot(target, self.num_classes)

        if self.mixup_method == "mixup_hidden":
            layer = np.random.randint(0, len(self.encoder))
        elif self.mixup_method == "mixup":
            layer = 0
        else:
            layer = -1

        if self.mixup_alpha is not None:
            lam = get_lambda(self.mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).to(x.device)
            lam = Variable(lam)

        for i, l in enumerate(self.encoder):
            x = l(x)
            if i == layer and target is not None:
                 x, target_reweighted = mixup_process(x, target_reweighted, lam)

        out = {}
        out['preds'] = nn.functional.softmax(x)
        out['logits'] = x
        out['target_reweighted'] = target_reweighted
        return out


    def loss(self, outputs, data, device, **kwargs):
        # - inputs
        if len(data) == 3:
            x, x_u, target = data
            x, x_u, target = list(map(lambda a: a.to(device), data))
        elif len(data) == 4:
            x, x_u, target, utarget = data
            x, x_u, target, utarget = list(map(lambda a: a.to(device), data))
            x = torch.cat([x, x_u], dim=0)
            target = torch.cat([target, utarget], dim=0)
        else:
            x, target = data
            x, target = list(map(lambda a: a.to(device), data))

        logits = outputs.get('logits', None)
        target_reweighted = outputs.get('targets_reweighted')

        loss = {}
        if target_reweighted is not None:
            target = target_reweighted
            loss['mixup'] = self.criterion_mixup(logits, target)
        else:
            loss['nll'] = self.criterion_softmax(logits, target.long())
        loss['opt'] = loss['nll']
        return loss
