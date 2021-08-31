import torch
import torch.nn as nn
from base import BaseModel
from .nn_utils import *
import torch.nn.functional as F

VALID_MIXUP_METHODS = ["mixup", "mixup_hidden", ""]

class MLP(BaseModel):
    def __init__(self,
                 input_dim,
                 num_classes,
                 hidden_dim=[128, ],
                 cont_idxs=[],
                 cat_idxs=[],
                 cat_dims=[],
                 encoder=None,
                 fine_tune=False,
                 mixup_method='',
                 mixup_alpha=1.0,
                 mixup_dist='alpha',
                 mixup_n=1,
                 p_m=0.3,
                 K=3,
                 embed=False,
                 **kwargs):
        super().__init__()
        assert mixup_method.lower() in VALID_MIXUP_METHODS, "Valid choices are {}".format(VALID_MIXUP_METHODS)

        # - data paramns
        self.input_dim = input_dim
        self.num_classes = num_classes

        # - mixup params
        self.mixup_method = mixup_method.lower()
        self.mixup_alpha = mixup_alpha
        self.mixup_n = mixup_n
        self.mixup_dist = mixup_dist

        # - encoder
        self.encoder = encoder
        if type(hidden_dim) == str:
            hidden_dim = eval(hidden_dim)
            hidden_dim = list(map(int, hidden_dim))
        self.hidden_dim = hidden_dim

        # - add embedding layer for cat columns
        if embed and encoder is None:
            self.embeddings = EmbeddingGenerator(input_dim, cat_dims, cat_idxs)
        else:
            self.embeddings = EmbeddingGenerator(input_dim, [], [])

        self.post_embed_dim = self.embeddings.post_embed_dim
        self.fine_tune = fine_tune
        hidden_dim = [self.post_embed_dim] + hidden_dim

        # - predictor
        self.predictor = nn.ModuleList()
        self.predictor.append(self.embeddings)
        for i in range(1, len(hidden_dim)):
            self.predictor.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                        nn.ReLU()
                    )
            )

        self.predictor.append(nn.Linear(hidden_dim[-1], self.num_classes))

        self.weight = kwargs.get('weight', None)
        if self.weight is not None:
            self.weight = torch.Tensor(self.weight)
        self.criterion_mixup = nn.BCELoss(reduction='none')
        self.criterion_softmax = nn.CrossEntropyLoss(weight=self.weight)

    def forward(self, data, device, **kwargs):
        # - input
        x, x_u, target, utarget = None, None, None, None
        if type(data) == torch.Tensor:
            x = data
            x = x.to(device)
            target = None
        elif len(data) == 3:
            x, x_u, target = list(map(lambda a: a.to(device), data))
        elif len(data) == 4:
            x, x_u, target, utarget = list(map(lambda a: a.to(device), data))
        elif len(data) == 5:
            x, x_u, target, utarget, _ = list(map(lambda a: a.to(device), data))
        else:
            x, target = list(map(lambda a: a.to(device), data))

        out_target = target
        # - flatten input..
        x = torch.flatten(x, start_dim=1)
        # - mixup methods
        target_reweighted = None
        utarget_reweighted = None
        indices = None
        uindices = None
        layer = -1

        if self.mixup_method == "mixup_hidden":
            layer = np.random.randint(0, len(self.predictor))
        elif self.mixup_method == "mixup":
            layer = 0

        if x_u is not None:
                x, target = x.to(device), target.to(device)
                if x_u is not None:
                    x_u = x_u.to(device)
                if utarget is not None:
                    utarget = utarget.to(device)
        else:
            x, target = x.to(device), target.to(device)
            if utarget is not None:
                utarget = utarget.to(device)

        # There's a dependency in the order.. of these ifs
        # Below must happen after the if statements above

        # - Mixup on validation
        if not kwargs.get('validation_target', True):
            target = None
            utarget = None

        # - convert labeled to one hot
        if target is not None and layer > -1:
            target_reweighted = to_one_hot(target, self.num_classes)
        # - convert unlabeled target to one hot
        if utarget is not None and layer > -1:
            utarget_reweighted = to_one_hot(utarget, self.num_classes)

        # - sample lam
        if self.mixup_alpha is not None:
            if self.mixup_n == -1: # - set n to batch_size
                mixup_n = x.shape[0]
            else:
                mixup_n = self.mixup_n

            lam = get_lambda(self.mixup_alpha, self.mixup_dist, n=mixup_n)
            if type(lam) == float:
                lam = [lam]
            lam = torch.from_numpy(np.array(lam).astype('float32')).to(x.device)
            lam = Variable(lam)
            lam = lam.reshape(-1, 1)

        # - enocde samples
        if self.encoder:
            if self.fine_tune:
                self.encoder = self.encoder.train()
            else:
                self.encoder = self.encoder.eval()

            x = self.encoder.encode(x)
            if (x_u is not None) and (self.mixup_method != ''):
                x_u = self.encoder.encode(x_u)


        # - feed samples to predictor
        for i, l in enumerate(self.predictor):
            # - normal smaples
            x = l(x) # l == 0 is the embedding layer if embed=True
                     # mixup embedding vectors for cat columns

            # - unlabeled samples
            if  x_u is not None and (self.mixup_method != ''):
                x_u = l(x_u)

            if i == layer and target_reweighted is not None:
                x, target_reweighted, indices = mixup_process(x, target_reweighted, lam)

            if (i == layer) and (x_u is not None) and (utarget_reweighted is not None) and (self.mixup_method != ''):
                x_u, utarget_reweighted, uindices = mixup_process(x_u, utarget_reweighted, lam)


        out = {}
        out['target_reweighted'] = target_reweighted
        out['utarget_reweighted'] = utarget_reweighted
        out['logits'] = x
        out['target'] = out_target
        out['ulogits'] = x_u
        out['layer'] = layer
        out['lam'] = lam
        out['uindices'] = uindices
        out['indices'] = indices
        out['preds'] = torch.nn.functional.softmax(x).detach()
        return out

    def loss(self, outputs, data, device, **kwargs):
        weights, utarget = None, None
        if len(data) == 3:
            x, x_u, target = data
            x, x_u, target = list(map(lambda a: a.to(device), data))
        elif len(data) == 4:
            x, x_u, target, utarget = data
            x, x_u, target, utarget = list(map(lambda a: a.to(device), data))
            x = torch.cat([x, x_u], dim=0)
            target = torch.cat([target, utarget], dim=0)
        elif len(data) == 5:
            x, x_u, target, utarget, uweight = data
            x, x_u, target, utarget, uweight = list(map(lambda a: a.to(device), data))
        else:
            x, target = data
            x, target = list(map(lambda a: a.to(device), data))

        # - set loss weights
        unlabeled_samples_weight = kwargs.get('unlabeled_classification_weight', 0.0)

        # - get outputs
        target_reweighted = outputs.get("target_reweighted", None)
        utarget_reweighted = outputs.get("utarget_reweighted", None)
        logits = outputs.get("logits", None)
        ulogits = outputs.get("ulogits", None)
        logits_x_u_aug_mixup = outputs.get('logits_x_u_aug_mixup', None)
        x_u_aug_mixup_idxs = outputs.get('logits_x_u_aug_mixup_idxs', None)
        x_u_aug_mixup_lams = outputs.get('logits_x_u_aug_mixup_lams', None)
        # - compute loss
        loss = {}
        if target_reweighted is not None:
            losses = self.criterion_mixup(F.softmax(logits), target_reweighted)
            if self.weight is not None:
                weights = torch.ones(losses.shape[0], device=device)
                for i, w in enumerate(self.weight):
                    weights[target == i] = weights[target == i] * w
                losses = losses * weights.unsqueeze(1)
            losses = losses.mean()

            loss['bce_loss'] = losses
        else:
            loss['nll'] = self.criterion_softmax(logits, target.long())

        if utarget_reweighted is not None:
            loses = unlabeled_samples_weight * self.criterion_mixup(F.softmax(ulogits), target_reweighted)
            if self.weight is not None:
                weights = torch.ones(losses.shape[0], device=device)
                for i, w in enumerate(self.weight):
                    weights[target == i] = weights[target == i] * w
                losses = losses * weights.unsqueeze(1)
            losses = losses.mean()
            loss['ubce_loss'] = losses
        elif ulogits is not None and (utarget is not None):
            loss['unll'] = unlabeled_samples_weight * self.criterion_softmax(ulogits, utarget.long())

        # - aggregate loss
        total_loss  =  0.0
        for k, v in loss.items():
            total_loss += loss[k]
        loss['opt'] = total_loss

        return loss
