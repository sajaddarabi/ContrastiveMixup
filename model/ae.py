from base import BaseModel
from .nn_utils import *
from .sup_con_loss import SupConLoss
from .loss import ae_loss, nll_loss, l2_loss, bce_loss, interp_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AE(BaseModel):
    def __init__(self,
                 input_dim,
                 hidden_dim=[128, ],
                 cont_idxs=[],
                 cat_idxs=[],
                 cat_dims=[],
                 cat_choice_function=F.log_softmax,
                 cat_loss=nll_loss,
                 mixup_within_class=True,
                 mixup_method='',
                 mixup_alpha=1.0,
                 mixup_dist='alpha',
                 mixup_n=1,
                 embed=True,
                 decoder_net=True,
                 projection_head='linear',
                 projection_dim=None,
                 num_layers=None,
                 num_classes=0.0,
                 **kwargs):
        super().__init__()
        assert mixup_method in ['', 'mixup_hidden', 'mixup', 'latent'], f'mixup_method not valid'

        # - hidden dim
        if type(hidden_dim) == str:
            hidden_dim = eval(hidden_dim)
            hidden_dim = list(map(int, hidden_dim))
        elif type(hidden_dim) != list:
            hidden_dim = [int(hidden_dim), ]
            if num_layers != None:
                hidden_dim = hidden_dim * num_layers

        # - projection dim
        if projection_dim is None:
            projection_dim = hidden_dim[-1]
        elif type(projection_dim) == list:
            projection_dim = projection_dim[0]
        elif type(projection_dim) == str:
            projection_dim = int(eval(projection_dim))
        elif type(projection_dim) == float:
            projection_dim = int(projection_dim)
        if (len(cat_idxs) == 0) and (len(cont_idxs) == 0):
            # assume all cont
            cont_idxs = list(range(input_dim))

        # - embeddings
        if embed:
            self.embeddings = EmbeddingGenerator(input_dim, cat_dims, cat_idxs)
        else:
            self.embeddings = EmbeddingGenerator(input_dim, [], [])

        # - ae params
        self.decoder_net = decoder_net
        self.input_dim = input_dim
        self.post_embed_dim = self.embeddings.post_embed_dim
        hidden_dim = [self.post_embed_dim] + hidden_dim
        self.num_classes = num_classes

        self.hidden_dim = hidden_dim
        self.cont_idxs = sorted(cont_idxs)
        self.cat_idxs, self.cat_dims = [], []

        if (len(cat_idxs) and len(cat_dims)):
            self.cat_idxs, self.cat_dims = zip(*sorted(zip(cat_idxs, cat_dims)))

        self.cat_loss = cat_loss
        self.cat_choice_function = cat_choice_function

        # - mixup params
        self.mixup_method = mixup_method
        self.mixup_alpha = mixup_alpha
        self.mixup_dist = mixup_dist
        self.mixup_n = mixup_n
        self.mixup_within_class = mixup_within_class

        # - encoder
        self.encoder = nn.ModuleList()
        self.encoder.append(self.embeddings)
        for i in range(1, len(hidden_dim)):
            self.encoder.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim[i-1], hidden_dim[i]),
                        #GBN(hidden_dim[i]),
                        #nn.PReLU(hidden_dim[i])
                        nn.ReLU()
                        )
                    )

        # - init projection layer for mixup methods
        if self.mixup_method:
            projection_layer = []
            if projection_head == 'linear':
                projection_layer.append(nn.Linear(hidden_dim[-1], projection_dim))
            elif projection_head == 'mlp':
                projection_layer.append(nn.Linear(hidden_dim[-1], hidden_dim[-1]))
                projection_layer.append(nn.ReLU())
                projection_layer.append(nn.Linear(hidden_dim[-1], projection_dim))
            self.projection_layer = nn.Sequential(*projection_layer)

            self.contrast_loss = SupConLoss(**kwargs)
            self.bce_loss = nn.BCELoss()


        # - decoder
        if self.decoder_net:
            # Decoder
            hidden_dim = hidden_dim
            self.decoder = []
            for i in range(len(hidden_dim) - 1, 1, -1):
                self.decoder.extend(
                        (
                            nn.Linear(hidden_dim[i], hidden_dim[i-1]),
                            #GBN(hidden_dim[i-1]),
                            nn.ReLU()
                            )
                        )
            self.decoder = torch.nn.Sequential(*self.decoder)

            if (len(self.cont_idxs) != 0):
                self.cont_net = nn.Sequential(
                        nn.Linear(hidden_dim[1], len(self.cont_idxs)),
                        )

            if (len(self.cat_idxs) != 0):
                self.cat_nets = nn.ModuleList()

            for i, n_cats in zip(self.cat_idxs, self.cat_dims):
                self.cat_nets.append(nn.Sequential(
                    nn.Linear(hidden_dim[1], n_cats),
                    Lambda(cat_choice_function)
                    ))

        self.apply(weight_init)

    def decode(self, z):
        'note: order of decoding is important for loss function'
        z = self.decoder(z)
        x_hat = []

        if (hasattr(self, 'cont_net')):
            x_hat.append(self.cont_net(z))

        if (hasattr(self, 'cat_nets')):
            for m in self.cat_nets:
                x_hat.append(m(z))

        return x_hat

    def encode(self, x, target=None):
        for i, l in enumerate(self.encoder):
            x = l(x)
        return x

    def forward(self, data, device=None, **kwargs):
        # - Input
        target, mixup_x, x_u, weights = None, None, None, None
        if type(data) == torch.Tensor:
            x = data
            x = x.to(device)
        elif len(data) == 3:# semi-supervised
            x, x_u, target = data
            x, x_u, target = x.to(device), x_u.to(device), target.to(device)

            if not self.mixup_within_class:
                x = torch.cat([x, x_u], dim=0)
                x_u = None

        elif len(data) == 4: # semi-supervised + pseudolabels
            x, x_u, target, utarget = data
            x, x_u, target, utarget = x.to(device), x_u.to(device), target.to(device), utarget.to(device)
            # TODO: seperate out x_u and x if you want to control loss weight
            x = torch.cat([x, x_u], dim=0)
            target = torch.cat([target, utarget], dim=0)
            x_u = None
        elif len(data) == 5:
            x, x_u, target, utarget, uweight = data
            x, x_u, target, utarget, uweight = x.to(device), x_u.to(device), target.to(device), utarget.to(device), uweight.to(device)
            # TODO: seperate out x_u and x if you want to control loss weight
            x = torch.cat([x, x_u], dim=0)
            target = torch.cat([target, utarget], dim=0)
            weights = torch.cat([torch.ones_like(uweight, device=uweight.device), uweight], dim=0)
            x_u = None
        else:
            x, target = data
            x, target = x.to(device), target.to(device)


        lambda_indices, one_minus_lambda_indices, lam = None, None, None

        # - Sample mixup layer
        if self.mixup_method == 'mixup_hidden':
            layer = np.random.randint(0, len(self.encoder))
        elif self.mixup_method == 'mixup':
            layer = 0
        elif self.mixup_method == 'latent':
            layer = len(self.encoder) - 1
        else:
            layer = -1

        # - Alpha for beta distribution
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

        # - Encode to latent space
        for i, l in enumerate(self.encoder):
            # - labeled samples
            x = l(x)

            # - unlabeled samples
            if x_u is not None:
                x_u = l(x_u)

            # - push mixup samples through the rest of the layers
            if mixup_x is not None:
                mixup_x = l(mixup_x)

            # - mixup classes
            if i == layer and target is not None:
                if self.mixup_within_class:
                    mixup_x, lambda_indices, one_minus_lambda_indices = mixup_class(x, target, lam)
                else:
                    mixup_x, lambda_indices = mixup_process_label_free(x, lam)

        z_mixup = mixup_x
        z = x
        z_u = x_u

        # - projection to metric space
        z_proj, z_proj_mixup = None, None
        if hasattr(self, 'projection_layer'):
            z_proj = self.projection_layer(z)
            z_proj = F.normalize(z_proj, dim=1)
            if z_mixup  is not None:
                z_proj_mixup = self.projection_layer(z_mixup )
                z_proj_mixup = F.normalize(z_proj_mixup, dim=1)

        # - Decode samples back to data space
        x_hat = None
        x_u_hat = None
        if self.decoder_net:
            x_hat = self.decode(z)
            if z_u is not None:
                x_u_hat = self.decode(z_u)

        x_hat_mixup = None
        if self.decoder_net and (z_mixup  is not None):
            x_hat_mixup = self.decode(z_mixup)

        # - Setup output dict
        output = {}
        output['x_hat'] = x_hat
        output['x_u_hat'] = x_u_hat
        output['z'] = z
        output['z_u'] = z_u
        output['x_hat_mixup'] = x_hat_mixup
        output['z_mixup'] = z_mixup
        output['z_proj_mixup'] = z_proj_mixup
        output['z_proj'] = z_proj
        output['lambda_indices'] = lambda_indices
        output['one_minus_lambda_indices'] = one_minus_lambda_indices
        output['preds'] = torch.cat(x_hat, dim=1)
        output['lam'] = lam

        return output

    def decode_sample(self, z):
        x_hat = self.decode(z)
        x_cont, x_cat = [], []

        if (hasattr(self, 'cont_net')):
            x_cont = x_hat.pop(0)

        if (hasattr(self, 'cat_nets')):
            for i in self.cat_idxs:
                x_cat.append(torch.argmax(x_hat.pop(0), dim=1))

        x = []
        cont_c, cat_c = 0, 0
        for i in range(self.input_dim):
            if i in self.cont_idxs:
                x.append(x_cont[:, cont_c].reshape(-1, 1))
                cont_c += 1
            elif i in self.cat_idxs:
                x.append(x_cat[cat_c].reshape(-1, 1))
                cat_c += 1
        x = torch.cat(x, dim=1)
        return x

    def loss(self, output, data, device, **kwargs):
        weights = None
        if type(data) == torch.Tensor:
            x = data
            x = x.to(device)
        elif len(data) == 3:# semi-supervised
            x, x_u, target = data
            x, x_u, target = x.to(device), x_u.to(device), target.to(device)
            #x = torch.cat([x, x_u], dim=0)
        elif len(data) == 4: # semi-supervised + pseudolabels
            x, x_u, target, utarget = data
            x, x_u, target, utarget = x.to(device), x_u.to(device), target.to(device), utarget.to(device)
        elif len(data) == 5:
            x, x_u, target, utarget, uweight = data
            x, x_u, target, utarget, uweight = x.to(device), x_u.to(device), target.to(device), utarget.to(device), uweight.to(device)
            # TODO: seperate out x_u and x if you want to control loss weight
            x = torch.cat([x, x_u], dim=0)
            target = torch.cat([target, utarget], dim=0)
            weights = torch.cat([torch.ones_like(uweight, device=uweight.device), uweight], dim=0)
        else:
            x, target = data
            x, target = x.to(device), target.to(device)

        loss = {'mse': 0.0, 'nll': 0.0, 'mse_mixup': 0.0, 'nll_mixup': 0.0}

        x_hat = output['x_hat']
        x_u_hat = output['x_u_hat']
        z = output['z']
        z_proj = output['z_proj']
        z_proj_mixup = output['z_proj_mixup']
        x_hat_mixup = output['x_hat_mixup']
        z_mixup  = output['z_mixup']
        lambda_indices = output['lambda_indices']
        one_minus_lambda_indices = output['one_minus_lambda_indices']
        lam = output['lam']

        l2_weight_decoder = kwargs.get('l2_weight_decoder', 0.0)
        latent_reg = kwargs.get('latent_reg', 0.0)
        recon_weight = kwargs.get('recon_weight', 1.0)
        contrastive_weight = kwargs.get('contrastive_weight', 0.0)
        mixup_weight_decoder = kwargs.get('mixup_weight_decoder', 0.0)

        cat_weight = kwargs.get('cat_weight', len(self.cat_idxs) / self.input_dim)
        cont_weight = kwargs.get('cont_weight', len(self.cont_idxs) / self.input_dim)

        if self.decoder_net and (len(self.cont_idxs)):
            out = x_hat.pop(0)
            loss['mse'] += recon_weight * cont_weight * ae_loss([out], x[:, self.cont_idxs], **kwargs)
            # - unlabeled samples
            if x_u_hat is not None:
                out = x_u_hat.pop(0)
                loss['mse'] += recon_weight * cont_weight * ae_loss([out], x_u[:, self.cont_idxs], **kwargs)

            if x_hat_mixup is not None:
                out = x_hat_mixup.pop(0)
                if self.mixup_within_class:
                    target_mixup = mixup_full_indices(x[:, self.cont_idxs], lambda_indices, one_minus_lambda_indices, lam)
                else:
                    target_mixup = mixup(x[:, self.cont_idxs], lambda_indices, lam)
                loss['mse_mixup'] += mixup_weight_decoder * cont_weight * \
                        ae_loss([out], target_mixup, **kwargs)

        if self.decoder_net and (len(self.cat_idxs)):
            for i, idx in enumerate(self.cat_idxs):
                out = x_hat.pop(0)
                loss['nll'] += recon_weight * cat_weight * self.cat_loss(out, x[:, idx].long())

                # - unlabeled samples
                if x_u_hat is not None:
                    out = x_u_hat.pop(0)
                    loss['nll'] += recon_weight * cat_weight * self.cat_loss(out, x_u[:, idx].long())
                if x_hat_mixup is not None:
                    out = x_hat_mixup.pop(0)
                    target_reweighted = to_one_hot(x[:, idx], self.cat_dims[i])
                    if self.mixup_within_class:
                        target_reweighted = mixup_full_indices(target_reweighted, lambda_indices, one_minus_lambda_indices, lam)
                    else:
                        target_reweighted = mixup(target_reweighted, lambda_indices, lam)

                    loss['nll_mixup'] += mixup_weight_decoder * cat_weight * self.bce_loss(F.softmax(out), target_reweighted)

        if self.decoder_net and l2_weight_decoder:
            l2_reg = 0.0
            for param in self.decoder.parameters():
                l2_reg += torch.norm(param)
            loss['decoder_reg'] = l2_weight_decoder * l2_reg

        if z_proj_mixup is not None:
            if self.mixup_within_class:
                z_m = z_proj_mixup[lambda_indices] # - order indices
                z_m = torch.unsqueeze(z_m, dim=1)
                z_proj = torch.unsqueeze(z_proj, dim=1)
                zs = torch.cat([z_proj, z_m], dim=1)

                if weights is not None:
                    weights_mixup = mixup_full_indices(weights, lambda_indices, one_minus_lambda_indices, lam)
                    weights_mixup[lambda_indices] = weights_mixup
                    weights = torch.cat([weights, weights_mixup], dim=0)

                loss['c_mixup'] = contrastive_weight * self.contrast_loss(zs, target, weights=weights)
            else: # random mixing
                mask = torch.eye(z_proj.size(0))
                mask[torch.arange(z_proj.size(0)), torch.arange(z_proj.size(0))] = lam.squeeze().cpu()
                mask[torch.arange(z_proj.size(0)), lambda_indices] = 1 - lam.squeeze().cpu()


                z_proj_mixup = torch.unsqueeze(z_proj_mixup, dim=1)
                z_proj = torch.unsqueeze(z_proj, dim=1)
                zs = torch.cat([z_proj, z_proj_mixup], dim=1)

                loss['c_mixup'] = contrastive_weight * self.contrast_loss(zs, mask=mask)


        if latent_reg:
            loss['latent_reg'] = latent_reg * torch.norm(z)

        total_loss  =  0.0
        for k, v in loss.items():
            total_loss += loss[k]

        loss['opt'] = total_loss
        return loss
