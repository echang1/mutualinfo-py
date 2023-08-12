import math
import numpy as np
import numpy.linalg as la
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.special import digamma
from sklearn import cluster
from sklearn.neighbors import BallTree, KDTree
from torch.autograd import Variable
from torch.nn.modules import Module
from torch.nn.parameter import Parameter

EPS = 1e-6

#from mighty.monitor.mutual_info.mutual_info.py
def to_bits(entropy_nats):
    """
        Converts nats to bits.
        Parameters
        ----------
        entropy_nats : float
            Entropy in nats.
        Returns
        -------
        float
            Entropy in bits.
        """
    log2e = math.log2(math.e)
    return entropy_nats * log2e

#from entropy_estimators.py
def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        #Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
        knn_points = points[knn]
        #Subtract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        #Calculate covariance matrix of k-nearest neighbor points, obtain eigenvectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        #Calculate PCA-bounding box using eigenvectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        #Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()
        #Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e

#from entropy_estimators.py
def add_noise(x, intens=1e-10):
    #small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)

#from entropy_estimators.py
def query_neighbors(tree, x, k):
    return tree.query(x, k=k+1)[0][:, k]

#from entropy_estimators.py
def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)

#from entropy_estimators.py
def avgdigamma(points, dvec):
    #This part finds number of neighbors in some radius in the marginal space
    #returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

#from entropy_estimators.py
def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric="chebyshev")
    return KDTree(points, metric="chebyshev")

#from Linear_Function.py
def _quantize(activations: torch.FloatTensor, n_bins=10) -> np.ndarray:
    model = cluster.MiniBatchKMeans(n_clusters=n_bins, batch_size=100)
    labels = model.fit_predict(activations)
    return labels

#from mine.models.layers.py
class ConcatLayer(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)

#from mine.models.layers.py 
class CustomSequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input

#from mine.models.mine.py
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None

#from mine.models.mine.py
def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema

#from mine.models.mine.py
def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)
    # Recalculate ema
    return t_log, running_mean

#from mine.models.mine.py
class T(nn.Module):
    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 400), nn.ReLU(), nn.Linear(400, 400), nn.ReLU(), nn.Linear(400, 400), nn.ReLU(), nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)

#from mine.models.mine.py
class MutualInformationEstimator(pl.LightningModule):
    def __init__(self, x_dim, z_dim, loss='mine', **kwargs):
        super().__init__()
        self.x_dim = x_dim
        self.T = CustomSequential(ConcatLayer(), nn.Linear(x_dim + z_dim, 100), nn.ReLU(),
                                  nn.Linear(100, 100), nn.ReLU(), nn.Linear(100, 1))

        self.energy_loss = Mine(self.T, loss=loss, alpha=kwargs['alpha'])

        self.kwargs = kwargs

        self.train_loader = kwargs.get('train_loader')
        self.test_loader = kwargs.get('test_loader')

    def forward(self, x, z):
        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        return self.energy_loss(x, z)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.kwargs['lr'])

    def training_step(self, batch, batch_idx):

        x, z = batch

        if self.on_gpu:
            x = x.cuda()
            z = z.cuda()

        loss = self.energy_loss(x, z)
        mi = -loss
        tensorboard_logs = {'loss': loss, 'mi': mi}
        tqdm_dict = {'loss_tqdm': loss, 'mi': mi}

        return {
            **tensorboard_logs, 'log': tensorboard_logs, 'progress_bar': tqdm_dict
        }

    def test_step(self, batch, batch_idx):
        x, z = batch
        loss = self.energy_loss(x, z)

        return {
            'test_loss': loss, 'test_mi': -loss
        }

    def test_end(self, outputs):
        avg_mi = torch.stack([x['test_mi']
                              for x in outputs]).mean().detach().cpu().numpy()
        tensorboard_logs = {'test_mi': avg_mi}

        self.avg_test_mi = avg_mi
        return {'avg_test_mi': avg_mi, 'log': tensorboard_logs}

    #@pl.data_loader
    def train_dataloader(self):
        if self.train_loader:
            return self.train_loader

        train_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return train_loader

    #@pl.data_loader
    def test_dataloader(self):
        if self.test_loader:
            return self.train_loader

        test_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return test_loader

#from mine.models.mine.py
class Mine(nn.Module):
    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method

        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in['mine_biased']:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):

        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=1e-4)
           
        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()
                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
        final_mi = self.mi(X, Y)
        return final_mi.item()
    
#from mine.utils.helpers.py
def batch(x, y, batch_size=1, shuffle=True):
    assert len(x) == len(
        y), "Input and target data must contain same number of elements"
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    n = len(x)

    if shuffle:
        rand_perm = torch.randperm(n)
        x = x[rand_perm]
        y = y[rand_perm]

    batches = []
    for i in range(n // batch_size):
        x_b = x[i * batch_size: (i + 1) * batch_size]
        y_b = y[i * batch_size: (i + 1) * batch_size]

        batches.append((x_b, y_b))
    return batches

#from ib_layers.py
def reparameterize(mu, logvar, batch_size, cuda=False, sampling=True):
    # output dim: batch_size * dim
    if sampling:
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(batch_size, std.size(0)).to(mu).normal_()
        eps = Variable(eps)
        return mu.view(1, -1) + eps * std.view(1, -1)
    else:
        return mu.view(1, -1)

#from ib_layers.py
class InformationBottleneck(Module):
    def __init__(self, dim, mask_thresh=0, init_mag=9, init_var=0.01,
                kl_mult=1, divide_w=False, sample_in_training=True, sample_in_testing=False, masking=False):
        super(InformationBottleneck, self).__init__()
        self.prior_z_logD = Parameter(torch.Tensor(dim))
        self.post_z_mu = Parameter(torch.Tensor(dim))
        self.post_z_logD = Parameter(torch.Tensor(dim))

        self.epsilon = 1e-8
        self.dim = dim
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        # if masking=True, apply mask directly
        self.masking = masking

        # initialization
        stdv = 1. / math.sqrt(dim)
        self.post_z_mu.data.normal_(1, init_var)
        self.prior_z_logD.data.normal_(-init_mag, init_var)
        self.post_z_logD.data.normal_(-init_mag, init_var)

        self.need_update_z = True # flag for updating z during testing
        self.mask_thresh = mask_thresh
        self.kl_mult=kl_mult
        self.divide_w=divide_w


    def adapt_shape(self, src_shape, x_shape):
        # to distinguish conv layers and fc layers
        # see if we need to expand the dimension of x
        new_shape = src_shape if len(src_shape)==2 else (1, src_shape[0])
        if len(x_shape)>2:
            new_shape = list(new_shape)
            new_shape += [1 for i in range(len(x_shape)-2)]
        return new_shape

    def get_logalpha(self):
        return self.post_z_logD.data - torch.log(self.post_z_mu.data.pow(2) + self.epsilon)

    def get_dp(self):
        logalpha = self.get_logalpha()
        alpha = torch.exp(logalpha)
        return alpha / (1+alpha)

    def get_mask_hard(self, threshold=0):
        logalpha = self.get_logalpha()
        hard_mask = (logalpha < threshold).float()
        return hard_mask

    def get_mask_weighted(self, threshold=0):
        logalpha = self.get_logalpha()
        mask = (logalpha < threshold).float()*self.post_z_mu.data
        return mask

    def forward(self, x):
        # 4 modes: sampling, hard mask, weighted mask, use mean value
        if self.masking:
            mask = self.get_mask_hard(self.mask_thresh)
            new_shape = self.adapt_shape(mask.size(), x.size())
            return x * Variable(mask.view(new_shape))

        bsize = x.size(0)
        if (self.training and self.sample_in_training) or (not self.training and self.sample_in_testing):
            z_scale = reparameterize(self.post_z_mu, self.post_z_logD, bsize, cuda=True, sampling=True)
            if not self.training:
                z_scale *= Variable(self.get_mask_hard(self.mask_thresh))
        else:
            z_scale = Variable(self.get_mask_weighted(self.mask_thresh))
        self.kld = self.kl_closed_form(x)
        new_shape = self.adapt_shape(z_scale.size(), x.size())
        return x * z_scale.view(new_shape)  

    def kl_closed_form(self, x):
        new_shape = self.adapt_shape(self.post_z_mu.size(), x.size())


        h_D = torch.exp(self.post_z_logD.view(new_shape))
        h_mu = self.post_z_mu.view(new_shape)

        KLD = torch.sum(torch.log(1 + h_mu.pow(2)/(h_D + self.epsilon) )) * x.size(1) / h_D.size(1)

        if x.dim() > 2:
            if self.divide_w:
                # divide it by the width
                KLD *= x.size()[2]
            else:
                KLD *= np.prod(x.size()[2:])
        return KLD * 0.5 * self.kl_mult

#from ib_vgg.py
# model configuration, (out_channels, kl_multiplier), 'M': Mean pooling, 'A': Average pooling
cfg = {
    'D6': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D5': [(64, 1.0/32**2), (64, 1.0/32**2), 'M', (128, 1.0/16**2), (128, 1.0/16**2), 'M', (256, 1.0/8**2), (256, 1.0/8**2), (256, 1.0/8**2), 
        'M', (512, 1.0/4**2), (512, 1.0/4**2), (512, 1.0/4**2), 'M', (512, 1.0/2**2), (512, 1.0/2**2), (512, 1.0/2**2), 'M'],
    'D4': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'],
    'D3': [(64, 0.1), (64, 0.1), 'M', (128, 0.5), (128, 0.5), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D2': [(64, 0.01), (64, 0.01), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D1': [(64, 0.1), (64, 0.1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'D0': [(64, 1), (64, 1), 'M', (128, 1), (128, 1), 'M', (256, 1), (256, 1), (256, 1), 
        'M', (512, 1), (512, 1), (512, 1), 'M', (512, 1), (512, 1), (512, 1), 'M'],
    'G':[(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), 
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'M'], # VGG 16 with one fewer FC
    'G5': [(64, 1.0/32), (64, 1.0/32), 'M', (128, 1.0/16), (128, 1.0/16), 'M', (256, 1.0/8), (256, 1.0/8), (256, 1.0/8), (256, 1.0/8),
        'M', (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), (512, 1.0/4), 'M', (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), (512, 1.0/2), 'A']
}

#from ib_vgg.py
class VGG_IB(nn.Module):
    def __init__(self, config=None, mag=9, batch_norm=False, threshold=0, 
                init_var=0.01, sample_in_training=True, sample_in_testing=False, n_cls=10, no_ib=False):
        super(VGG_IB, self).__init__()

        self.init_mag = mag
        self.threshold = threshold
        self.config = config
        self.init_var = init_var
        self.sample_in_training = sample_in_training
        self.sample_in_testing = sample_in_testing
        self.no_ib = no_ib

        self.conv_layers, conv_kl_list = self.make_conv_layers(cfg[config], batch_norm)
        print('Using structure {}'.format(cfg[config]))

        fc_ib1 = InformationBottleneck(512, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing)
        fc_ib2 = InformationBottleneck(512, mask_thresh=threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    sample_in_training=sample_in_training, sample_in_testing=sample_in_testing)
        self.n_cls = n_cls
        if self.config in ['G', 'D6']:
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)] if no_ib else \
                            [nn.Linear(512, 512), nn.ReLU(), fc_ib1, nn.Linear(512, self.n_cls)] 
            self.fc_layers = nn.Sequential(*fc_layer_list)
            self.kl_list = conv_kl_list + [fc_ib1]
        elif self.config == 'G5':
            self.fc_layers = nn.Sequential(nn.Linear(512, self.n_cls))
            self.kl_list = conv_kl_list
        else:
            fc_layer_list = [nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, self.n_cls)] if no_ib else \
                    [nn.Linear(512, 512), nn.ReLU(), fc_ib1, nn.Linear(512, 512), nn.ReLU(), fc_ib2, nn.Linear(512, self.n_cls)]
            self.fc_layers = nn.Sequential(*fc_layer_list)
            self.kl_list = conv_kl_list + [fc_ib1, fc_ib2]

    def make_conv_layers(self, config, batch_norm):
        layers, kl_list = [], []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v[0], kernel_size=3, padding=1)
                in_channels = v[0]
                ib = InformationBottleneck(v[0], mask_thresh=self.threshold, init_mag=self.init_mag, init_var=self.init_var, 
                    kl_mult=v[1], sample_in_training=self.sample_in_training, sample_in_testing=self.sample_in_testing)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                if not self.no_ib:
                    layers.append(ib)
                    kl_list.append(ib)
        return nn.Sequential(*layers), kl_list

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layers(x).view(batch_size, -1)
        x = self.fc_layers(x)

        if self.training and (not self.no_ib):
            ib_kld = self.kl_list[0].kld
            for ib in self.kl_list[1:]:
                ib_kld += ib.kld
            
            return x, ib_kld
        else:
            return x

    def get_masks(self, hard_mask=True, threshold=0):
        masks = []
        if hard_mask:
            masks = [ib_layer.get_mask_hard(threshold) for ib_layer in self.kl_list]
            return masks, [np.sum(mask.cpu().numpy()==0) for mask in masks]
        else:
            masks = [ib_layer.get_mask_weighted(threshold) for ib_layer in self.kl_list]
            return masks

    def print_compression_ratio(self, threshold, writer=None, epoch=-1):
        # applicable for structures with global pooling before fc
        _, prune_stat = self.get_masks(hard_mask=True, threshold=threshold)
        conv_shapes = [v[0] for v in cfg[self.config] if type(v) is not str]

        if self.config in ['G', 'D6']:
            fc_shapes = [512]
        elif self.config == 'G5':
            fc_shapes = []
        else:
            fc_shapes = [512, 512]

        net_shape = [ out_channels-prune_stat[idx] for idx, out_channels in enumerate(conv_shapes+fc_shapes)]
        #conv_shape_with_pool = [v[0] if v != 'M' else 'M' for v in cfg[self.config]]
        current_n, hdim, last_channels, flops, fmap_size = 0, 64, 3, 0, 32
        for n, pruned_channels in enumerate(prune_stat):
            if n < len(conv_shapes):
                current_channels = cfg[self.config][current_n][0] - pruned_channels
                flops += (fmap_size**2) * 9 * last_channels * current_channels
                last_channels = current_channels
                current_n += 1
                if type(cfg[self.config][current_n]) is str:
                    current_n += 1
                    fmap_size /= 2
                    hdim *= 2
            else:
                current_channels = 512 - pruned_channels
                flops += last_channels * current_channels
                last_channels = current_channels
        flops += last_channels * self.n_cls

        total_params, pruned_params, remain_params = 0, 0, 0
        # total number of conv params
        in_channels, in_pruned = 3, 0
        for n, n_out in enumerate(conv_shapes):
            n_params = in_channels * n_out * 9
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_stat[n]) * 9
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out
            in_pruned = prune_stat[n]
        # fc layers
        offset = len(prune_stat) - len(fc_shapes)
        for n, n_out in enumerate(fc_shapes):
            n_params = in_channels * n_out
            total_params += n_params
            n_remain = (in_channels - in_pruned) * (n_out - prune_stat[n+offset])
            remain_params += n_remain
            pruned_params += n_params - n_remain
            in_channels = n_out
            in_pruned = prune_stat[n+offset]
        total_params += in_channels * self.n_cls
        remain_params += (in_channels - in_pruned) * self.n_cls
        pruned_params += in_pruned * self.n_cls

        print('total parameters: {}, pruned parameters: {}, remaining params:{}, remain/total params:{}, remaining flops: {}, '
              'each layer pruned: {},  remaining structure:{}'.format(total_params, pruned_params, remain_params, 
                    float(total_params-pruned_params)/total_params, flops, prune_stat, net_shape))
        #if writer is not None:
            #writer.add_scalar('flops', flops, epoch)
            

