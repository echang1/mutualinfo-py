#----
#CORRECTION FUNCTIONS
#----

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

#----
#UTILITY FUNCTIONS
#----

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
    #This part finds the number of neighbors in some radius in the marginal space
    #returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))

#from entropy_estimators.py
def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')

#from Linear_Function.py
def _quantize(activations: torch.FloatTensor) -> np.ndarray:
    n_bins = 10
    model = cluster.MiniBatchKMeans(n_clusters=n_bins, batch_size=100)
    labels = model.fit_predict(activations)
    return labels


#import all mine.utils

#from mine.models.laters import ConcatLayer, CustomSequential

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

    @pl.data_loader
    def train_dataloader(self):
        if self.train_loader:
            return self.train_loader

        train_loader = torch.utils.data.DataLoader(
            FunctionDataset(self.kwargs['N'], self.x_dim,
                            self.kwargs['sigma'], self.kwargs['f']),
            batch_size=self.kwargs['batch_size'], shuffle=True)
        return train_loader

    @pl.data_loader
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
            opt = torch.optim.Adam(self.parameters(), le=1e-4)
           
        for iter in range(1, iters + 1):
            m_mi = 0
            for x, y in utils.batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()
                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
        final_mi = self.mi(X, Y)
        print(f"Final MI: {final_mi}")
        return final_mi
