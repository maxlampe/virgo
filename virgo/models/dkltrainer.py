""""""

import gpytorch
import torch
import tqdm
from torch.optim.lr_scheduler import MultiStepLR


class DKLTrainer:
    """"""

    def __init__(
        self,
        model,
        likelihood,
        train_loader,
        val_loader,
        epochs: int = 20,
        lr: float = 0.1,
        optimizer=None,
        scheduler=None,
        mll=None,
    ):
        self._model = model
        self._likelihood = likelihood
        self.train_loader = train_loader
        self.val_loader = val_loader
        self._epochs = epochs
        self._lr = lr

        if optimizer is None:
            self._optimizer = torch.optim.Adam(
                [
                    {
                        "params": self._model.feature_extractor.parameters(),
                        "weight_decay": 1e-4,
                    },
                    {
                        "params": self._model.gp_layer.hyperparameters(),
                        "lr": self._lr * 0.01,
                    },
                    {"params": self._model.gp_layer.variational_parameters()},
                    {"params": self._likelihood.parameters()},
                ],
                lr=self._lr,
            )
        else:
            self._optimizer = optimizer

        if mll is None:
            self._mll = gpytorch.mlls.VariationalELBO(
                self._likelihood,
                self._model.gp_layer,
                num_data=len(self.train_loader.dataset),
            )
        else:
            self._mll = mll

        if scheduler is None:
            self._scheduler = MultiStepLR(
                self._optimizer,
                milestones=[0.5 * epochs, 0.75 * epochs],
                gamma=0.1,
            )
        else:
            self._scheduler = scheduler

    def train(self):
        """"""

        for epoch in range(1, self._epochs + 1):
            with gpytorch.settings.use_toeplitz(False):
                self._train(epoch)
                self._validate()
            self._scheduler.step()
            # state_dict = model.state_dict()
            # likelihood_state_dict = likelihood.state_dict()
            # torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')

    def _train(self, epoch: int):
        self._model.train()
        self._likelihood.train()

        minibatch_iter = tqdm.notebook.tqdm(
            self.train_loader,
            desc=f"(Epoch {epoch}) Minibatch",
        )
        with gpytorch.settings.num_likelihood_samples(8):
            for data, target in minibatch_iter:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                self._optimizer.zero_grad()
                output = self._model(data)
                loss = -self._mll(output, target)
                loss.backward()
                self._optimizer.step()
                minibatch_iter.set_postfix(loss=loss.item())

    def _validate(self):
        self._model.eval()
        self._likelihood.eval()

        correct = 0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
            for data, target in self.val_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                # This gives us 16 samples from the predictive distribution
                output = self._likelihood(self._model(data))
                # Taking the mean over all of the sample we've drawn
                pred = output.probs.mean(0).argmax(-1)
                correct += pred.eq(target.view_as(pred)).cpu().sum()
        print(
            "Test set: Accuracy: {}/{} ({}%)".format(
                correct,
                len(self.val_loader.dataset),
                100.0 * correct / float(len(self.val_loader.dataset)),
            )
        )
