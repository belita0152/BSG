from pathlib import Path

import torch
from torch.nn import Module
from torch.optim import Optimizer, Adam
from torch.utils.data import Dataset, DataLoader

from byol.byol import BYOL
import torch.distributed as dist
from accelerate import Accelerator
from torch.nn import SyncBatchNorm


class BYOLTrainer(Module):
    def __init__(
        self,
        net: Module,
        *,
        image_size: int,
        hidden_layer: str,
        learning_rate: float,
        dataset: Dataset,
        num_train_steps: int,
        batch_size: int = 16,
        optimizer_klass = Adam,
        checkpoint_every: int = 1000,
        checkpoint_folder: str = './checkpoints',
        byol_kwargs: dict = dict(),
        optimizer_kwargs: dict = dict(),
        accelerator_kwargs: dict = dict(),
    ):
        super().__init__()


        self.accelerator = Accelerator(**accelerator_kwargs)

        if dist.is_initialized() and dist.get_world_size() > 1:
            net = SyncBatchNorm.convert_sync_batchnorm(net)

        self.net = net

        self.byol = BYOL(net, image_size = image_size, hidden_layer = hidden_layer, **byol_kwargs)

        self.optimizer = optimizer_klass(self.byol.parameters(), lr = learning_rate, **optimizer_kwargs)

        self.dataloader = DataLoader(dataset, shuffle = True, batch_size = batch_size)

        self.num_train_steps = num_train_steps

        self.checkpoint_every = checkpoint_every
        self.checkpoint_folder = Path(checkpoint_folder)
        self.checkpoint_folder.mkdir(exist_ok = True, parents = True)
        assert self.checkpoint_folder.is_dir()

        # prepare with accelerate

        (
            self.byol,
            self.optimizer,
            self.dataloader
        ) = self.accelerator.prepare(
            self.byol,
            self.optimizer,
            self.dataloader
        )

        self.register_buffer('step', torch.tensor(0))

    def wait(self):
        return self.accelerator.wait_for_everyone()

    def print(self, msg):
        return self.accelerator.print(msg)

    def forward(self):
        def cycle(dl):
            while True:
                for batch in dl:
                    yield batch

        step = self.step.item()
        data_it = cycle(self.dataloader)

        for _ in range(self.num_train_steps):
            images = next(data_it)

            with self.accelerator.autocast():
                loss = self.byol(images)
                self.accelerator.backward(loss)

            self.print(f'loss {loss.item():.3f}')

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.wait()

            self.byol.update_moving_average()

            self.wait()

            if not (step % self.checkpoint_every) and self.accelerator.is_main_process:
                checkpoint_num = step // self.checkpoint_every
                checkpoint_path = self.checkpoint_folder / f'checkpoint.{checkpoint_num}.pt'
                torch.save(self.net.state_dict(), str(checkpoint_path))

            self.wait()

            step += 1

        self.print('training complete')


if __name__ == '__main__':
    import torch
    from byol import BYOL
    from torchvision import models

    resnet = models.resnet50(pretrained=True)

    learner = BYOL(
        resnet,
        image_size=256,
        hidden_layer='avgpool'
    )

    opt = torch.optim.Adam(learner.parameters(), lr=3e-4)


    def sample_unlabelled_images():
        return torch.randn(20, 3, 256, 256)


    for _ in range(100):
        images = sample_unlabelled_images()
        loss = learner(images)
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()  # update moving average of target encoder

    # save your improved network
    # torch.save(resnet.state_dict(), './improved-net.pt')