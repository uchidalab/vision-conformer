import pytorch_lightning as pl
from utils.models import LitModel, VisionConformer
from utils.datamodules import FashionMNISTDataModule
import mlflow.pytorch
from mlflow.tracking import MlflowClient


# Data
print('==> Preparing data..')
fashionmnist_dm = FashionMNISTDataModule(batch_size=256)

# Model
print('==> Building model..')
model = VisionConformer(
    image_size=28,
    patch_size=4,
    num_classes=10,
    dim=256,
    depth=3,
    heads=4,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1,
    pool='cls',
    channels=1,
    hidden_channels=32,
    cnn_depth=2,
)
model = LitModel(model)

# Training
num_epochs = 500
trainer = pl.Trainer(
    progress_bar_refresh_rate=1,
    max_epochs=num_epochs,
    gpus=1,
    logger=pl.loggers.TensorBoardLogger('./lightning_logs/', name='VisionConformer'),
)
print(f'==> Train the model for {num_epochs} epochs..')
mlflow.pytorch.autolog()
with mlflow.start_run() as run:
    trainer.fit(model, datamodule=fashionmnist_dm)
    trainer.test(model, datamodule=fashionmnist_dm)
