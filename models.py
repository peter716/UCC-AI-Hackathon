import torch
import torch.nn as nn
import torchvision
import lightning.pytorch as pl
from torchmetrics.classification import Dice
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from segmentation_models_pytorch import SegFormer
from metrics import SMAPIoUMetric
from unet import UNET

class SegModel(pl.LightningModule):
    def __init__(self, learning_rate = 1e-3):
        super(SegModel, self).__init__()
        self.learning_rate = learning_rate
        self.num_classes = 2

        # self.net = torchvision.models.segmentation.fcn_resnet50(num_classes = self.num_classes)
        self.net = torchvision.models.segmentation.deeplabv3_resnet101(num_classes = self.num_classes)
        # self.net = torchvision.models.segmentation.fcn_resnet50(pretrained = True,num_classes = 21)

        # self.weights = torchvision.models.segmentation.FCN_ResNet50_Weights.DEFAULT
        # self.net = torchvision.models.segmentation.fcn_resnet50(weights = self.weights)

        # self.net.classifier[4] = torch.nn.Conv2d(512, self.num_classes, kernel_size=(1, 1))

        # self.net = UNET(in_channels = 3, out_channels = 1)
        # self.net = torchvision.models.segmentation.deeplabv3_resnet50(num_classes = self.num_classes)
        # self.criterion = nn.BCEWithLogitsLoss()
        self.criterion = nn.CrossEntropyLoss()
        # self.criterion = DiceBCELoss()
        self.evaluator = SMAPIoUMetric()

    def forward(self, x):
        # transformed_x = self.transform_batch(x)
        # return self.net(transformed_x)
        return self.net(x)

    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)["out"]
        loss = self.criterion(out, mask)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    
    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()

        # print("image shape")
        # print(img.shape)
        mask = mask.long()
        # mask = mask.unsqueeze(1)
        out = self.forward(img)["out"]
        # out = self.forward(img)
        # loss = self.criterion(out, mask.float())
        loss = self.criterion(out, mask)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self.forward(img)["out"]
        loss = self.criterion(out, mask)

        probs = torch.softmax(out, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds = preds.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        
        self.evaluator.process(input={"pred": preds, "gt": mask})

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self) -> None:
        metrics = self.evaluator.evaluate(0)
        self.log(
            f"val_high_vegetation_IoU",
            metrics["high_vegetation__IoU"],
            sync_dist=True,
        )
        self.log(f"val_mIoU", metrics["mIoU"], sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        # optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4)     
        return  {"optimizer": optimizer,
                 "lr_scheduler" : {
                  "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, min_lr=1e-6, verbose = True),
                  # "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0),
                  # "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1),
                  "monitor": "val_loss",
                  "frequency": 1
                 } }

        # return optimizer


