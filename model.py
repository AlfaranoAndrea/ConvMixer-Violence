import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
import glob
import json

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )
    
class ConvMixerModule(pl.LightningModule):
    def __init__(self, checkpoint=None):
        super().__init__()
        torch.set_printoptions(threshold=100000)
        self.labels= []
        self.predicted= []
        self.json_parameter = json.load(open("config.json"))
        self.model = ConvMixer(
                        self.json_parameter['hdim'], 
                        self.json_parameter['depth'], 
                        patch_size=self.json_parameter['psize'], 
                        kernel_size=self.json_parameter['conv-ks'], 
                        n_classes=1000)
        self.lossFunction= nn.CrossEntropyLoss()

        model_path =glob.glob('./checkpoint_convmixer/*')
        model_load = torch.load(model_path[0])

        self.model.load_state_dict(model_load)
        self.model = nn.Sequential(*list(self.model.children())[:-3],
                nn.AdaptiveAvgPool2d((8,8)),
                nn.Flatten(),
                nn.Linear(self.json_parameter['hdim']*64, 101)
                )
        
    def adapt_for_n_classes(self, n):
        self.model = nn.Sequential(*list(self.model.children())[:-3],
                nn.AdaptiveAvgPool2d((8,8)),
                nn.Flatten(),
                nn.Linear(self.json_parameter['hdim']*64, n)
                )

                
    def forward (self, x):
        return self.model(x)
        

    def configure_optimizers(self):
        optimizer=optim.AdamW(self.parameters(), lr=self.json_parameter["lr"])
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler. MultiStepLR(optimizer, 
                        milestones=[2,4 ,8,16], # List of epoch indices
                        gamma =0.75),
            'name': 'my_logging_name'
        }
        
       
       
       
        
        return [optimizer], [lr_scheduler]

     
    
    def save(self, path= './models/best'):
        torch.save(self, path)

    def training_step(self, batch, batch_idx):
        x, y= batch    
      #  print("trainingtrainingtrainingtrainingtrainingtraining")
      #  print(f"y={y} y.shape={y.shape}")
        with torch.cuda.amp.autocast():
            y_predicted= self(x)
            loss = self.lossFunction(y_predicted, y)
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self,batch,batch_idx):
        x, y= batch    
        with torch.cuda.amp.autocast():
            y_predicted= self(x)
            loss = self.lossFunction(y_predicted, y)
        _, label_predicted = torch.max(y_predicted.data, 1)

        self.labels.append(y.cpu())
        self.predicted.append(label_predicted.cpu())
        self.log('val_loss', loss)
        return loss
    
    def validation_epoch_end(self, outputs):
        target= torch.cat(self.labels,0)
        pred= torch.cat(self.predicted,0)

       # comparison= torch.cat(
       #     (
       #     torch.reshape(target, (-1, 1)), 
       #     torch.reshape(pred, (-1, 1))
       #     ),
       # 1)

        print(classification_report(target, pred))
#        print(comparison)
        
        self.labels=[]
        self.predicted= []



    def test_step(self, batch, batch_idx):
        x, y= batch    
        with torch.cuda.amp.autocast():
            y_predicted= self(x)
            loss = self.lossFunction(y_predicted, y)
        _, label_predicted = torch.max(y_predicted.data, 1)
        self.labels.append(y.cpu())
        self.predicted.append(label_predicted.cpu())
        #print(classification_report(y.cpu(), label_predicted.cpu()))
        self.log('test_loss', loss)
        return loss

    def test_epoch_end(self, outputs):

        target= torch.cat(self.labels,0)
        pred= torch.cat(self.predicted,0)

        comparison= torch.cat(
            (
            torch.reshape(target, (-1, 1)), 
            torch.reshape(pred, (-1, 1))
            ),
        1)


        print(classification_report(target, pred))
        print(comparison)
        
        self.labels=[]
        self.predicted= []
