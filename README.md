# vision-conformer

Implemention of Vision Conformer, a vision classification model that conbines Vision Transformer and CNN.  
![vic](https://user-images.githubusercontent.com/54788782/158154662-c13b9e98-ea4a-4f8d-b0ee-178c92ef8a33.png)  

The contents are under developing.  
If you have any questions, please touch me. akihiro.kusuda@human.ait.kyushu-u.ac.jp  

## Usage

You can create a conda environment with envirionment.yaml.  

```sh
conda env create -f envirion.yaml
conda activate vision-conformer
```

You can train the model by running main.py.

```sh
python main.py
```

You can also check the results on mlflow dashboard.

```sh
mlflow ui
```

## Parameters

In main.py, you can change the parameters for this model.  

```py
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
```