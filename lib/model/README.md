# MODEL

No description

## Folder structure

```script
model                   - all model AI you created must be here.
|   abstract            - all abstract use in model must be here
|   |
|   └───README.md       - say about it self
|
|   resnet              - you create a model called "resnet" then you must save it in here
|   __init__.py         - this file make this folder as module
│   config.py           - file config for model
|   experiment.ipynb    - file to test own model
└───README.md           - description about this folder and its file. This folder structure check at here too
```

## ResNet workflow

Resnet stacket by manny blocks.
Each block have **residual** link from input to output.


