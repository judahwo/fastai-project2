from fastai.vision.all import *

path = Path('concrete-dataset')

# DataBlock is a high-level API that allows you to create a DataLoaders object from a source of data
datablock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(460),
    batch_tfms=aug_transforms(size=224, min_scale=0.75)
)

dls = datablock.dataloaders(path)

# cnn_learner is a high-level API that allows you to create a Learner object from a DataLoaders object
learn = cnn_learner(dls, resnet18, metrics=accuracy)
learn.fit_one_cycle(4)
learn.show_results()
learn.export('concrete_classifier.pkl')