import torch
from accelerate import Accelerator
from loguru import logger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils.trainer import Trainer
from utils.dataset import SegmentationDataset, collate_fn
from utils.tools import seed_everything, train_transform
from model.model import Dinov2ForSemanticSegmentation



class Arguments:

    def __init__(self):
        # model name or path
        self.model_name_or_path = "facebook/dinov2-base"

        # train
        self.epochs = 10
        self.batch_size = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1e-4
        self.train_part = False

        # data
        self.data_path = "./data/PreliminaryData"  # used in kaggle: /kaggle/input/lanelines/PreliminaryData


if __name__ == "__main__":
    args = Arguments()

    # seed
    seed_everything()

    # loading model
    from utils.tools import id2label
    model = Dinov2ForSemanticSegmentation.from_pretrained(args.model_name_or_path,
                                                          id2label=id2label,
                                                          num_labels=len(id2label))

    if args.train_part:
        for name, param in model.named_parameters():
            if name.startswith("dinov2"):
                param.requires_grad = False

    # loading data
    from pathlib import Path
    train_pic_path = Path(args.data_path) / "train_pic"
    train_tag_path = Path(args.data_path) / "train_tag"

    train_dataset_lines = []
    for pic, tag in zip(sorted(train_pic_path.iterdir()), sorted(train_tag_path.iterdir())):
        data = {"image_path": pic, "label_path": tag}
        train_dataset_lines.append(data)
    logger.info("Data num: " + str(len(train_dataset_lines)))

    train_dataset = SegmentationDataset(train_dataset_lines[:6000], transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # load optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs,
                                                           eta_min=0,
                                                           last_epoch=-1,
                                                           verbose=False)
    accelerator = Accelerator()

    # start train
    trainer = Trainer(args=args,
                      model=model,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      accelerator=accelerator)
    trainer.train(train_data_loader=train_dataloader,
                  test_data_loader=None,
                  id2label=id2label)

    # save model
    trainer.save_model("dino_v2_finetuned")

