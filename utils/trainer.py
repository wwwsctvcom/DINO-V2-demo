import torch
from tqdm import tqdm
from pathlib import Path
from utils.tools import get_lr
from accelerate import Accelerator

class Trainer:

    def __init__(self,
                 args=None,
                 model=None,
                 optimizer=None,
                 scheduler=None,
                 accelerator=None,
                 ):
        # metric
        import evaluate
        self.metric = evaluate.load("mean_iou")

        self.args = args
        if self.args is None:
            raise ValueError("args is None!")

        # load model, optimizer
        self.model = model
        self.optimizer = optimizer
        if optimizer is None:
            raise ValueError("optimizer is None!")

        # load scheduler and accelerator
        self.scheduler = scheduler
        self.accelerator = accelerator if accelerator is not None else Accelerator()

    def train(self, train_data_loader=None, test_data_loader=None, id2label=None):
        if id2label is None:
            raise ValueError("id2label is None!")

        train_data_loader, test_data_loader, self.model, self.optimizer = self.accelerator.prepare(train_data_loader,
                                                                                                   test_data_loader,
                                                                                                   self.model,
                                                                                                   self.optimizer)

        for epoch in range(1, self.args.epochs + 1):
            train_total_loss = 0.0
            with tqdm(enumerate(train_data_loader), total=len(train_data_loader),
                      desc=f'Epoch: {epoch}/{self.args.epochs}', postfix=dict) as train_pbar:
                for step, batch in train_pbar:
                    pixel_values = batch["pixel_values"]
                    labels = batch["labels"]

                    # backward, calculate gradient
                    with self.accelerator.autocast():
                        # forward
                        outputs = self.model(pixel_values, labels=labels)
                        loss = outputs.loss
                        self.accelerator.backward(loss)
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                    # zero the gradient
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # lr scheduler
                    if self.scheduler is not None:
                        self.scheduler.step()

                    train_total_loss += self.accelerator.gather(loss).item()
                    train_pbar.set_postfix(**{"lr": get_lr(self.optimizer),
                                              "train average loss": train_total_loss / (step + 1),
                                              "train loss": loss.item()})

                    # evaluate
                    with torch.no_grad():
                        predicted = outputs.logits.argmax(dim=1)

                        # note that the metric expects predictions + labels as numpy arrays
                        self.metric.add_batch(predictions=predicted.detach().cpu().numpy(),
                                              references=labels.detach().cpu().numpy())

                    # let's print loss and metrics every 100 batches
                    mean_iou, mean_acc = 0.0, 0.0
                    if step % 30 == 0:
                        metrics = self.metric.compute(num_labels=len(id2label),
                                                      ignore_index=0,
                                                      reduce_labels=False, )
                        mean_iou = metrics["mean_iou"]
                        mean_acc = metrics["mean_accuracy"]

                    # update pbar
                    train_pbar.set_postfix(**{"lr": get_lr(self.optimizer),
                                              "train average loss": train_total_loss / (step + 1),
                                              "train loss": loss.item(),
                                              "mean_iou": mean_iou,
                                              "mean_acc": mean_acc})

    def save_model(self, out_dir: str = None):
        if not Path(out_dir).exists():
            Path(out_dir).mkdir()

        # unwrap model
        # self.accelerator.wait_for_everyone()
        # self.model = self.accelerator.unwrap_model(self.model)

        # save model
        self.model.save_pretrained(out_dir, torch_dtype=torch.float16)