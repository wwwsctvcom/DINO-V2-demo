import torch
from transformers.modeling_outputs import SemanticSegmenterOutput
from transformers import Dinov2Model, Dinov2PreTrainedModel

class LinearClassifier(torch.nn.Module):
    def __init__(self, in_channels, tokenW=32, tokenH=32, num_labels=1):
        super(LinearClassifier, self).__init__()

        self.in_channels = in_channels
        self.width = tokenW
        self.height = tokenH
        self.classifier = torch.nn.Conv2d(in_channels, num_labels, (1, 1))

    def forward(self, embeddings):
        embeddings = embeddings.reshape(-1, self.height, self.width, self.in_channels)  # [batch, 1024, 768]
        embeddings = embeddings.permute(0, 3, 1, 2)  # [batch, in_channels, height, width] == [batch, 768, 32, 32]

        return self.classifier(embeddings)  # [batch, num_labels, height, width]


class Dinov2ForSemanticSegmentation(Dinov2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.dinov2 = Dinov2Model(config)
        self.classifier = LinearClassifier(config.hidden_size, 32, 32, config.num_labels)

    def forward(self, pixel_values, output_hidden_states=False, output_attentions=False, labels=None):
        # use frozen features
        outputs = self.dinov2(pixel_values,
                              output_hidden_states=output_hidden_states,
                              output_attentions=output_attentions)
        # get the patch embeddings - so we exclude the CLS token
        patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # [batch, 1024, 768]

        # convert to logits and upsample to the size of the pixel values
        logits = self.classifier(patch_embeddings)  # [batch, num_labels, height, width]
        logits = torch.nn.functional.interpolate(logits, size=pixel_values.shape[2:], mode="bilinear",
                                                 align_corners=False)  # [batch, num_labels, 448, 448]
        loss = None
        if labels is not None:
            # important: we're going to use 0 here as ignore index instead of the default -100
            # as we don't want the model to learn to predict background
            loss_fct = torch.nn.CrossEntropyLoss()
            #         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0)
            loss = loss_fct(logits.squeeze(), labels.squeeze())

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
