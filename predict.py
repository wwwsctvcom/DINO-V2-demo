import torch
import numpy as np
from PIL import Image
from pathlib import Path
from utils.tools import test_transform, visualize_map
from model.model import Dinov2ForSemanticSegmentation

if __name__ == "__main__":
    # test image
    from train import Arguments
    args = Arguments()
    test_image = Image.open(Path(args.data_path) / "PreliminaryData/test/10000000.jpg")

    # process pixel values
    pixel_values = test_transform(image=np.array(test_image))["image"]
    pixel_values = torch.tensor(pixel_values)
    pixel_values = pixel_values.permute(2, 0, 1).unsqueeze(0)  # convert to (batch_size, num_channels, height, width)
    print(pixel_values.shape)

    # loading model
    from utils.tools import id2label

    model = Dinov2ForSemanticSegmentation.from_pretrained("dino_v2_finetuned",
                                                          id2label=id2label,
                                                          num_labels=len(id2label))

    # calculate outputs
    with torch.no_grad():
        outputs = model(pixel_values.to("cpu"))

    upsampled_logits = torch.nn.functional.interpolate(outputs.logits,
                                                       size=test_image.size[::-1],
                                                       mode="bilinear", align_corners=False)
    predicted_map = upsampled_logits.argmax(dim=1)

    visualize_map(test_image, predicted_map.squeeze().cpu())
