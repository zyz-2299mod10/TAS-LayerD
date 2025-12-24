import argparse
import datasets
from PIL import Image
from typing import List, Dict

from tools.utils.crello_utils import prepare_layers_from_crello_sample
from tools.evaluator import evaluate_layer_decomposition
from tools.render import render_layers


def eval(
    gt_layers: List[Image.Image],
    pred_layers: List[Image.Image],
    input_image: Image.Image,
) -> Dict[str, float]:
    """
    Evaluate the performance of the model.
    """
    return evaluate_layer_decomposition(pred_layers, gt_layers, input_image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate the output with Crello dataset")
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()

    dataset = datasets.load_dataset("cyberagent/crello", revision="5.0.0")
    test_dataset = dataset["test"]

    idx = args.sample_index
    sample = test_dataset[idx]
    gt_layers = prepare_layers_from_crello_sample(sample)
    rendered_image = render_layers(gt_layers)
    pred_layers = gt_layers
    result = eval(gt_layers, pred_layers, rendered_image)
    print(result)
