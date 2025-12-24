import argparse
import logging
import os
import os.path as osp
from glob import glob

from PIL import Image
from tqdm import tqdm

from layerd.models.layerd import LayerD
from layerd.utils.log import setup_logging

logger = logging.getLogger(__name__)


def infer_dataset(args: argparse.Namespace) -> None:
    """Run decomposition on a dataset of images."""

    # Model setup
    layerd = LayerD(
        matting_hf_card=args.matting_hf_card,
        matting_process_size=tuple(args.matting_process_size) if args.matting_process_size else None,
        matting_weight_path=args.matting_weight_path,
        kernel_scale=args.kernel_scale,
        extract_text=args.extract_text,
        ocr_lang_list=args.ocr_lang_list if args.ocr_lang_list else None,
    )
    layerd.to(args.device)

    # Handle different input types: file, directory, or glob pattern
    if len(args.input) == 1:
        if osp.isfile(args.input[0]):
            paths = args.input
        elif osp.isdir(args.input[0]):
            paths = sorted(glob(osp.join(args.input[0], "*")))
            paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))]
    else:
        paths = args.input

    if not paths:
        raise ValueError(f"No files found matching pattern: {args.input}")

    paths = paths[args.start_index : args.end_index]
    os.makedirs(args.output_dir, exist_ok=True)
    for i, path in enumerate(paths):
        logger.info(f"Processing {path} ({i + 1}/{len(paths)})")

        fn = osp.basename(path).split(".")[0]
        image_pil = Image.open(path)

        layers = layerd.decompose(image_pil, max_iterations=args.max_iterations)

        # save_dir = osp.join(args.output_dir, fn)
        # os.makedirs(save_dir, exist_ok=True)

        for i, layer in enumerate(layers):
            # layer.save(osp.join(save_dir, f"{i:04d}.png"))
            layer.save(osp.join(args.output_dir, f"{fn}_{i}.png"))

    logger.info("Inference completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input/output
    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Input image file(s), directory, or glob pattern. Can specify multiple files.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for processing images")
    parser.add_argument("--end-index", type=int, default=None, help="End index for processing images")
    # Model
    parser.add_argument(
        "--matting-hf-card",
        type=str,
        default="cyberagent/layerd-birefnet",
        help="HuggingFace model card for matting model",
    )
    parser.add_argument("--matting-weight-path", type=str, default=None, help="Path to matting model weights")
    parser.add_argument(
        "--matting-process-size",
        type=int,
        nargs=2,
        default=None,
        help="Process size for matting model. If not set, use model's default size in huggingface config.",
    )
    # Inference
    parser.add_argument("--max-iterations", type=int, default=7, help="Maximum iterations for layer decomposition")
    parser.add_argument("--kernel-scale", type=float, default=0.015, help="Kernel scale for mask expansion/shrinkage")
    # Text extraction (OCR)
    parser.add_argument("--extract-text", action="store_true", help="Extract text as the first layer using OCR")
    parser.add_argument(
        "--ocr-lang-list",
        type=str,
        nargs="+",
        default=None,
        help='List of languages for OCR, e.g., "en" "ch_sim". Defaults to ["en"] if --extract-text is enabled.',
    )
    # Others
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the models on")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    args = parser.parse_args()

    setup_logging(level=args.log_level, use_tqdm_handler=True)
    infer_dataset(args)
