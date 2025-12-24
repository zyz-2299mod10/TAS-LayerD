import argparse
import logging
import os
import os.path as osp
import io
import math

import numpy as np
import datasets
from PIL import Image, ImageEnhance
from tqdm import tqdm

from layerd.data.crello import make_top_layers_alpha_and_image_pair
try:
    from layerd.models.inpaint import LamaInpaint
except ImportError:
    LamaInpaint = None

from layerd.utils.log import setup_logging

logger = logging.getLogger(__name__)

class PicCollagePILRenderer:
    def __init__(self, features=None):
        self.features = features

    def get_layer_types(self, example) -> tuple:
        types = example.get("type", [])
        return tuple(types)

    def get_is_transparent(self, example) -> tuple:
        images = example["image"]
        return tuple(self._check_transparency(img) for img in images)

    def _check_transparency(self, image: Image.Image) -> bool:
        if image.mode != "RGBA":
            return False
        alpha = image.split()[-1]
        return not alpha.getextrema()[1] == 255

    def _process_layer(self, img, left, top, width, height, angle, canvas_size):
        target_w, target_h = int(width), int(height)
        
        if target_w <= 0 or target_h <= 0:
            return Image.new("RGBA", canvas_size, (0, 0, 0, 0))
            
        img_resized = img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        img_rgba = img_resized.convert("RGBA")
        
        cx, cy = left + width / 2, top + height / 2
        
        rotated_img = img_rgba.rotate(angle, resample=Image.BICUBIC, expand=True)
        
        rw, rh = rotated_img.size
        paste_x = int(cx - rw / 2)
        paste_y = int(cy - rh / 2)

        layer_canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
        layer_canvas.paste(rotated_img, (paste_x, paste_y), rotated_img)
        
        return layer_canvas

    def render_layers(self, example, short_side_size=360, render_text=False):
        raw_canvas_w = example["canvas_width"]
        raw_canvas_h = example["canvas_height"]
        
        scale = short_side_size / min(raw_canvas_w, raw_canvas_h)
        final_w = int(raw_canvas_w * scale)
        final_h = int(raw_canvas_h * scale)
        
        layers_bytes = []
        
        num_layers = len(example["image"])
        
        for i in range(num_layers):
            img = example["image"][i]
            
            left = example["left"][i] * scale
            top = example["top"][i] * scale
            width = example["width"][i] * scale
            height = example["height"][i] * scale
            angle = example["angle"][i] 
            layer_pil = self._process_layer(img, left, top, width, height, angle, (final_w, final_h))
            
            with io.BytesIO() as bio:
                layer_pil.save(bio, format="PNG")
                layers_bytes.append(bio.getvalue())
                
        return layers_bytes

    def render(self, example, short_side_size=360, layer_indices=None, render_text=False):
        raw_canvas_w = example["canvas_width"]
        raw_canvas_h = example["canvas_height"]
        
        scale = short_side_size / min(raw_canvas_w, raw_canvas_h)
        final_w = int(raw_canvas_w * scale)
        final_h = int(raw_canvas_h * scale)
        
        composite = Image.new("RGBA", (final_w, final_h), (0, 0, 0, 0))
        
        if layer_indices is None:
            layer_indices = range(len(example["image"]))
            
        for i in layer_indices:
            img = example["image"][i]
            left = example["left"][i] * scale
            top = example["top"][i] * scale
            width = example["width"][i] * scale
            height = example["height"][i] * scale
            angle = example["angle"][i]
            
            layer_pil = self._process_layer(img, left, top, width, height, angle, (final_w, final_h))
            
            composite.alpha_composite(layer_pil)
            
        with io.BytesIO() as bio:
            composite.save(bio, format="PNG")
            return bio.getvalue()


def generate_crello_matting(args: argparse.Namespace) -> None:
    dataset_name = "WalkerHsu/DLCV2025_final_project_piccollage"
    
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = datasets.load_dataset(dataset_name, cache_dir=args.hf_cache_dir)
    renderer = PicCollagePILRenderer(features=dataset["train"].features)
    
    if args.inpainting and LamaInpaint:
        inpaint_model = LamaInpaint(device=args.device) 
    else:
        inpaint_model = None

    available_splits = [s for s in args.splits if s in dataset.keys()]
    if not available_splits:
        logger.warning(f"Requested splits {args.splits} not found. Using available: {list(dataset.keys())}")
        available_splits = list(dataset.keys())

    for split in available_splits:
        os.makedirs(osp.join(args.output_dir, split, "im"), exist_ok=True)
        os.makedirs(osp.join(args.output_dir, split, "gt"), exist_ok=True)

        start_i = 0 if args.start_index is None else max(args.start_index, 0)
        end_i = len(dataset[split]) if args.end_index is None else min(args.end_index, len(dataset[split]))
        ds = dataset[split].select(range(start_i, end_i))
        n_ds = len(ds) if args.num_samples < 0 else min(len(ds), args.num_samples)

        for idx, example in enumerate(tqdm(ds, ncols=0, desc=f"Generating dataset ({split})", total=n_ds)):
            sample_i = idx + start_i
            
            # make sure example contain all we want
            if "image" not in example or "left" not in example:
                logger.warning(f"Skipping {example.get('id', idx)} due to missing fields.")
                continue

            try:
                pairs, inpainted_pairs, layers_pil = make_top_layers_alpha_and_image_pair(
                    example,
                    renderer,
                    short_side_size=args.short_side_size,
                    exclude_text=args.exclude_text,
                    exclude_transparent=True, 
                    inpaint_model=inpaint_model,
                )
            except Exception as e:
                logger.error(f"Error processing sample {example.get('id', idx)}: {e}")
                continue

            # Save training pairs
            for j, (fg_mask, image) in enumerate(pairs):
                fg_mask_pil = Image.fromarray(fg_mask)
                fg_mask_pil.save(
                    osp.join(args.output_dir, split, "gt", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}.png")
                )
                image.save(
                    osp.join(args.output_dir, split, "im", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}.png")
                )
            
            for j, pair in enumerate(inpainted_pairs):
                if pair is None:
                    continue
                fg_mask, image = pair
                fg_mask_pil = Image.fromarray(fg_mask)
                fg_mask_pil.save(
                    osp.join(
                        args.output_dir, split, "gt", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}_inpainted.png"
                    )
                )
                image.save(
                    osp.join(
                        args.output_dir, split, "im", f"{split}_{sample_i:06d}_{example['id']}_{j:02d}_inpainted.png"
                    )
                )
            
            if args.save_layers and len(layers_pil) > 0:
                os.makedirs(osp.join(args.output_dir, split, "composite"), exist_ok=True)
                if pairs:
                    composite_image = pairs[0][1]
                    composite_image.save(
                        osp.join(args.output_dir, split, "composite", f"{split}_{sample_i:06d}_{example['id']}.png")
                    )
                
                out_dir = osp.join(args.output_dir, split, "layers", f"{split}_{sample_i:06d}_{example['id']}")
                os.makedirs(osp.join(out_dir), exist_ok=True)
                for j, layer_pil in enumerate(layers_pil):
                    layer_pil.save(osp.join(out_dir, f"{split}_{sample_i:06d}_{example['id']}_{j:02d}.png"))

            if (idx + 1) == args.num_samples:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Huggingface cache dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save results")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "validation", "test"], help="Dataset splits")
    parser.add_argument("--short-side-size", type=int, default=1024, help="Short side size for rendering")
    parser.add_argument("--exclude-text", action="store_true", help="Exclude text layers")
    parser.add_argument("--inpainting", action="store_true", help="Use inpainting to fill occluded regions")
    parser.add_argument("--save-layers", action="store_true", help="Save filtered layers and composite images")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the inpainting model on")
    parser.add_argument("--num-samples", type=int, default=-1, help="Number of samples to process per split")
    parser.add_argument("--start-index", type=int, default=None, help="Start index to process")
    parser.add_argument("--end-index", type=int, default=None, help="End index to process")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--revision", type=str, default=None, help="Dataset version/revision")
    args = parser.parse_args()

    setup_logging(level=args.log_level, use_tqdm_handler=True)
    logger.info(f"Arguments: {args}")
    generate_crello_matting(args)