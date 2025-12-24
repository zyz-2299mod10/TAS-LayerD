import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from layerd.models.inpaint import build_inpaint
from layerd.models.matting import build_matting

from .helpers import (
    estimate_fg_alpha,
    estimate_fg_color,
    expand_mask,
    find_flat_color_region_ccs,
    refine_background,
    shrink_mask_ratio,
)

try:
    import easyocr
except ImportError:
    easyocr = None

from layerd.utils.split_words import segment_image
from layerd.utils.split_layerd import refine_layers
from layerd.utils.hisam_util import patchify_sliding, unpatchify_sliding
from hi_sam.modeling.build import model_registry
from hi_sam.modeling.predictor import SamPredictor
import warnings
warnings.filterwarnings("ignore")

class LayerD:
    def __init__(
        self,
        matting_hf_card: str = "cyberagent/layerd-birefnet",
        matting_process_size: tuple[int, int] | None = None,
        matting_weight_path: str | None = None,
        use_unblend: bool = True,
        bg_refine: bool = True,
        fg_refine: bool = True,
        fg_refine_num_colors: int = 2,
        bg_refine_num_colors: int = 10,
        kernel_scale: float = 0.015,
        extract_text: bool = False,
        ocr_lang_list: list[str] | None = None,
        device: str = "cpu",
    ) -> None:
        """Initialize LayerD model for image decomposition.

        Args:
            matting_hf_card: Hugging Face model card for the matting model.
            matting_process_size: Optional size (width, height) to resize images for matting processing.
            matting_weight_path: Optional path to local model weights. Overrides hugging face model if provided.
            use_unblend: Whether to use unblending technique for foreground color estimation.
            bg_refine: Whether to refine background with palette-based color assignment.
            fg_refine: Whether to refine foreground alpha and colors using flat color regions.
            fg_refine_num_colors: Number of colors for foreground refinement.
            bg_refine_num_colors: Number of colors for background refinement.
            kernel_scale: Scale factor to determine kernel size for mask expansion based on image dimensions.
            extract_text: Whether to extract text as the first layer using OCR.
            ocr_lang_list: List of languages for OCR, e.g., ["en", "ch_sim"]. Defaults to ["en"].
            device: Device to run models on ("cpu" or "cuda").
        """

        self.matting_model = build_matting(
            "birefnet",
            hf_card=matting_hf_card,
            process_image_size=matting_process_size,
            weight_path=matting_weight_path,
        )
        self.inpaint_model = build_inpaint("lama")
        self.use_unblend = use_unblend
        self.bg_refine = bg_refine
        self.fg_refine = fg_refine
        self.extract_text = extract_text
        if self.extract_text:
            if easyocr is None:
                raise ImportError("easyocr is not installed. Please install it with 'pip install easyocr'")
            if ocr_lang_list is None:
                ocr_lang_list = ["en"]
            self.ocr_reader = easyocr.Reader(ocr_lang_list)

            class HiSAMArgs:
                def __init__(self, checkpoint, model_type, attn_layers=1, prompt_len=12):
                    self.checkpoint = checkpoint
                    self.model_type = model_type
                    self.attn_layers = attn_layers
                    self.prompt_len = prompt_len
                    self.hier_det = True  # Need hi_decoder for text detection

            hi_sam_ckpt = './hi_sam/pretrained_checkpoint/efficient_hi_sam_s.pth'
            model_type = 'vit_s'
            hisam_args = HiSAMArgs(hi_sam_ckpt, model_type)
            hisam = model_registry[model_type](hisam_args)
            hisam.eval()
            hisam.to(device)
            self.text_predictor = SamPredictor(hisam)

        # Parameters for refinement
        self.fg_refine_num_colors = fg_refine_num_colors
        self.bg_refine_num_colors = bg_refine_num_colors
        self._kernel_scale = kernel_scale
        self._th_alpha = 0.005  # threshold for hard alpha mask
        self._unblend_alpha_clip = [0, 0.95]  # clipping range for unblending
        self._palette_percentile = 0.99  # percentile for palette color selection in both fg and bg refinement
        self._bg_refine_n_outer_ratio = 0.2  # ratio for outer region to determine bg flatness
        self._fg_refine_n_inner_ratio = 0.1  # ratio for inner region to be refined
        self.to(device)

    def _calc_kernel_size(self, image: Image.Image) -> tuple[int, int]:
        kernel_size = (round(image.height * self._kernel_scale), round(image.width * self._kernel_scale))
        return kernel_size
    
    def _get_hisam_config(self, box_height: int, box_width: int) -> tuple[int, int]:
        if box_height < 30 or box_width < 30:
            return 512, 192
        else:
            return 1024, 384

    def _extract_text_layer(self, image: Image.Image) -> tuple[Image.Image | None, Image.Image]:
        image_rgb = np.array(image.convert("RGB"))
        h, w = image_rgb.shape[:2]
        kernel_size = self._calc_kernel_size(image)

        # Detect text using easyocr
        text_detections = self.ocr_reader.readtext(image_rgb)
        if not text_detections:
            return None, image, None, None
        
        full_text_mask = np.zeros((h, w), dtype=np.float32)
        for bbox, text, score in tqdm(text_detections):
            # bbox: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            pts = np.array(bbox, dtype=np.int32)
            x_min = np.min(pts[:, 0])
            x_max = np.max(pts[:, 0])
            y_min = np.min(pts[:, 1])
            y_max = np.max(pts[:, 1])

            box_h = y_max - y_min
            box_w = x_max - x_min

            # ROI with padding to avoid boundary effects
            patch_size = 512
            padding = patch_size // 4
            roi_x1 = max(0, int(x_min - padding))
            roi_y1 = max(0, int(y_min - padding))
            roi_x2 = min(w, int(x_max + padding))
            roi_y2 = min(h, int(y_max + padding))

            roi_h = roi_y2 - roi_y1
            roi_w = roi_x2 - roi_x1
            
            if roi_h <= 0 or roi_w <= 0:
                continue

            roi_img = image_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
            
            roi_mask_accum = []
            if box_h < 30 or box_w < 30:
                patches, h_slice_list, w_slice_list = patchify_sliding(roi_img, patch_size=512, stride=384)

                for patch in patches:
                    self.text_predictor.set_image(patch)
                    m, hr_m, score, hr_score = self.text_predictor.predict(multimask_output=False, return_logits=True)
                    
                    if isinstance(hr_m, np.ndarray):
                        if len(hr_m.shape) == 3:
                            mask_patch = hr_m[0]
                        else:
                            mask_patch = hr_m
                    else:
                        mask_patch = hr_m[0].cpu().numpy()
                    
                    roi_mask_accum.append(mask_patch)       

                roi_mask_logits = unpatchify_sliding(roi_mask_accum, h_slice_list, w_slice_list, (roi_h, roi_w))
                roi_binary_mask = (roi_mask_logits > self.text_predictor.model.mask_threshold).astype(np.float64)
            else:
                self.text_predictor.set_image(roi_img)
                mask, hr_mask, score, hr_score = self.text_predictor.predict(multimask_output=False)
                roi_binary_mask = hr_mask.astype(np.float64)

            current_roi_area = full_text_mask[roi_y1:roi_y2, roi_x1:roi_x2]
            full_text_mask[roi_y1:roi_y2, roi_x1:roi_x2] = np.maximum(current_roi_area, roi_binary_mask)

        text_alpha_mask = full_text_mask > 0.5
        if not text_alpha_mask.any():
            return None, image, None, None

        text_alpha = text_alpha_mask.astype(np.float64)

        # Create inpaint mask and get inpainted background
        inpaint_mask = expand_mask(text_alpha_mask, kernel_size=kernel_size)
        bg = self.inpaint_model(image_rgb, inpaint_mask)
        if self.bg_refine:
            bg = refine_background(
                bg, inpaint_mask, max_num_colors=self.bg_refine_num_colors, n_outer_ratio=self._bg_refine_n_outer_ratio
            )
        background = Image.fromarray(bg)

        # Create foreground layer with precise text contours
        if self.use_unblend:
            fg_rgb = estimate_fg_color(image_rgb, bg, text_alpha, self._unblend_alpha_clip)
        else:
            fg_rgb = image_rgb.copy()

        alpha_uint8 = (text_alpha * 255).astype(np.uint8)
        foreground = Image.fromarray(np.dstack([fg_rgb, alpha_uint8])).convert("RGBA")
        bboxes = [item[0] for item in text_detections]

        return foreground, background, text_alpha_mask, bboxes

    def _decompose_step(self, image: Image.Image) -> tuple[Image.Image | None, Image.Image]:
        image_rgb = np.array(image.convert("RGB"))
        kernel_size = self._calc_kernel_size(image)

        alpha = self.matting_model(image)
        hard_mask = alpha > self._th_alpha
        if hard_mask.sum() == 0:  # No content
            return None, image
        if np.mean(hard_mask) > 0.99:  # Full content
            return None, image

        if self.fg_refine:
            color_masks, colors, ccs = find_flat_color_region_ccs(
                image_rgb, hard_mask, max_num_colors=self.fg_refine_num_colors, percentile=self._palette_percentile
            )
            # Shrink connected components to be refined
            shrinked_ccs = [
                ccs[i] if len(colors[i]) == 0 else shrink_mask_ratio(ccs[i], self._fg_refine_n_inner_ratio)
                for i in range(len(ccs))
            ]
            inpaint_mask = expand_mask(np.any(shrinked_ccs + sum(color_masks, []), axis=0), kernel_size)
        else:
            inpaint_mask = expand_mask(hard_mask, kernel_size=kernel_size)

        bg = self.inpaint_model(image_rgb, inpaint_mask)

        if self.bg_refine:
            bg = refine_background(
                bg, inpaint_mask, max_num_colors=self.bg_refine_num_colors, n_outer_ratio=self._bg_refine_n_outer_ratio
            )

        if self.use_unblend:
            fg_rgb = estimate_fg_color(image_rgb, bg, alpha, self._unblend_alpha_clip)
        else:
            fg_rgb = image_rgb.copy()

        if self.fg_refine:
            for colors_cc, color_masks_cc, cc in zip(colors, color_masks, ccs):
                _refined_alpha = np.zeros_like(alpha)
                _refined_color = np.zeros_like(fg_rgb)
                _nonzero_mask_counts = np.zeros_like(alpha)
                for color, color_mask in zip(colors_cc, color_masks_cc):
                    color_mask_expanded = expand_mask(color_mask, kernel_size)
                    _refined_alpha_color = estimate_fg_alpha(color_mask_expanded, color, bg, image_rgb)
                    if _refined_alpha_color is not None:
                        _refined_alpha = np.maximum(_refined_alpha, _refined_alpha_color)
                        _refined_color[_refined_alpha_color > 0] = color
                        _nonzero_mask_counts += (_refined_alpha_color > 0).astype(int)
                color_boundary_mask = _nonzero_mask_counts > 1
                if _refined_alpha.sum() > 0:
                    inner_cc = (~shrink_mask_ratio(cc, self._fg_refine_n_inner_ratio)) & cc
                    target_mask = ((alpha == 0) | inner_cc) & (~color_boundary_mask)
                    alpha[target_mask] = np.maximum(alpha[target_mask], _refined_alpha[target_mask])
                    fg_rgb[target_mask & (_refined_alpha > 0)] = _refined_color[target_mask & (_refined_alpha > 0)]

        background = Image.fromarray(bg)
        foreground = Image.fromarray(np.dstack([fg_rgb, np.array(alpha * 255, dtype=np.uint8)])).convert("RGBA")

        return foreground, background

    def decompose(self, image: Image.Image, max_iterations: int = 3) -> list[Image.Image]:
        """Decompose an image into layers of foregrounds and backgrounds.
        Args:
            image: Input PIL Image to decompose.
            max_iterations: Maximum number of decomposition iterations.
        Returns:
            List of PIL Images representing the layers, starting with the final background.
        """

        bg_list = []
        fg_list = []
        current_bg = image.convert("RGB")

        if self.extract_text:
            text_layer, current_bg, text_alpha, bboxes = self._extract_text_layer(image)
            if text_layer is not None:
                text_layers, _, _ = segment_image(np.array(text_layer), bboxes, kernel_w=80, kernel_h=10)

                for t in text_layers:
                    layer_pil = Image.fromarray(t, 'RGBA')
                    fg_list.append(layer_pil)
            else:
                print("Can't found any TEXT")

        for _ in range(max_iterations):
            fg, new_bg = self._decompose_step(current_bg)
            if fg is None:
                break

            bg_list.append(new_bg)
            refined_layers = refine_layers(
                [fg], 
                min_area = 100,
                easy_threshold = 20,
                max_clusters = 8,
            )
            fg_list.extend(refined_layers)
            # fg_list.append(fg)
            current_bg = new_bg

        # if nothing was decomposed at all
        if not fg_list and text_layer is None:
            return [image.convert("RGBA")]

        if bg_list:
            final_bg = bg_list[-1].convert("RGBA")
        else:
            final_bg = current_bg.convert("RGBA")

        # layer order: [final_bg, layer1, layer2, ..., text_layer(top)]
        final_layers = [final_bg] + fg_list[::-1]

        return final_layers

    def to(self, device: str) -> "LayerD":
        self.matting_model.to(device)
        self.inpaint_model.to(device)
        return self
