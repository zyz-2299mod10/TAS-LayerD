import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def _compute_component_features(layers: List[Image.Image]):

    feats = []
    areas = []

    if len(layers) == 0:
        return np.zeros((0, 6), dtype=np.float32), []

    h, w = np.array(layers[0]).shape[:2]

    for layer in layers:
        arr = np.array(layer)
        alpha = arr[:, :, 3]
        mask = alpha > 0

        area = int(mask.sum())
        if area == 0:
            feats.append([0, 0, 0, 0.5, 0.5, 0.0])
            areas.append(0)
            continue

        ys, xs = np.nonzero(mask)
        cx = float(xs.mean()) / max(1, w)   # 0~1
        cy = float(ys.mean()) / max(1, h)   # 0~1

        rgb = arr[mask][:, :3].astype(np.float32).mean(axis=0) / 255.0  # 0~1
        log_area = np.log(area + 1.0)

        feats.append([rgb[0], rgb[1], rgb[2], cx, cy, log_area])
        areas.append(area)

    return np.asarray(feats, dtype=np.float32), areas


def group_components_by_kmeans_refined(
    cc_layers: List[Image.Image],
    easy_threshold: int = 20,
    max_clusters: int = 8,
    big_ratio: float = 0.3,
    size_ratio_threshold: float = 2.0,
) -> List[Image.Image]:
    
    n = len(cc_layers)
    if n == 0:
        return []
    if n == 1:
        return cc_layers
    if n <= easy_threshold:
        return cc_layers

    feats, areas = _compute_component_features(cc_layers)

    areas = np.asarray(areas)
    max_area = areas.max()
    if max_area <= 0:
        return cc_layers
    min_area = areas[areas > 0].min()

    if max_area <= 0 or max_area / max(1, min_area) < size_ratio_threshold:
        small_idx = [i for i, a in enumerate(areas) if a > 0]
        big_idx = []
    else:
        big_thresh = big_ratio * float(max_area)
        big_idx   = [i for i, a in enumerate(areas) if a >= big_thresh]
        small_idx = [i for i, a in enumerate(areas) if 0 < a < big_thresh]

    big_layers = [cc_layers[i] for i in big_idx]

    if len(small_idx) == 0:
        return big_layers

    X = feats[small_idx]

    max_k = min(max_clusters, len(small_idx))
    if max_k <= 1:
        small_groups = [small_idx]
    else:
        best_k = 2
        best_inertia = None
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertia = km.inertia_ / len(X)
            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia = inertia
                best_k = k

        km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        small_groups = []
        for lab in sorted(set(labels)):
            idx_group = [small_idx[i] for i, l in enumerate(labels) if l == lab]
            small_groups.append(idx_group)

    h, w = np.array(cc_layers[0]).shape[:2]
    merged_small_layers: List[Image.Image] = []

    for idx_list in small_groups:
        merged_arr = np.zeros((h, w, 4), dtype=np.uint8)
        for idx in idx_list:
            arr = np.array(cc_layers[idx])
            mask = arr[:, :, 3] > 0
            merged_arr[mask] = arr[mask]
        merged_small_layers.append(Image.fromarray(merged_arr, "RGBA"))

    return big_layers + merged_small_layers


def group_components_by_kmeans(
    cc_layers: List[Image.Image],
    easy_threshold: int = 20,
    max_clusters: int = 8,
) -> List[Image.Image]:
    
    n = len(cc_layers)
    if n == 0:
        return []
    if n == 1:
        return cc_layers
    if n <= easy_threshold:
        return cc_layers

    feats, areas = _compute_component_features(cc_layers)

    valid_idx = [i for i, a in enumerate(areas) if a > 0]
    if len(valid_idx) <= 1:
        return cc_layers

    X = feats[valid_idx]

    max_k = min(max_clusters, len(valid_idx))
    if max_k <= 1:
        return cc_layers

    best_k = 2
    best_score = -1

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X)
        if len(set(labels_k)) < 2 or min(np.bincount(labels_k)) < 2:
            continue
        score = silhouette_score(X, labels_k)  

        if score > best_score:
            best_score = score
            best_k = k

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels_valid = km.fit_predict(X)

    labels = np.full(n, -1, dtype=int)
    for idx, lab in zip(valid_idx, labels_valid):
        labels[idx] = lab

    h, w = np.array(cc_layers[0]).shape[:2]
    groups = {}
    for i, lab in enumerate(labels):
        if lab < 0:
            continue
        groups.setdefault(int(lab), []).append(i)

    merged_layers: List[Image.Image] = []
    for _, idx_list in sorted(groups.items()):
        merged_arr = np.zeros((h, w, 4), dtype=np.uint8)

        for idx in idx_list:
            arr = np.array(cc_layers[idx])
            mask = arr[:, :, 3] > 0
            merged_arr[mask] = arr[mask]

        merged_layers.append(Image.fromarray(merged_arr, "RGBA"))

    return merged_layers


def split_by_connected_components(layer: Image.Image, min_area: int = 100) -> List[Image.Image]:

    layer_array = np.array(layer)
    alpha = layer_array[:, :, 3]
    
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(alpha, connectivity=8)
    
    new_layers = []
    for i in range(1, num_labels):  
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_area:
            continue
        
        mask = (labels == i).astype(np.uint8) * 255
        
        new_layer_array = layer_array.copy()
        new_layer_array[:, :, 3] = np.where(mask > 0, alpha, 0)
        
        new_layer = Image.fromarray(new_layer_array, 'RGBA')
        new_layers.append(new_layer)
    
    return new_layers


def refine_layers(input_layers: List[Image.Image], 
                  min_area: int = 100,
                  easy_threshold: int = 20,
                  max_clusters: int = 8,
                  use_refined_kmeans: bool = False,
                  big_ratio: float = 0.3,
                  size_ratio_threshold: float = 2.0) -> List[Image.Image]:
    
    refined_layers = []
    
    for i, layer in enumerate(input_layers):
        print(f"Processing layer {i+1}/{len(input_layers)}...")
        
        cc_layers = split_by_connected_components(layer, min_area)
        
        if use_refined_kmeans:
            new_layers = group_components_by_kmeans_refined(
                cc_layers, 
                easy_threshold=easy_threshold, 
                max_clusters=max_clusters,
                big_ratio=big_ratio,
                size_ratio_threshold=size_ratio_threshold
            )
        else:
            new_layers = group_components_by_kmeans(
                cc_layers, 
                easy_threshold=easy_threshold, 
                max_clusters=max_clusters
            )
        
        if not new_layers:
            new_layers = [layer]
        
        print(f"  Split into {len(new_layers)} sub-layers")
        refined_layers.extend(new_layers)
    
    return refined_layers


def process_sample(input_dir: Path, output_dir: Path, sample_id: str, 
                  min_area: int = 100,
                  easy_threshold: int = 20,
                  max_clusters: int = 8,
                  use_refined_kmeans: bool = False,
                  big_ratio: float = 0.3,
                  size_ratio_threshold: float = 2.0):
    
    sample_input_dir = input_dir / sample_id
    sample_output_dir = output_dir / sample_id
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    
    layer_files = sorted(sample_input_dir.glob("*.png"))
    input_layers = []
    for layer_file in layer_files:
        if layer_file.stem == 'reconstructed':  
            continue
        layer = Image.open(layer_file).convert('RGBA')
        input_layers.append(layer)
    
    print(f"\n{sample_id}: {len(input_layers)} input layers")
    
    refined_layers = refine_layers(
        input_layers, 
        min_area=min_area,
        easy_threshold=easy_threshold,
        max_clusters=max_clusters,
        use_refined_kmeans=use_refined_kmeans,
        big_ratio=big_ratio,
        size_ratio_threshold=size_ratio_threshold
    )
    
    print(f"Result: {len(refined_layers)} refined layers")
    
    for i, layer in enumerate(refined_layers):
        output_path = sample_output_dir / f"{i:04d}.png"
        layer.save(output_path)
    
    return len(input_layers), len(refined_layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Layer Refinement (CC + KMeans Grouping)")
    parser.add_argument("--input_dir", type=str, required=True, help="LayerD output directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Refined output directory")
    parser.add_argument("--min_area", type=int, default=100,
                       help="Minimum area for connected components")
    parser.add_argument("--easy_threshold", type=int, default=10,
                       help="If CC components <= this, skip KMeans grouping")
    parser.add_argument("--max_clusters", type=int, default=5,
                       help="Maximum number of clusters for KMeans grouping")
    parser.add_argument("--use_refined_kmeans", action="store_true",
                       help="Use refined KMeans method (big/small component separation)")
    parser.add_argument("--big_ratio", type=float, default=0.3,
                       help="Area ratio threshold for big components (only used with --use_refined_kmeans)")
    parser.add_argument("--size_ratio_threshold", type=float, default=2.0,
                       help="Max/min area ratio threshold for size similarity check (only used with --use_refined_kmeans)")
    parser.add_argument("--sample_id", type=str, help="Process specific sample")
    parser.add_argument("--all", action="store_true", help="Process all samples")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.all:
        samples = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        
        total_input = 0
        total_output = 0
        
        for sample_dir in samples:
            sample_id = sample_dir.name
            try:
                n_input, n_output = process_sample(
                    input_dir, output_dir, sample_id, 
                    args.min_area, args.easy_threshold, args.max_clusters,
                    args.use_refined_kmeans, args.big_ratio, args.size_ratio_threshold
                )
                total_input += n_input
                total_output += n_output
            except Exception as e:
                print(f"Error processing {sample_id}: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Summary:")
        print(f"  Total input layers: {total_input}")
        print(f"  Total output layers: {total_output}")
        print(f"  Average refinement: {total_input/len(samples):.1f} -> {total_output/len(samples):.1f} layers/sample")
        print(f"{'='*60}")
    
    elif args.sample_id:
        process_sample(
            input_dir, output_dir, args.sample_id, 
            args.min_area, args.easy_threshold, args.max_clusters,
            args.use_refined_kmeans, args.big_ratio, args.size_ratio_threshold
        )
    
    else:
        print("Please specify --sample_id or --all")