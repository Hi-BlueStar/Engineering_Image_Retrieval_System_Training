import os
import json
import uuid
import shutil
import random
import logging
import torch
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExpertGUI")

app = Flask(__name__)

# Security: Generate a CSRF token for the session
SESSION_CSRF_TOKEN = str(uuid.uuid4())

WORKSPACE_ROOT = Path("/home/master-user/Desktop/Engineering_Image_Retrieval_System_Training").resolve()
DATASET_ROOT = WORKSPACE_ROOT / "dataset_v2"
SPLIT_JSON_PATH = DATASET_ROOT / "validation_split.json"

# Helper path mapping
def to_local(canonical_path_str: str) -> Path:
    """Map canonical docker path starting with /workspace to local workspace path."""
    canonical_path_str = canonical_path_str.replace("\\", "/")
    if canonical_path_str.startswith("/workspace"):
        return WORKSPACE_ROOT / canonical_path_str[len("/workspace"):].lstrip("/")
    return Path(canonical_path_str).resolve()

def to_canonical(local_path_str: str) -> str:
    """Map local workspace path to canonical docker path starting with /workspace."""
    local_path_str = os.path.normpath(local_path_str).replace("\\", "/")
    workspace_str = str(WORKSPACE_ROOT).replace("\\", "/")
    if local_path_str.startswith(workspace_str):
        rel = os.path.relpath(local_path_str, start=workspace_str).replace("\\", "/")
        return f"/workspace/{rel}"
    return local_path_str

# Symlink builder logic matching build_dataset_v2
def build_symlinks(mapping: dict, target_dir: Path):
    """Create relative symlinks to save disk space."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for category, paths in mapping.items():
        cat_dir = target_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        for p_str in paths:
            # We resolve the local path of the target image
            p = to_local(p_str)
            link_path = cat_dir / p.name
            if link_path.is_symlink() or link_path.exists():
                try:
                    if link_path.is_dir() and not link_path.is_symlink():
                        shutil.rmtree(link_path)
                    else:
                        link_path.unlink()
                except Exception as e:
                    logger.warning(f"Error removing existing path {link_path}: {e}")
            try:
                # Create a relative symbolic link
                rel_path = os.path.relpath(p.resolve(), start=link_path.parent)
                link_path.symlink_to(rel_path)
            except Exception:
                # Fallback: Copy if symlink is not supported
                try:
                    shutil.copy2(p, link_path)
                except Exception as e:
                    logger.error(f"Copy fallback failed: {p} -> {link_path} | Error: {e}")

@app.route("/")
def index():
    # Pass the CSRF token to the template
    return render_template("index.html", csrf_token=SESSION_CSRF_TOKEN)

@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    if not SPLIT_JSON_PATH.exists():
        return jsonify({"error": "validation_split.json not found"}), 404
        
    try:
        with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Get category names
        categories_dir = WORKSPACE_ROOT / "data" / "converted_labeled_images" / "吉輔提供資料"
        categories = []
        if categories_dir.exists():
            categories = sorted([d.name for d in categories_dir.iterdir() if d.is_dir()])
        
        # Add unlabeled category to list
        unlabeled_dir = WORKSPACE_ROOT / "data" / "converted_images"
        if unlabeled_dir.exists():
            categories.append("converted_images")
            
        return jsonify({
            "seeds": data.get("seeds", []),
            "gt_selections": data.get("gt_selections", {}),
            "distractors": data.get("distractors", []),
            "categories": categories,
            "stats": {
                "total_v": len(data.get("V", [])),
                "total_t_small": len(data.get("T_small", [])),
                "total_t_large": len(data.get("T_large", []))
            }
        })
    except Exception as e:
        logger.error(f"Error reading validation split: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/category/<category_name>", methods=["GET"])
def get_category_images(category_name):
    # Sanitize category_name to prevent directory traversal
    category_name = os.path.basename(category_name)
    if category_name == "converted_images":
        category_dir = WORKSPACE_ROOT / "data" / "converted_images"
    else:
        category_dir = WORKSPACE_ROOT / "data" / "converted_labeled_images" / "吉輔提供資料" / category_name
    
    if not category_dir.exists() or not category_dir.is_dir():
        return jsonify({"error": "Category not found"}), 404
        
    try:
        images = []
        for p in sorted(category_dir.glob("*.png")):
            canonical_path = to_canonical(str(p))
            images.append({
                "name": p.name,
                "canonical_path": canonical_path
            })
        return jsonify({"images": images})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Global variables for features and path indices
FEATURES_CACHE = None
IMAGE_PATHS = []
PATH_TO_IDX = {}

def get_features_cache():
    global FEATURES_CACHE, IMAGE_PATHS, PATH_TO_IDX
    if FEATURES_CACHE is not None:
        return FEATURES_CACHE, IMAGE_PATHS, PATH_TO_IDX
        
    cache_path = DATASET_ROOT / "features_cache.pt"
    if not cache_path.exists():
        logger.warning(f"Feature cache not found at {cache_path}.")
        return None, [], {}
        
    try:
        logger.info("Loading feature cache...")
        cache = torch.load(cache_path, map_location="cpu")
        FEATURES_CACHE = cache["features"]
        # L2 Normalize the features so dot product equals cosine similarity
        FEATURES_CACHE = FEATURES_CACHE / torch.norm(FEATURES_CACHE, dim=1, keepdim=True)
        
        # Paths from cache are strings. Convert to canonical format.
        raw_paths = cache["paths"]
        IMAGE_PATHS = [to_canonical(str(p)) for p in raw_paths]
        PATH_TO_IDX = {p: i for i, p in enumerate(IMAGE_PATHS)}
        logger.info(f"Loaded normalized features for {len(IMAGE_PATHS)} images.")
    except Exception as e:
        logger.error(f"Error loading features cache: {e}")
        
    return FEATURES_CACHE, IMAGE_PATHS, PATH_TO_IDX

# Try loading it immediately when the module is run so it's ready
try:
    get_features_cache()
except Exception as e:
    logger.warning(f"Could not load feature cache on import: {e}")

@app.route("/api/candidates", methods=["GET"])
def get_top_candidates():
    seed_path = request.args.get("path")
    if not seed_path:
        return jsonify({"error": "Missing path parameter"}), 400
        
    # Standardize seed path
    canonical_seed = to_canonical(str(to_local(seed_path)))
    
    features, paths, path_to_idx = get_features_cache()
    if features is None:
        return jsonify({"error": "Feature cache not initialized. Please run precompute_features.py first."}), 500
        
    if canonical_seed not in path_to_idx:
        logger.warning(f"Seed not found in feature index: {canonical_seed}")
        return jsonify({"error": "Seed not found in candidate pool"}), 404
        
    try:
        seed_idx = path_to_idx[canonical_seed]
        seed_feat = features[seed_idx].unsqueeze(0) # [1, D]
        
        # Compute cosine similarity with all features
        similarities = (seed_feat @ features.T).squeeze(0) # [N]
        similarities[seed_idx] = -1.0 # Exclude itself
        
        # Get Top 50 indices
        top_k = 50
        top_k_indices = torch.topk(similarities, k=min(top_k, len(paths) - 1)).indices.tolist()
        
        candidates = []
        for idx in top_k_indices:
            p_str = paths[idx]
            p_local = to_local(p_str)
            candidates.append({
                "name": p_local.name,
                "canonical_path": p_str
            })
            
        return jsonify({"images": candidates})
    except Exception as e:
        logger.error(f"Error computing candidates: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/image", methods=["GET"])
def get_image():
    path_arg = request.args.get("path")
    if not path_arg:
        return jsonify({"error": "Missing path parameter"}), 400
        
    # Resolve the path to local absolute path
    local_path = to_local(path_arg)
    resolved_path = str(local_path.resolve())
    
    # SECURITY: Verify path boundary strictly to prevent directory traversal
    allowed_dirs = [
        str((WORKSPACE_ROOT / "data" / "converted_images").resolve()),
        str((WORKSPACE_ROOT / "data" / "converted_labeled_images").resolve()),
        str(DATASET_ROOT.resolve())
    ]
    
    is_allowed = False
    for allowed in allowed_dirs:
        try:
            # Check if resolved_path is inside allowed dir
            common = os.path.commonpath([resolved_path, allowed])
            if common == allowed:
                is_allowed = True
                break
        except ValueError:
            # Paths on different drives or invalid paths
            continue
            
    if not is_allowed:
        logger.warning(f"SECURITY ALERT: Blocked access to path {resolved_path}")
        return "Forbidden", 403
        
    if not os.path.exists(resolved_path):
        return "Not Found", 404
        
    return send_file(resolved_path)

@app.route("/api/save", methods=["POST"])
def save_dataset():
    # SECURITY: Validate CSRF token in custom header
    csrf_token = request.headers.get("X-CSRF-Token")
    if csrf_token != SESSION_CSRF_TOKEN:
        logger.warning("SECURITY ALERT: CSRF token mismatch")
        return jsonify({"error": "Forbidden: CSRF token invalid"}), 403
        
    payload = request.json
    if not payload:
        return jsonify({"error": "Missing JSON payload"}), 400
        
    new_gt_selections = payload.get("gt_selections")
    new_distractors = payload.get("distractors")
    
    if new_gt_selections is None or new_distractors is None:
        return jsonify({"error": "Invalid payload keys"}), 400
        
    try:
        # Load the original validation_split.json to keep standard values (seeds, split_ratio, etc.)
        with open(SPLIT_JSON_PATH, "r", encoding="utf-8") as f:
            original_meta = json.load(f)
            
        seeds = original_meta.get("seeds", [])
        
        # 1. Collect all validation set V images (Seeds + GTs + Distractors)
        all_v_images = set(seeds)
        for s in seeds:
            gts = new_gt_selections.get(str(s), [])
            all_v_images.update(gts)
            
        all_v_images.update(new_distractors)
        
        # 2. Get all universe images (Da + Db)
        da_dir = WORKSPACE_ROOT / "data" / "converted_labeled_images"
        db_dir = WORKSPACE_ROOT / "data" / "converted_images"
        
        all_da_paths = [to_canonical(str(p)) for p in da_dir.rglob("*.png") if p.is_file()]
        all_db_paths = [to_canonical(str(p)) for p in db_dir.rglob("*.png") if p.is_file()]
        
        all_da_set = set(all_da_paths)
        all_universe_set = all_da_set.union(all_db_paths)
        
        # 3. Calculate remaining images for training sets T_small and T_large
        t_small_images = sorted(list(all_da_set.difference(all_v_images)))
        t_large_images = sorted(list(all_universe_set.difference(all_v_images)))
        
        # 4. Save metadata JSON
        meta = {
            "seeds": seeds,
            "gt_selections": new_gt_selections,
            "distractors": sorted(list(new_distractors)),
            "V": sorted(list(all_v_images)),
            "T_small": t_small_images,
            "T_large": t_large_images
        }
        
        with open(SPLIT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Updated {SPLIT_JSON_PATH} successfully.")
        
        # 5. Rebuild symlinks in dataset_v2
        split_ratio = 0.8
        seed = 42
        rng = random.Random(seed)
        
        # Build T_small
        t_small_shuffled = t_small_images[:]
        rng.shuffle(t_small_shuffled)
        split_idx_s = int(len(t_small_shuffled) * split_ratio)
        t_small_train = t_small_shuffled[:split_idx_s]
        t_small_test = t_small_shuffled[split_idx_s:]
        
        # Clean T_small first
        t_small_dir = DATASET_ROOT / "T_small" / f"Run_01_Seed_{seed}"
        if t_small_dir.exists():
            shutil.rmtree(t_small_dir)
        build_symlinks(
            {"Component_Dataset/train": t_small_train, "Component_Dataset/test": t_small_test},
            t_small_dir
        )
        
        # Build T_large
        t_large_shuffled = t_large_images[:]
        rng.shuffle(t_large_shuffled)
        split_idx_l = int(len(t_large_shuffled) * split_ratio)
        t_large_train = t_large_shuffled[:split_idx_l]
        t_large_test = t_large_shuffled[split_idx_l:]
        
        # Clean T_large first
        t_large_dir = DATASET_ROOT / "T_large" / f"Run_01_Seed_{seed}"
        if t_large_dir.exists():
            shutil.rmtree(t_large_dir)
        build_symlinks(
            {"Component_Dataset/train": t_large_train, "Component_Dataset/test": t_large_test},
            t_large_dir
        )
        
        # Build V
        v_mapping = {}
        for i, s_str in enumerate(seeds):
            group_name = f"group_{i:03d}"
            # Seed itself + GTs
            group_images = [s_str] + new_gt_selections.get(s_str, [])
            # Deduplicate while preserving order
            group_images = list(dict.fromkeys(group_images))
            v_mapping[group_name] = group_images
            
        for j, dist in enumerate(new_distractors):
            v_mapping[f"distractor_{j:04d}"] = [dist]
            
        # Clean V first
        v_dir = DATASET_ROOT / "V"
        if v_dir.exists():
            shutil.rmtree(v_dir)
        build_symlinks(v_mapping, v_dir)
        
        logger.info("Successfully rebuilt symlink structure for V, T_small, and T_large.")
        
        return jsonify({
            "success": True,
            "stats": {
                "total_v": len(all_v_images),
                "total_t_small": len(t_small_images),
                "total_t_large": len(t_large_images)
            }
        })
    except Exception as e:
        logger.error(f"Error rebuilding dataset splits: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Bound strictly to 127.0.0.1 for security
    app.run(host="127.0.0.1", port=5003, debug=True)
