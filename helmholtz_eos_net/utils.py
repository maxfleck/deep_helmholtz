import os

def fix_paths(obj, new_root):
    """Re-root any absolute paths that share the anchor directory name of new_root."""
    anchor = os.path.basename(new_root)
    if isinstance(obj, dict):
        return {k: fix_paths(v, new_root) for k, v in obj.items()}
    if isinstance(obj, str) and os.sep in obj and anchor in obj:
        suffix = obj.split(anchor, 1)[1]
        return new_root + suffix
    return obj
