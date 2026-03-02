"""
集中维护 Hugging Face 相关的缓存目录与模型快照路径，避免脚本之间出现不一致的硬编码。
"""

from pathlib import Path

# 统一的 Hugging Face 缓存根目录。后续新增模型时请同样下载到此目录，便于脚本共享缓存。
HF_CACHE_ROOT = Path(r"H:/huggingface-cache/hub")
# 确保目录存在，减少因目录缺失导致的加载错误。
HF_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

# 本地缓存的 CLIPScore 模型快照路径（openai/clip-vit-base-patch16）。
DEFAULT_CLIPSCORE_SNAPSHOT = (
    HF_CACHE_ROOT
    / "models--openai--clip-vit-base-patch16"
    / "snapshots"
    / "57c216476eefef5ab752ec549e440a49ae4ae5f3"
)


def resolve_model_identifier(model_name: str) -> str:
    """
    根据输入判断是本地路径还是 Hugging Face 模型名，并返回 transformers 可以识别的字符串。
    使用中文注释详细说明：如果输入是本地目录且存在，就直接返回绝对路径，避免误触网络；否则返回原始模型名，由 transformers 自动下载到统一缓存目录。
    """

    model_path = Path(model_name)
    if model_path.exists():
        return str(model_path)
    return model_name
