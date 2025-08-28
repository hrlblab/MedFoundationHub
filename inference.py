# inference.py ¡ª adapter/registry driven, MedGemma built-in, ready for extension
import os
import json
from typing import Dict, Tuple, Any
from peft import PeftModel
import torch
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, AutoModelForImageTextToText
from transformers import AutoModelForVision2Seq, BitsAndBytesConfig
from transformers import LlavaForConditionalGeneration
from transformers import BlipProcessor, BlipForConditionalGeneration
import uuid
import numpy as np
import os
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# -----------------------
# Paths / State
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
CONFIG_PATH = os.path.join(MODEL_DIR, "model.json")
os.makedirs(MODEL_DIR, exist_ok=True)

# Cache: model_key -> adapter instance
_LOADED_MODELS: Dict[str, Any] = {}

# Adapter registry: adapter_name -> adapter class
_ADAPTERS: Dict[str, type] = {}


# -----------------------
# Config helpers
# -----------------------
def load_model_config() -> Dict[str, dict]:
    """Read model.json into a Python dict."""
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_model_config(cfg: Dict[str, dict]):
    """Write config dictionary back to model.json."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

def _sanitize_key(name: str) -> str:
    """Normalize key: replace '-', '/' with '_' and lowercase."""
    return name.replace("-", "_").replace("/", "_").lower()

def _infer_model_key_from_path(path: str) -> str:
    """Derive model key from local directory path."""
    if not path:
        return None
    return _sanitize_key(os.path.basename(path.rstrip("/")))

def _detect_weights(local_model_dir: str) -> bool:
    """Check if the directory contains any common HF model weight files."""
    files = [
        "config.json",
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin",
    ]
    return any(os.path.isfile(os.path.join(local_model_dir, f)) for f in files)

def _guess_adapter_from_dir(model_dir: str) -> str:
    """
    Heuristic to guess adapter type from the directory name or config.
    Extend this when adding new model families.
    """
    name = os.path.basename(model_dir).lower()
    if "medgemma" in name:
        return "medgemma_it"
    # Default fallback
    if "flare25_qwen2.5vl" in name:
        return "flare25_qwen2_5vl"
    if "llava_med_v15" in name:
        return "llava_med_v15_mistral_7b"
    return "medgemma_it"


# -----------------------
# Model registration (HF / local)
# -----------------------
def add_model_from_local(local_model_dir: str):
    """Register a model from a local directory into model.json."""
    if not os.path.isdir(local_model_dir) or not _detect_weights(local_model_dir):
        raise RuntimeError(f"Not a valid HF weights dir: {local_model_dir}")

    key = _infer_model_key_from_path(local_model_dir)
    cfg = load_model_config()

    adapter = cfg.get(key, {}).get("adapter") or cfg.get(key, {}).get("infer_fn") or _guess_adapter_from_dir(local_model_dir)
    cfg[key] = {
        "alias": " ".join(key.split("_")).title(),
        "source": local_model_dir,
        "adapter": adapter,          # New field for internal adapter name
        "infer_fn": adapter,         # Backwards-compatible for /api/models
        "trust_remote_code": True,
        "dtype": "auto",
        "device_map": "auto",
    }
    _LOADED_MODELS.pop(key, None)  # Clear any existing cache
    save_model_config(cfg)
    return [key]

def add_model_from_hf(hf_model_id: str):
    """Download from HuggingFace and register locally."""
    local_dir = os.path.join(MODEL_DIR, hf_model_id.replace("/", "_"))
    if not os.path.exists(local_dir) or not _detect_weights(local_dir):
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=["*"],
        )
    return add_model_from_local(local_dir)


# -----------------------
# Adapter base class + registry
# -----------------------
def register_adapter(name: str):
    """Decorator to register an adapter class under a given name."""
    def deco(cls):
        _ADAPTERS[name] = cls
        cls.adapter_name = name
        return cls
    return deco

class BaseAdapter:
    """
    Base adapter class.
    Each adapter must implement:
        - load(): prepare model & processor, store in self.bundle
        - infer(image, prompt, **kwargs) -> str
    """
    adapter_name: str = "base"

    def __init__(self, meta: dict):
        self.meta = meta
        self.bundle = None

    def load(self):
        raise NotImplementedError

    def infer(self, image: Image.Image, prompt: str, **gen_cfg) -> str:
        raise NotImplementedError

    @staticmethod
    def _pick_dtype() -> torch.dtype:
        """Choose preferred dtype: bfloat16 if CUDA available, else float32."""
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32

    @staticmethod
    def _to_device_fp(batch: dict, device: torch.device, dtype: torch.dtype):
        """Move tensor batch to device, applying dtype only to floating-point tensors."""
        out = {}
        for k, v in batch.items():
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=dtype)
            else:
                out[k] = v.to(device)
        return out

    @staticmethod
    def _ensure_rgb(img: Image.Image) -> Image.Image:
        """Convert PIL image to RGB if not already."""
        return img if img.mode == "RGB" else img.convert("RGB")


# -----------------------
# Built-in adapter: MedGemma IT (google/medgemma-4b-it)
# -----------------------
@register_adapter("medgemma_it")
class MedGemmaAdapter(BaseAdapter):
    def load(self):
        dtype = self._pick_dtype()
        src = self.meta["source"]
        model = AutoModelForImageTextToText.from_pretrained(
            src,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        proc = AutoProcessor.from_pretrained(src)
        device = model.device
        self.bundle = (proc, model, device, dtype)

    def infer(self, image: Image.Image, prompt: str, **gen_cfg) -> str:
        if self.bundle is None:
            self.load()
        proc, model, device, dtype = self.bundle

        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image for 'image'")
        image = self._ensure_rgb(image)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [
                {"type": "text", "text": prompt or "Describe this medical image."},
                {"type": "image", "image": image},
            ]},
        ]
        inputs = proc.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = self._to_device_fp(inputs, model.device, dtype)
        input_len = inputs["input_ids"].shape[-1]

        max_new_tokens = int(gen_cfg.get("max_new_tokens", 2000))
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens
            )
        gen = out[0][input_len:]
        return proc.decode(gen, skip_special_tokens=True)

@register_adapter("flare25_qwen2_5vl")
class Flare25QwenAdapter(BaseAdapter):
    """
    LoRA for leoyinn/flare25-qwen2.5vl on top of Qwen/Qwen2.5-VL-7B-Instruct.
    Mirrors the model card: Vision2Seq base + PEFT adapter.
    """

    def load(self):
        import os, torch

        meta = self.meta
        dtype = self._pick_dtype()
        adapter_dir = meta["source"]  # local path for leoyinn/flare25-qwen2.5vl
        base_model = meta.get("base_model") or "Qwen/Qwen2.5-VL-7B-Instruct"

        # Offload folder (needed when device_map="auto" sharding to disk)
        offload_dir = meta.get("offload_dir") or os.path.join(os.path.dirname(__file__), "offload_qwen")
        os.makedirs(offload_dir, exist_ok=True)

        # Optional 4-bit quant (saves a lot of VRAM)
        load_in_4bit = bool(meta.get("load_in_4bit", False))
        bnb_cfg = None
        if load_in_4bit:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # 1) Load base with AutoModelForVision2Seq (as per README)
        base_kwargs = dict(
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            offload_folder=offload_dir if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        if load_in_4bit:
            base = AutoModelForVision2Seq.from_pretrained(
                base_model, quantization_config=bnb_cfg, **base_kwargs
            )
        else:
            base = AutoModelForVision2Seq.from_pretrained(
                base_model, torch_dtype=dtype, **base_kwargs
            )

        # 2) Attach LoRA adapter (PEFT)
        model = PeftModel.from_pretrained(base, adapter_dir)

        # 3) Processor: prefer adapter repo (has chat_template), fallback to base
        try:
            proc = AutoProcessor.from_pretrained(adapter_dir, trust_remote_code=False)
        except Exception:
            proc = AutoProcessor.from_pretrained(base_model, trust_remote_code=False)

        self.bundle = (proc, model, model.device, dtype)

    def infer(self, image: Image.Image, prompt: str, **gen_cfg) -> str:
        if self.bundle is None:
            self.load()
        proc, model, device, dtype = self.bundle

        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image for 'image'")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1) Build chat messages (include an image content entry)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},  # <-- placeholder; the actual pixel data1 is passed below
                {"type": "text", "text": prompt},
            ]},
        ]

        # 2) Get templated text *with* image tokens; DO NOT tokenize here
        templated = proc.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,  # <-- important
        )

        # 3) Now let the processor align text+images and create tensors
        inputs = proc(
            text=[templated],
            images=[image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # 4) Move tensors to model device; cast only float tensors
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(model.device, dtype=dtype)
            else:
                inputs[k] = v.to(model.device)

        max_new_tokens = int(gen_cfg.get("max_new_tokens", 2560))
        do_sample = bool(gen_cfg.get("do_sample", False))
        temperature = float(gen_cfg.get("temperature", 0.7 if do_sample else 0.0))
        top_p = float(gen_cfg.get("top_p", 0.9 if do_sample else 1.0))

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        # Decode; processor.decode works for Qwen2.5-VL batches too
        if hasattr(proc, "decode"):
            return proc.decode(out[0], skip_special_tokens=True)
        tok = getattr(proc, "tokenizer", None)
        return tok.decode(out[0], skip_special_tokens=True) if tok else str(out[0])

@register_adapter("llava_med_v15_mistral_7b")
class LlavaMedV15MistralAdapter(BaseAdapter):
    """
    Adapter for microsoft/llava-med-v1.5-mistral-7b (full checkpoint, no LoRA).
    Uses Transformers' LlavaForConditionalGeneration + AutoProcessor.
    Requires a recent transformers (>=4.50+; check docs if older).
    """

    def load(self):
        import os, torch

        meta = self.meta
        dtype = self._pick_dtype()
        repo_or_path = meta["source"]  # local dir for microsoft/llava-med-v1.5-mistral-7b

        # Optional disk offload if device_map="auto" needs to spill (low VRAM rigs)
        offload_dir = meta.get("offload_dir") or os.path.join(os.path.dirname(__file__), "offload_llava_med")
        os.makedirs(offload_dir, exist_ok=True)

        self.model = LlavaForConditionalGeneration.from_pretrained(
            repo_or_path,
            torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            offload_folder=offload_dir if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,  # model is native to HF; no custom code needed
        )
        self.processor = AutoProcessor.from_pretrained(
            repo_or_path,
            trust_remote_code=False
        )

        # (Optional) better batching behavior per HF doc tip
        try:
            self.processor.tokenizer.padding_side = "left"
        except Exception:
            pass

        self.bundle = (self.processor, self.model, self.model.device, dtype)

    def infer(self, image: Image.Image, prompt: str, **gen_cfg) -> str:
        if self.bundle is None:
            self.load()
        proc, model, device, dtype = self.bundle

        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Chat-style conversation; pass image via `images=[...]` so the template inserts image tokens
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt or "Describe the key medical findings."},
            ]},
        ]

        # Let the template tokenize AND insert image tokens; request a Mapping (not bare tensor)
        inputs = proc.apply_chat_template(
            messages,
            images=[image],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # Move to device; only cast float tensors
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(model.device, dtype=dtype)
            else:
                inputs[k] = v.to(model.device)

        max_new_tokens = int(gen_cfg.get("max_new_tokens", 256))
        do_sample = bool(gen_cfg.get("do_sample", False))
        temperature = float(gen_cfg.get("temperature", 0.7 if do_sample else 0.0))
        top_p = float(gen_cfg.get("top_p", 0.9 if do_sample else 1.0))

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        # Batch decode; LLaVA processors support it
        return proc.batch_decode(out, skip_special_tokens=True)[0]
@register_adapter("blip_caption_base")
class BlipCaptionBaseAdapter(BaseAdapter):
    """
    Adapter for Salesforce/blip-image-captioning-base.
    Simple image -> text captioning. No chat template.
    """

    def load(self):
        import os, torch

        meta = self.meta
        dtype = self._pick_dtype()
        repo_or_path = meta["source"]  # local dir for Salesforce/blip-image-captioning-base

        # It's a small model; still support auto sharding/offload to be consistent
        offload_dir = meta.get("offload_dir") or os.path.join(os.path.dirname(__file__), "offload_blip")
        os.makedirs(offload_dir, exist_ok=True)

        self.model = BlipForConditionalGeneration.from_pretrained(
            repo_or_path,
            torch_dtype=(dtype if torch.cuda.is_available() else torch.float32),
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            offload_folder=offload_dir if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        self.processor = BlipProcessor.from_pretrained(repo_or_path, trust_remote_code=False)
        self.bundle = (self.processor, self.model, self.model.device, dtype)

    def infer(self, image, prompt: str, **gen_cfg) -> str:
        if self.bundle is None:
            self.load()
        proc, model, device, dtype = self.bundle

        if not isinstance(image, Image.Image):
            raise TypeError("Expected PIL.Image")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # BLIP can optionally take a text prompt as guidance (e.g., "a detailed medical description:")
        text = prompt.strip() if (prompt and isinstance(prompt, str)) else None

        inputs = proc(images=image, text=text, return_tensors="pt")
        # move to device; cast only float tensors
        for k, v in list(inputs.items()):
            if hasattr(v, "dtype") and v.dtype.is_floating_point:
                inputs[k] = v.to(model.device, dtype=dtype)
            else:
                inputs[k] = v.to(model.device)

        max_new_tokens = int(gen_cfg.get("max_new_tokens", 64))  # captions don't need 500 tokens, hero
        do_sample = bool(gen_cfg.get("do_sample", False))
        temperature = float(gen_cfg.get("temperature", 0.7 if do_sample else 0.0))
        top_p = float(gen_cfg.get("top_p", 0.9 if do_sample else 1.0))

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
            )

        # BLIP uses a tokenizer inside the processor
        tok = getattr(proc, "tokenizer", None)
        if tok is not None:
            return tok.decode(out[0], skip_special_tokens=True)
        # processor sometimes exposes .decode too; fallback either way
        if hasattr(proc, "decode"):
            return proc.decode(out[0], skip_special_tokens=True)
        return str(out[0])

@register_adapter("conch_embed")
class ConchAdapter(BaseAdapter):
    """
    Adapter for MahmoodLab/CONCH (pathology feature extractor).
    - Backbone: ViT-Base/16, img_size=256, embed_dim=768, depth=12
    - Loads from HF gated repo via timm pretrained_cfg hf_hub_id
    - Returns an embedding vector and saves it to disk; DOES NOT generate free-form text
    """

    def load(self):
        # Use the exact backbone + settings matching the checkpoint to avoid key mismatches.
        # Key details:
        #   - vit_base_patch16_224 backbone
        #   - img_size=256  -> 16x16 patches => 256 tokens + 1 CLS => pos_embed length 257
        #   - num_classes=0 to expose features (no classifier)
        #   - global_pool="" (disable fc_norm) because checkpoint uses "norm", not "fc_norm"
        hf_id = "MahmoodLab/CONCH"  # the HF repo; access is gated
        dtype = self._pick_dtype()

        # Create model
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            pretrained_cfg={"hf_hub_id": hf_id},
            img_size=256,
            num_classes=0,
            global_pool="",
        )

        self.model.eval()

        # Device policy (GPU if available, else CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Build transforms consistent with the model's configuration
        self.cfg = resolve_data_config({"input_size": (3, 256, 256)}, model=self.model)
        self.transform = create_transform(**self.cfg)

        self.device = device
        self.dtype = dtype
        self.bundle = (self.model, self.transform, self.device, self.dtype)

    def _extract_tokens(self, feats: torch.Tensor | dict) -> torch.Tensor:
        """
        Robustly get a [B, T, C] tokens tensor or [B, C] embedding from timm outputs.
        - Prefer dict['x'] if present (tokens including CLS)
        - Fallback to dict['pooled'] / first tensor / raw tensor
        - If [B, T, C], return as-is (caller decides CLS vs mean pool)
        """
        if isinstance(feats, dict):
            x = feats.get("x", None)
            if x is None:
                x = feats.get("pooled", None)
            if x is None:
                for v in feats.values():
                    if torch.is_tensor(v):
                        x = v
                        break
            if x is None:
                raise RuntimeError("CONCH: Unable to locate feature tensor in model outputs.")
            return x
        # feats is a tensor
        return feats

    def _forward_embedding(self, pixel: torch.Tensor) -> torch.Tensor:
        """
        Forward once and produce a single [B, C] embedding.
        Strategy:
          - forward_features -> tokens
          - if tokens are [B, T, C], take CLS token (index 0)
          - if already [B, C], use directly
        """
        on_cuda = torch.cuda.is_available()
        if on_cuda:
            # autocast on CUDA only; CPU autocast is not universally supported across torch versions
            autocast_ctx = torch.autocast(device_type="cuda", dtype=self.dtype)
        else:
            # no autocast on CPU to avoid dtype issues
            class _NullCtx:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            autocast_ctx = _NullCtx()

        with torch.no_grad(), autocast_ctx:
            feats = self.model.forward_features(pixel)
            tokens = self._extract_tokens(feats)  # [B, T, C] or [B, C]
            if tokens.dim() == 3:
                emb = tokens[:, 0]  # CLS
            else:
                emb = tokens
            return emb  # [B, C]

    def infer(self, image: Image.Image, prompt: str, **gen_cfg) -> str:
        """
        Input: PIL.Image (ideally a pathology tile, not a full WSI)
        Output: brief text summary + saves a .npy embedding file
        """
        if self.bundle is None:
            self.load()
        model, transform, device, dtype = self.bundle

        if not isinstance(image, Image.Image):
            raise TypeError("CONCH expects a PIL.Image as 'image'")
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        pixel = transform(image).unsqueeze(0).to(device)  # [1, C, H, W]

        # Forward to get [1, C] embedding
        emb = self._forward_embedding(pixel)  # [1, C]
        if emb.dim() > 2:
            # safety squeeze if some timm variant returns extra dims
            emb = emb.mean(dim=1)

        emb_np = emb.detach().cpu().float().numpy().squeeze()
        dim = int(emb_np.shape[0])
        norm = float(np.linalg.norm(emb_np) + 1e-8)

        # Save vector to disk
        out_dir = gen_cfg.get("save_dir") or self.meta.get("save_dir") or os.path.join(os.path.dirname(__file__), "embeddings")
        os.makedirs(out_dir, exist_ok=True)
        out_pth = os.path.join(out_dir, f"conch_{uuid.uuid4().hex[:8]}.npy")
        np.save(out_pth, emb_np)

        # Return a compact summary string for your current UI (which expects text)
        return (f"[CONCH] embedding: dim={dim}, L2={norm:.3f}. saved: {out_pth}\n"
                f"Tip: use embeddings for MIL, retrieval, clustering; not for text generation.")


# -----------------------
# Loader / Dispatcher
# -----------------------
def _get_adapter_instance(model_key: str) -> BaseAdapter:
    """Fetch or load the adapter instance for the given model key."""
    cfg = load_model_config()
    if model_key not in cfg:
        raise ValueError(f"Unknown model: {model_key}")
    meta = cfg[model_key]
    adapter_key = meta.get("adapter") or meta.get("infer_fn") or "medgemma_it"
    if adapter_key not in _ADAPTERS:
        raise NotImplementedError(f"Adapter '{adapter_key}' not registered")

    if model_key in _LOADED_MODELS:
        inst = _LOADED_MODELS[model_key]
        if isinstance(inst, BaseAdapter):
            return inst

    inst = _ADAPTERS[adapter_key](meta)
    inst.load()
    _LOADED_MODELS[model_key] = inst
    return inst

def load_model(model_key: str):
    """
    Public loader to match existing backend API.
    Returns the adapter instance.
    """
    return _get_adapter_instance(model_key)


# -----------------------
# Public API
# -----------------------
def generate_report(image: Image.Image, user_prompt: str, model_key: str) -> str:
    """
    Main entry point for the backend to run inference.
    Dispatches to the appropriate adapter based on model.json.
    """
    adapter = _get_adapter_instance(model_key)
    return adapter.infer(image=image, prompt=user_prompt or "")
