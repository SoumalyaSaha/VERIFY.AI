"""
Nonescape full-model wrapper (no reimplementation).
e3ntity/nonescape NonescapeClassifier (DINOv2-large + EfficientNetV2-L,
2.2GB safetensors) used exactly as published: repo preprocess_image +
forward; fake_probability = probs[1] ("synthetic"), verified behaviorally
(clean Gemini/ChatGPT score 0.74-1.0 synthetic; real phone photos 0.70-0.98
authentic).

CPU placement: the fp32 weights (2.2GB) exceed current VRAM headroom
(~1.3GB free with all other services resident), so this service runs on CPU
(~1.2s/image). No numerics deviation (published fp32 path).
"""
import io, os, sys, time, logging
import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

_this_dir = os.path.dirname(os.path.abspath(__file__))
_vendor_pkg = os.path.join(_this_dir, "nonescape")
if os.path.isdir(_vendor_pkg) and _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
elif not os.path.isdir(_vendor_pkg):
    _legacy = os.getenv("NONESCAPE_SRC", "")
    if _legacy and os.path.isdir(_legacy):
        sys.path.insert(0, _legacy)

os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/app/hf_cache"))
os.environ.setdefault("HF_HUB_CACHE", os.getenv("HF_HUB_CACHE", "/app/hf_cache/hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.getenv("HUGGINGFACE_HUB_CACHE", "/app/hf_cache/hub"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("nonescape")
app = FastAPI(title="Nonescape AI-Image Detector")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
DEVICE = torch.device("cpu")
WEIGHTS_PATH = os.getenv("NONESCAPE_WEIGHTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "weights", "nonescape-v0.safetensors"))

model = None


@app.on_event("startup")
async def load_model():
    global model
    from nonescape import NonescapeClassifier
    logger.info(f"Loading full NonescapeClassifier from {WEIGHTS_PATH} (strict default)")
    model = NonescapeClassifier.from_pretrained(WEIGHTS_PATH)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Weights loaded: {n_params / 1e6:.1f}M params")
    model = model.to(DEVICE).eval()
    logger.info(f"Nonescape ready on {DEVICE}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "Nonescape",
        "variant": "nonescape-v0-full",
        "device": str(DEVICE),
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    t0 = time.time()
    try:
        from nonescape import preprocess_image
        img = Image.open(io.BytesIO(await file.read())).convert("RGB")
        tensor = preprocess_image(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            probs = model(tensor)
            fake_prob = probs[0][1].item()
        return {
            "model": "Nonescape",
            "fake_probability": round(fake_prob, 4),
            "verdict": "fake" if fake_prob > 0.5 else "real",
            "latency_ms": int((time.time() - t0) * 1000),
        }
    except Exception as e:
        logger.error(f"Nonescape detect error: {e}")
        raise HTTPException(500, detail=str(e))
