# ============================================================================
# MERGED FASTAPI: TIFF PROCESSING + VLN IMAGE QUERY WITH CLAUDE SONNET 4.5
# ============================================================================

# INSTALL:
# pip install fastapi uvicorn python-multipart rasterio matplotlib scikit-image opencv-python-headless torch albumentations pillow transformers scikit-learn anthropic

import os, io, json, re, warnings, numpy as np
warnings.filterwarnings('ignore')

from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import rasterio
import cv2
from skimage.filters import gaussian
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import BlipForConditionalGeneration, BlipProcessor
from anthropic import Anthropic

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not ANTHROPIC_API_KEY:
    print("=" * 80)
    print("WARNING: ANTHROPIC_API_KEY not set!")
    print("Please set it as an environment variable:")
    print("  export ANTHROPIC_API_KEY='your-api-key-here'")
    print("=" * 80)

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# FASTAPI INIT
# ======================
app = FastAPI(title="Merged TIFF Processing & VLN API with Claude")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ======================
# LOAD VLN MODELS
# ======================
print("Loading BLIP model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
print("✅ BLIP model loaded successfully!")

print("Initializing Claude API client...")
if ANTHROPIC_API_KEY:
    claude_client = Anthropic(api_key=ANTHROPIC_API_KEY)
    print("✅ Claude API client initialized!")
else:
    claude_client = None
    print("❌ Claude API client NOT initialized - API key missing")

# ======================
# TIFF PROCESSING FUNCTIONS
# ======================
class CustomTransforms:
    @staticmethod
    def to_tensor(data):
        return torch.from_numpy(data.copy()).float()
    
    @staticmethod
    def resize_tensor(tensor, size):
        if len(tensor.shape) == 2: 
            tensor = tensor[np.newaxis, np.newaxis, :, :]
        elif len(tensor.shape) == 3: 
            tensor = tensor[np.newaxis, :, :, :]
        resized = F.interpolate(torch.tensor(tensor), size=size, mode='bilinear', align_corners=False)
        return resized.squeeze().numpy()

transforms = CustomTransforms()

def tiff_to_png(tiff_array):
    if tiff_array.shape[0] >= 3:
        rgb = np.stack([tiff_array[0], tiff_array[1], tiff_array[2]], axis=2)
    elif tiff_array.shape[0] == 2:
        rgb = np.stack([tiff_array[0], tiff_array[1], np.zeros_like(tiff_array[0])], axis=2)
    elif tiff_array.shape[0] == 1:
        gray = tiff_array[0]
        rgb = np.stack([gray, gray, gray], axis=2)
    else:
        raise ValueError("No bands to create PNG")
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    rgb = (rgb * 255).astype(np.uint8)
    return Image.fromarray(rgb)

def zoom_tiff(tiff_array, zoom_factor):
    bands, H, W = tiff_array.shape
    y0, x0 = int(H*(1-zoom_factor)/2), int(W*(1-zoom_factor)/2)
    y1, x1 = y0 + int(H*zoom_factor), x0 + int(W*zoom_factor)
    return tiff_array[:, y0:y1, x0:x1]

def normalize_bands(img):
    bands_norm = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        band = img[i].astype(np.float32)
        p2, p98 = np.percentile(band, [2, 98])
        bands_norm[i] = np.clip((band - p2)/(p98 - p2 + 1e-8), 0, 1)
    return bands_norm

def anisotropic_diffusion(img, iterations=10, kappa=30, dt=0.2):
    img = img.copy()
    for _ in range(iterations):
        nabla_n = np.roll(img, -1, axis=0) - img
        nabla_s = np.roll(img, 1, axis=0) - img
        nabla_e = np.roll(img, -1, axis=1) - img
        nabla_w = np.roll(img, 1, axis=1) - img
        c_n = np.exp(-(nabla_n/kappa)**2)
        c_s = np.exp(-(nabla_s/kappa)**2)
        c_e = np.exp(-(nabla_e/kappa)**2)
        c_w = np.exp(-(nabla_w/kappa)**2)
        img += dt*(c_n*nabla_n + c_s*nabla_s + c_e*nabla_e + c_w*nabla_w)
    return img

def enhance_bands(bands):
    enhanced = np.zeros_like(bands)
    for i in range(bands.shape[0]):
        enhanced[i] = anisotropic_diffusion(bands[i])
        blurred = gaussian(enhanced[i], sigma=1)
        enhanced[i] = np.clip(enhanced[i] + 0.5*(enhanced[i]-blurred), 0, 1)
    return enhanced

def super_resolve(bands, scale=2):
    sr_bands = np.zeros((bands.shape[0], bands.shape[1]*scale, bands.shape[2]*scale), dtype=np.float32)
    for i in range(bands.shape[0]):
        sr_bands[i] = cv2.resize(bands[i], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return sr_bands

def calculate_ndvi(red, nir):
    return (nir - red)/(nir + red + 1e-8)

def extract_features(sr_bands):
    # NDVI
    if sr_bands.shape[0] >= 5:
        ndvi = calculate_ndvi(sr_bands[3], sr_bands[4])
    else:
        ndvi = np.zeros((sr_bands.shape[1], sr_bands.shape[2]))
    
    # Anomaly (PCA)
    flat = sr_bands.reshape(sr_bands.shape[0], -1).T
    n_components = min(3, flat.shape[1]) if flat.shape[1] > 1 else 0
    if n_components >= 1:
        try:
            pca = PCA(n_components=n_components).fit_transform(flat)
            mean, cov_inv = np.mean(pca, axis=0), np.linalg.pinv(np.cov(pca.T))
            distances = np.array([np.sqrt((p-mean)@cov_inv@(p-mean)) for p in pca]).reshape(sr_bands.shape[1], sr_bands.shape[2])
            anomalies = distances > (np.mean(distances)+2*np.std(distances))
        except:
            anomalies = np.zeros((sr_bands.shape[1], sr_bands.shape[2]))
    else:
        anomalies = np.zeros((sr_bands.shape[1], sr_bands.shape[2]))
    
    return ndvi, anomalies

# ======================
# VLN PROCESSING FUNCTIONS
# ======================
def generate_blip_caption(image: Image.Image) -> str:
    inputs = blip_processor(images=image, return_tensors="pt").to(DEVICE)
    out = blip_model.generate(**inputs, max_length=50)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def metadata_to_text(metadata: dict) -> str:
    text = f"Image ID: {metadata['image_id']}. Bands: {', '.join(metadata['bands'])}. "
    text += "Features detected: "
    for f in metadata["features_detected"]:
        text += f"{f['type']} at {f['coordinates']} (confidence {f['confidence']}). "
    text += f"Classification: {metadata['classification_method']}."
    return text

def llm_generate(prompt: str) -> str:
    """Generate response using Claude Sonnet 4.5"""
    if not claude_client:
        return "Error: Claude API key not configured. Please set ANTHROPIC_API_KEY environment variable."
    
    try:
        message = claude_client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error calling Claude API: {str(e)}"

def vln_fusion_pipeline(image: Image.Image, metadata: dict):
    caption = generate_blip_caption(image)
    meta_text = metadata_to_text(metadata)

    prompt = f"""You are a planetary science AI. Based on the image caption and metadata,
write a detailed description suitable for 3D terrain analysis.

Image caption:
{caption}

Metadata:
{meta_text}

Respond with:
1. Scientific description (2–4 sentences)
2. Three bullet points for 3D reconstruction hints
3. A JSON with keys: semantic_description, 3D_inference_tags, provenance_notes"""

    fusion_output = llm_generate(prompt)

    result = {
        "image_id": metadata["image_id"],
        "blip_caption": caption,
        "metadata_text": meta_text,
        "fusion_output": fusion_output
    }

    # Extract JSON if present
    m = re.search(r"\{[\s\S]*\}", fusion_output)
    if m:
        try:
            result["fusion_json"] = json.loads(m.group(0))
        except:
            result["fusion_json_raw"] = m.group(0)

    return result

# ======================
# API ENDPOINTS
# ======================

@app.post("/process_tiff/")
async def process_tiff(file: UploadFile, zoom_factor: float = Form(0.8)):
    """Process TIFF file with zoom, enhancement, super-resolution, NDVI, and anomaly detection"""
    
    contents = await file.read()
    tiff_path = os.path.join(UPLOAD_DIR, "temp.tif")
    with open(tiff_path, "wb") as f: 
        f.write(contents)

    # Load TIFF
    with rasterio.open(tiff_path) as src:
        tiff_array = src.read()

    # Generate unique session ID for this processing
    session_id = os.path.splitext(file.filename)[0]
    session_dir = os.path.join(OUTPUT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Original PNG
    original_png = tiff_to_png(tiff_array)
    original_path = os.path.join(session_dir, "original.png")
    original_png.save(original_path)

    # Zoom
    zoomed = zoom_tiff(tiff_array, zoom_factor)
    zoomed_png = tiff_to_png(zoomed)
    zoomed_path = os.path.join(session_dir, "zoomed.png")
    zoomed_png.save(zoomed_path)

    # Normalize + Enhance
    normed = normalize_bands(zoomed)
    enhanced = enhance_bands(normed)
    enhanced_png = tiff_to_png(enhanced)
    enhanced_path = os.path.join(session_dir, "enhanced.png")
    enhanced_png.save(enhanced_path)

    # Super Resolution
    sr = super_resolve(enhanced)
    sr_png = tiff_to_png(sr)
    sr_path = os.path.join(session_dir, "super_res.png")
    sr_png.save(sr_path)

    # Features (NDVI & Anomalies)
    ndvi, anomalies = extract_features(sr)
    ndvi_img = Image.fromarray(((ndvi - ndvi.min())/(ndvi.max()-ndvi.min()+1e-8)*255).astype(np.uint8))
    ndvi_path = os.path.join(session_dir, "ndvi.png")
    ndvi_img.save(ndvi_path)
    
    anomalies_img = Image.fromarray((anomalies.astype(np.uint8)*255))
    anomalies_path = os.path.join(session_dir, "anomalies.png")
    anomalies_img.save(anomalies_path)

    return {
        "session_id": session_id,
        "files": {
            "original": f"/download/{session_id}/original.png",
            "zoomed": f"/download/{session_id}/zoomed.png",
            "enhanced": f"/download/{session_id}/enhanced.png",
            "super_resolution": f"/download/{session_id}/super_res.png",
            "ndvi": f"/download/{session_id}/ndvi.png",
            "anomalies": f"/download/{session_id}/anomalies.png"
        }
    }

@app.post("/vln_query/")
async def vln_query(
    file: UploadFile = File(...),
    query: str = Form(None)
):
    """Process image with VLN (Vision-Language-Navigation) pipeline using Claude"""
    
    # Open image directly with PIL
    image = Image.open(file.file).convert("RGB")

    # Save image to uploads
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    image.save(file_path)

    # Dummy metadata (replace with real API if available)
    metadata = {
        "image_id": os.path.splitext(file.filename)[0],
        "bands": ["RGB"],
        "features_detected": [
            {"type": "crater", "coordinates": [128, 128], "radius": 45, "confidence": 0.94}
        ],
        "classification_method": "unsupervised clustering"
    }

    # VLN fusion
    result = vln_fusion_pipeline(image, metadata)

    # Optional user query
    if query:
        user_prompt = f"{result['fusion_output']}\n\nUser Query: {query}\nAnswer concisely:"
        llm_answer = llm_generate(user_prompt)
        result["user_query"] = {"query": query, "answer": llm_answer}

    return JSONResponse(result)

@app.get("/download/{session_id}/{filename}")
async def download_file(session_id: str, filename: str):
    """Download processed images"""
    file_path = os.path.join(OUTPUT_DIR, session_id, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png", filename=filename)
    return JSONResponse({"error": "File not found"}, status_code=404)

# ======================
# HTML FRONTEND
# ======================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TIFF Processing & VLN with Claude Sonnet 4.5</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 1200px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            .section { background: #f5f5f5; padding: 20px; margin: 20px 0; border-radius: 8px; }
            input[type="file"], input[type="text"], input[type="number"] { 
                margin: 10px 0; padding: 8px; width: 100%; max-width: 400px; 
            }
            button { 
                background: #007bff; color: white; padding: 10px 20px; 
                border: none; border-radius: 4px; cursor: pointer; margin: 10px 5px;
            }
            button:hover { background: #0056b3; }
            .results { margin-top: 20px; }
            .image-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .image-item { text-align: center; }
            .image-item img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
            .image-item a { display: inline-block; margin-top: 10px; }
            .badge { 
                display: inline-block; 
                background: #28a745; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 4px; 
                font-size: 12px;
                margin-left: 10px;
            }
        </style>
    </head>
    <body>
        <h1>🛰️ TIFF Processing & VLN API <span class="badge">Claude Sonnet 4.5</span></h1>
        
        <div class="section">
            <h2>1. TIFF Processing (Zoom, Enhance, SR, NDVI, Anomalies)</h2>
            <form id="tiffForm">
                <input type="file" id="tiffFile" accept=".tif,.tiff" required/><br/>
                <label>Zoom Factor (0.1-1.0):</label>
                <input type="number" id="zoomFactor" step="0.1" min="0.1" max="1.0" value="0.8"/><br/>
                <button type="submit">Process TIFF</button>
            </form>
            <div id="tiffResults" class="results"></div>
        </div>

        <div class="section">
            <h2>2. VLN Image Query (Vision-Language-Navigation with Claude)</h2>
            <form id="vlnForm">
                <input type="file" id="vlnFile" accept="image/*" required/><br/>
                <input type="text" id="vlnQuery" placeholder="Ask a question about the image (optional)"/><br/>
                <button type="submit">Process VLN</button>
            </form>
            <div id="vlnResults" class="results"></div>
        </div>

        <script>
            // TIFF Processing
            document.getElementById('tiffForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('tiffFile').files[0]);
                formData.append('zoom_factor', document.getElementById('zoomFactor').value);
                
                document.getElementById('tiffResults').innerHTML = '<p>Processing...</p>';
                
                const response = await fetch('/process_tiff/', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                let html = '<h3>Results:</h3><div class="image-grid">';
                for (const [name, url] of Object.entries(data.files)) {
                    html += `
                        <div class="image-item">
                            <h4>${name.toUpperCase()}</h4>
                            <img src="${url}" alt="${name}"/>
                            <br/><a href="${url}" download>Download</a>
                        </div>
                    `;
                }
                html += '</div>';
                document.getElementById('tiffResults').innerHTML = html;
            };

            // VLN Processing
            document.getElementById('vlnForm').onsubmit = async (e) => {
                e.preventDefault();
                const formData = new FormData();
                formData.append('file', document.getElementById('vlnFile').files[0]);
                const query = document.getElementById('vlnQuery').value;
                if (query) formData.append('query', query);
                
                document.getElementById('vlnResults').innerHTML = '<p>Processing with Claude Sonnet 4.5...</p>';
                
                const response = await fetch('/vln_query/', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                let html = '<h3>VLN Results:</h3>';
                html += `<p><strong>BLIP Caption:</strong> ${data.blip_caption}</p>`;
                html += `<p><strong>Metadata:</strong> ${data.metadata_text}</p>`;
                html += `<p><strong>Claude Fusion Output:</strong></p><pre style="background:#fff;padding:15px;border-radius:4px;">${data.fusion_output}</pre>`;
                if (data.user_query) {
                    html += `<p><strong>Your Query:</strong> ${data.user_query.query}</p>`;
                    html += `<p><strong>Claude Answer:</strong> ${data.user_query.answer}</p>`;
                }
                document.getElementById('vlnResults').innerHTML = html;
            };
        </script>
    </body>
    </html>
    """

# ======================
# RUN SERVER
# ======================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*80)
    print("Starting server on http://127.0.0.1:8000")
    print("="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)