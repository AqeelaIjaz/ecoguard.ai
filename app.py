# ╔══════════════════════════════════════════════════════════════╗
# ║  EcoGuard.AI — Hugging Face Spaces                          ║
# ║  Smart Pest & Pollinator Identification System              ║
# ║  Languages: English | اردو | 中文 | हिंदी | 한국어           ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  FILES NEEDED IN YOUR HUGGING FACE SPACE:                  ║
# ║  1. app.py              ← this file                        ║
# ║  2. requirements.txt    ← the other file                   ║
# ║  3. ecoguard_model.pth  ← your trained model               ║
# ║  4. class_labels.json   ← your class labels                ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  TO GET ecoguard_model.pth and class_labels.json:          ║
# ║  Run this in your Colab/Kaggle notebook:                   ║
# ║                                                             ║
# ║  # Find your files                                         ║
# ║  import os                                                  ║
# ║  for root,dirs,files in os.walk("/content"):               ║
# ║      for f in files:                                       ║
# ║          if f.endswith(".pth") or f=="class_labels.json":  ║
# ║              print(os.path.join(root,f))                   ║
# ║                                                             ║
# ║  # Download them                                           ║
# ║  from google.colab import files                            ║
# ║  files.download("path/shown/above/ecoguard_model.pth")     ║
# ║  files.download("path/shown/above/class_labels.json")      ║
# ╠══════════════════════════════════════════════════════════════╣
# ║  OPTIONAL: Add Groq AI key for AI advice                   ║
# ║  Hugging Face Space → Settings → Variables and secrets     ║
# ║  Name: GROQ_API_KEY    Value: gsk_your_key_here            ║
# ║  Get free key: console.groq.com                            ║
# ╚══════════════════════════════════════════════════════════════╝

import os
import json
import warnings
import requests
import numpy as np
import cv2
import torch
import torch.nn as nn
import timm
import folium
import pandas as pd
import tempfile
from PIL import Image
from torchvision import transforms
from folium.plugins import MarkerCluster
from datetime import datetime
import gradio as gr

warnings.filterwarnings("ignore")

# ── Groq API key (set as Space Secret in HF Settings) ────────
GROQ_API_KEY   = os.environ.get("GROQ_API_KEY", "")
GROQ_AVAILABLE = bool(GROQ_API_KEY)

# ════════════════════════════════════════════════════════════
#  LANGUAGE SYSTEM
#  5 languages: English | اردو Urdu | 中文 Chinese | हिंदी Hindi | 한국어 Korean
# ════════════════════════════════════════════════════════════

LANG_NAMES = {
    "en": "English",
    "ur": "اردو",
    "zh": "中文",
    "hi": "हिंदी",
    "ko": "한국어",
}

ROLE_LABELS = {
    "en": {
        "pest":                    "Pest",
        "pollinator":              "Pollinator",
        "predator":                "Predator (Beneficial)",
        "biological_control_agent":"Biological Control Agent",
        "decomposer":              "Decomposer",
        "invasive":                "Invasive Species",
        "neutral":                 "Neutral Insect",
        "unknown":                 "Unknown — Low Confidence",
    },
    "ur": {
        "pest":                    "کیڑا آفت",
        "pollinator":              "جرگ آور",
        "predator":                "شکاری (فائدہ مند)",
        "biological_control_agent":"حیاتیاتی کنٹرول ایجنٹ",
        "decomposer":              "سڑنے والا",
        "invasive":                "حملہ آور نوع",
        "neutral":                 "غیر جانبدار کیڑا",
        "unknown":                 "نامعلوم",
    },
    "zh": {
        "pest":                    "害虫",
        "pollinator":              "授粉者",
        "predator":                "天敌（有益）",
        "biological_control_agent":"生物防治天敌",
        "decomposer":              "分解者",
        "invasive":                "入侵种",
        "neutral":                 "中性昆虫",
        "unknown":                 "未知",
    },
    "hi": {
        "pest":                    "कीट",
        "pollinator":              "परागणकर्ता",
        "predator":                "शिकारी (लाभकारी)",
        "biological_control_agent":"जैविक नियंत्रण एजेंट",
        "decomposer":              "विघटनकारी",
        "invasive":                "आक्रामक प्रजाति",
        "neutral":                 "तटस्थ कीट",
        "unknown":                 "अज्ञात",
    },
    "ko": {
        "pest":                    "해충",
        "pollinator":              "수분 매개자",
        "predator":                "포식자 (유익)",
        "biological_control_agent":"생물학적 방제 천적",
        "decomposer":              "분해자",
        "invasive":                "침입종",
        "neutral":                 "중립 곤충",
        "unknown":                 "미확인",
    },
}

REC_TITLES = {
    "en": {
        "pest":                    "Pest Detected — Apply IPM Protocol",
        "pollinator":              "Pollinator Detected — Protect and Conserve",
        "predator":                "Beneficial Predator — Do NOT Spray",
        "biological_control_agent":"Biological Control Agent — Extremely Valuable",
        "decomposer":              "Decomposer Detected — Soil Health Indicator",
        "invasive":                "INVASIVE SPECIES ALERT — Report Immediately",
        "neutral":                 "Neutral Insect — No Action Required",
        "unknown":                 "Identification Uncertain — Expert Review Required",
    },
    "ur": {
        "pest":                    "کیڑا پایا گیا — IPM اپنائیں",
        "pollinator":              "جرگ آور پایا گیا — تحفظ ضروری",
        "predator":                "فائدہ مند شکاری — اسپرے نہ کریں",
        "biological_control_agent":"حیاتیاتی کنٹرول — انتہائی قیمتی",
        "decomposer":              "سڑنے والا پایا گیا — مٹی کا اشارہ",
        "invasive":                "حملہ آور الرٹ — فوری اطلاع دیں",
        "neutral":                 "غیر جانبدار — کوئی کارروائی نہیں",
        "unknown":                 "غیر یقینی — ماہر کی ضرورت",
    },
    "zh": {
        "pest":                    "发现害虫——应用IPM协议",
        "pollinator":              "发现授粉者——保护和保育",
        "predator":                "有益天敌——禁止施药",
        "biological_control_agent":"生物防治天敌——极其宝贵",
        "decomposer":              "发现分解者——土壤健康指标",
        "invasive":                "入侵种警报——立即上报",
        "neutral":                 "中性昆虫——无需行动",
        "unknown":                 "识别不确定——需要专家",
    },
    "hi": {
        "pest":                    "कीट मिला — IPM प्रोटोकॉल लागू करें",
        "pollinator":              "परागणकर्ता मिला — सुरक्षा जरूरी",
        "predator":                "लाभकारी शिकारी — स्प्रे न करें",
        "biological_control_agent":"जैविक नियंत्रण — अत्यंत मूल्यवान",
        "decomposer":              "विघटनकारी मिला — मिट्टी स्वास्थ्य",
        "invasive":                "आक्रामक प्रजाति अलर्ट — रिपोर्ट करें",
        "neutral":                 "तटस्थ कीट — कोई कार्रवाई नहीं",
        "unknown":                 "पहचान अनिश्चित — विशेषज्ञ चाहिए",
    },
    "ko": {
        "pest":                    "해충 발견 — IPM 프로토콜 적용",
        "pollinator":              "수분 매개자 발견 — 보호 필요",
        "predator":                "유익한 포식자 — 농약 절대 금지",
        "biological_control_agent":"생물학적 방제 천적 — 매우 귀중",
        "decomposer":              "분해자 발견 — 토양 건강 지표",
        "invasive":                "침입종 경보 — 즉시 신고",
        "neutral":                 "중립 곤충 — 조치 불필요",
        "unknown":                 "식별 불확실 — 전문가 검토 필요",
    },
}

IPM_STEPS = {
    "pest": [
        "Count pests first — act only if above economic injury threshold",
        "Protect natural enemies already present: ladybirds, lacewings, parasitic wasps",
        "Use physical traps: yellow sticky traps for aphids, pheromone traps for moths",
        "Cultural controls: crop rotation, resistant varieties, remove crop debris",
        "Neem oil (azadirachtin) as first spray option if threshold exceeded",
        "Chemical pesticides: last resort only — spray at dusk to protect pollinators",
    ],
    "pollinator": [
        "Do not spray near this insect — especially avoid neonicotinoids",
        "Plant wildflower strips along field edges to provide habitat",
        "Provide shallow water dishes with pebbles near the field",
        "If spraying is unavoidable, apply only at dusk or early dawn",
        "Report significant population declines to your extension officer",
    ],
    "predator": [
        "Establish a no-spray zone where this predator is active",
        "Provide habitat: leaf litter, hedgerows, insectary plants (dill, fennel, yarrow)",
        "Allow 1-2 weeks for the predator population to reduce pest numbers",
        "Predatory beetles and lacewings are available commercially if needed",
    ],
    "biological_control_agent": [
        "Absolute no-spray zone — organic pesticides also harm these insects",
        "Plant insectary plants: dill, fennel, carrot flowers, sweet alyssum, phacelia",
        "Commercial BCAs available: Trichogramma wasps, Aphidius colemani, Encarsia formosa",
        "Record and submit this sighting to iNaturalist — it is scientifically valuable",
    ],
    "decomposer": [
        "Leave crop residues on the soil surface where possible",
        "Minimise deep tillage — it destroys decomposer communities",
        "Avoid heavy machinery on wet soil to prevent compaction",
        "Use soil fungicides only when absolutely essential",
    ],
    "invasive": [
        "Take clear photographs and record GPS coordinates, date, count, crop affected",
        "Report to your National Plant Protection Organization (NPPO) TODAY",
        "Do not move plants, soil, or farm equipment from the affected field",
        "Ask your extension officer about approved emergency control methods",
        "Scout your entire farm weekly and alert neighbouring farmers",
    ],
    "neutral": [
        "Continue your normal field scouting schedule",
        "Record in your farm biodiversity log",
        "No pesticide needed — do not spray",
    ],
    "unknown": [
        "Retake the photograph in good natural light, closer to the insect",
        "Submit to iNaturalist.org for free community identification",
        "Contact your agricultural extension worker for in-person assessment",
        "Do not spray — the insect may be a beneficial predator or pollinator",
    ],
}

IPM_WARNINGS = {
    "pest":                    "Never use broad-spectrum insecticides — they kill ladybirds, lacewings, and bees that control pests for free.",
    "pollinator":              "Pollinator populations have declined 30-50% globally. Farms with pollinator habitat produce 20-40% higher yields.",
    "predator":                "One adult ladybird eats up to 5,000 aphids in its lifetime. Killing predators increases your pesticide costs.",
    "biological_control_agent":"Finding this insect in your field means your ecosystem is functioning well. Protect it at all costs.",
    "decomposer":              "It takes approximately 500 years to form 1 centimetre of topsoil. Decomposers are the engine of this process.",
    "invasive":                "CRITICAL: Every day of delay increases the spread risk. Report immediately.",
    "neutral":                 "More than 90% of insect species are neutral or beneficial. Most insects on your farm are not your enemies.",
    "unknown":                 "Never apply pesticides based on an uncertain identification. Wait for confirmation.",
}

RISK_LEVELS = {
    "pest":                    "HIGH",
    "pollinator":              "LOW",
    "predator":                "LOW",
    "biological_control_agent":"LOW",
    "decomposer":              "LOW",
    "invasive":                "CRITICAL",
    "neutral":                 "LOW",
    "unknown":                 "UNCERTAIN",
}

ROLE_SPECIES = {
    "pest":                    "Aphis gossypii",
    "pollinator":              "Apis mellifera",
    "predator":                "Coccinella septempunctata",
    "biological_control_agent":"Trichogramma evanescens",
    "decomposer":              "Geotrupes stercorarius",
    "invasive":                "Halyomorpha halys",
    "neutral":                 "Chironomus riparius",
}


# ════════════════════════════════════════════════════════════
#  IMAGE PROCESSING PIPELINE
#  Same 5-step pipeline used during training
# ════════════════════════════════════════════════════════════

def process_image(img_array: np.ndarray) -> np.ndarray:
    """Apply CLAHE, white balance, background suppression, sharpening, saturation."""
    bgr = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Step 1: CLAHE contrast (LAB space — does not change colours)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    bgr = cv2.cvtColor(cv2.merge([clahe.apply(L), a, b]), cv2.COLOR_LAB2BGR)

    # Step 2: White balance — removes green leaf colour cast
    f = bgr.astype(np.float32)
    means = [np.mean(f[:, :, c]) for c in range(3)]
    overall = sum(means) / 3.0
    scales = [overall / m if m > 1 else 1.0 for m in means]
    corr = np.stack([np.clip(f[:, :, c] * scales[c], 0, 255) for c in range(3)], axis=2)
    bgr = np.clip(0.4 * f + 0.6 * corr, 0, 255).astype(np.uint8)

    # Step 3: Background suppression
    blur_k = 21
    bg = cv2.GaussianBlur(bgr, (blur_k, blur_k), 0)
    diff = cv2.absdiff(bgr, bg)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    fg = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (15, 15), 0)
    bgr = np.clip(bgr.astype(np.float32) * (1.0 - 0.25 * (1 - fg)[..., np.newaxis]),
                  0, 255).astype(np.uint8)

    # Step 4: Sharpening
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=2.0)
    bgr = np.clip(cv2.addWeighted(bgr, 1.4, blurred, -0.4, 0), 0, 255).astype(np.uint8)

    # Step 5: Saturation boost ×1.15
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return bgr


# ════════════════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════════════════

def load_model():
    """Load trained model and class labels from disk."""
    with open("class_labels.json") as f:
        data = json.load(f)

    num_classes  = data["num_classes"]
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
    model_choice = data.get("model_choice", "efficientnet")
    img_size     = data.get("img_size", 260)
    device       = "cuda" if torch.cuda.is_available() else "cpu"

    if model_choice == "cnn":
        class InsectCNN(nn.Module):
            def __init__(self, nc):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
                    nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2),nn.Dropout2d(0.15),
                    nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                    nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2),nn.Dropout2d(0.20),
                    nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                    nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2),nn.Dropout2d(0.25),
                    nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                    nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2),nn.Dropout2d(0.25),
                    nn.Conv2d(256,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                    nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                    nn.MaxPool2d(2,2),nn.Dropout2d(0.30),
                    nn.Conv2d(512,512,3,padding=1),nn.BatchNorm2d(512),nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4,4)),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),nn.Linear(512*4*4,1024),nn.ReLU(inplace=True),
                    nn.Dropout(0.5),nn.Linear(1024,256),nn.ReLU(inplace=True),
                    nn.Dropout(0.4),nn.Linear(256,nc),
                )
            def forward(self, x):
                return self.classifier(self.features(x))

        model = InsectCNN(num_classes).to(device)
    else:
        model = timm.create_model(
            "efficientnet_b2", pretrained=False, num_classes=num_classes
        ).to(device)

    ckpt = torch.load("ecoguard_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    saved_acc = ckpt.get("val_acc", 0)

    return model, idx_to_class, num_classes, img_size, device, saved_acc, model_choice


MODEL, IDX2CLS, NUM_CLASSES, IMG_SIZE, DEVICE, SAVED_ACC, MODEL_CHOICE = load_model()

INFER_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

sightings = []


# ════════════════════════════════════════════════════════════
#  PREDICTION
# ════════════════════════════════════════════════════════════

@torch.no_grad()
def predict(image_np: np.ndarray):
    if image_np is None:
        return None, None, None
    proc_bgr = process_image(image_np)
    proc_rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(proc_rgb)
    tensor   = INFER_TF(pil_img).unsqueeze(0).to(DEVICE)
    probs    = torch.softmax(MODEL(tensor), dim=1)[0]
    conf, pidx = probs.max(0)
    pred = IDX2CLS[pidx.item()]
    top5_v, top5_i = torch.topk(probs, min(5, NUM_CLASSES))
    top5 = [(IDX2CLS[i.item()], v.item()) for v, i in zip(top5_v, top5_i)]
    return pred, conf.item(), top5


# ════════════════════════════════════════════════════════════
#  GROQ AI
# ════════════════════════════════════════════════════════════

ECO_SYSTEM = (
    "You are an expert agricultural entomologist and IPM advisor. "
    "Give practical non-chemical-first advice for sustainable farming. "
    "Be concise (under 180 words), factual, and easy for farmers to understand."
)


def call_groq(user_msg: str, max_tokens: int = 320) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": ECO_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


# ════════════════════════════════════════════════════════════
#  BIODIVERSITY APIs (iNaturalist + GBIF — both free)
# ════════════════════════════════════════════════════════════

def inat_lookup(species_name: str) -> dict:
    if not species_name:
        return {}
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/taxa",
            params={"q": species_name, "per_page": 1, "order_by": "observations_count"},
            timeout=8,
        )
        res = r.json().get("results", [])
        if not res:
            return {}
        t = res[0]
        return {
            "name":    t.get("name", ""),
            "common":  t.get("preferred_common_name", ""),
            "obs":     t.get("observations_count", 0),
            "wiki":    t.get("wikipedia_url", ""),
        }
    except Exception:
        return {}


def gbif_lookup(species_name: str) -> dict:
    if not species_name or len(species_name.split()) < 2:
        return {}
    try:
        r = requests.get(
            "https://api.gbif.org/v1/species/match",
            params={"name": species_name, "kingdom": "Animalia"},
            timeout=8,
        )
        d = r.json()
        if d.get("matchType") == "NONE":
            return {}
        return {"order": d.get("order", ""), "family": d.get("family", "")}
    except Exception:
        return {}


def inat_nearby_invasives(lat: float, lon: float, radius_km: int = 100) -> list:
    try:
        r = requests.get(
            "https://api.inaturalist.org/v1/observations",
            params={
                "lat": lat, "lng": lon, "radius": radius_km,
                "quality_grade": "research", "photos": "true",
                "taxon_name": "Insecta", "introduced": "true",
                "per_page": 15, "order_by": "observed_on",
            },
            timeout=12,
        )
        return [
            {
                "taxon":  (o.get("taxon") or {}).get("name", "Unknown"),
                "common": (o.get("taxon") or {}).get("preferred_common_name", ""),
                "date":   o.get("observed_on", ""),
                "place":  o.get("place_guess", ""),
            }
            for o in r.json().get("results", [])
        ]
    except Exception:
        return []


# ════════════════════════════════════════════════════════════
#  RECOMMENDATION BUILDER
# ════════════════════════════════════════════════════════════

def get_recommendation(role: str, confidence: float, lang: str = "en") -> str:
    if confidence < 0.45:
        role = "unknown"
    titles   = REC_TITLES.get(lang, REC_TITLES["en"])
    title    = titles.get(role, titles["unknown"])
    steps    = IPM_STEPS.get(role, IPM_STEPS["unknown"])
    warning  = IPM_WARNINGS.get(role, IPM_WARNINGS["unknown"])
    lines = [f"## {title}", "\n### Recommended Actions"]
    for i, step in enumerate(steps, 1):
        lines.append(f"\n**{i}.** {step}")
    lines.append(f"\n\n---\n**Important:** {warning}")
    return "\n".join(lines)


def lang_code(lang_str: str) -> str:
    """Extract language code from dropdown value like 'English (EN)'."""
    for code, name in LANG_NAMES.items():
        if name in lang_str or code.upper() in lang_str:
            return code
    return "en"


# ════════════════════════════════════════════════════════════
#  MAP BUILDER
# ════════════════════════════════════════════════════════════

def build_sightings_map(center_lat=None, center_lon=None) -> str:
    lat0 = center_lat or 30.0
    lon0 = center_lon or 70.0
    if sightings:
        lat0 = float(np.mean([s["lat"] for s in sightings]))
        lon0 = float(np.mean([s["lon"] for s in sightings]))
    m       = folium.Map(location=[lat0, lon0], zoom_start=6, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)
    COLORS  = {
        "pest": "red", "pollinator": "orange", "predator": "green",
        "biological_control_agent": "darkgreen", "decomposer": "purple",
        "invasive": "darkred", "neutral": "gray", "unknown": "lightgray",
    }
    for s in sightings:
        popup = (
            f"<div style='font-family:sans-serif;font-size:13px;'>"
            f"<b>{s['role'].replace('_', ' ').title()}</b><br>"
            f"Confidence: {s['confidence']}%<br>"
            f"Date: {s['timestamp']}</div>"
        )
        folium.Marker(
            [s["lat"], s["lon"]],
            popup=folium.Popup(popup, max_width=220),
            icon=folium.Icon(color=COLORS.get(s["role"], "blue"), icon="bug", prefix="fa"),
            tooltip=s["role"].replace("_", " ").title(),
        ).add_to(cluster)
        if s["role"] == "invasive":
            folium.Circle(
                [s["lat"], s["lon"]], radius=5000,
                color="darkred", fill=True, fill_opacity=0.08,
                tooltip="Invasive species risk zone — 5 km",
            ).add_to(m)
    return m._repr_html_()


# ════════════════════════════════════════════════════════════
#  APP HANDLER FUNCTIONS
# ════════════════════════════════════════════════════════════

def predict_field(image, crop, location, season, lat_str, lon_str, lang_str):
    lang = lang_code(lang_str)
    roles = ROLE_LABELS.get(lang, ROLE_LABELS["en"])

    if image is None:
        return "*Upload a photo and click Identify Insect*", "", "", "", "", ""

    pred, conf, top5 = predict(image)
    if pred is None:
        return "Prediction failed — please try again.", "", "", "", "", ""

    role_label = roles.get(pred, pred)
    risk       = RISK_LEVELS.get(pred, "UNCERTAIN")

    class_md = f"## {role_label}"
    if conf < 0.45:
        class_md += "\n\n> **Low confidence** — please retake in better light, closer to the insect."

    risk_display = {
        "HIGH":      "HIGH — Crop Pest",
        "CRITICAL":  "CRITICAL — Invasive Species",
        "LOW":       "LOW — Beneficial Insect",
        "UNCERTAIN": "UNCERTAIN — Expert Review Needed",
    }.get(risk, risk)

    recommendation = get_recommendation(pred, conf, lang)

    ai_advice = ""
    if GROQ_AVAILABLE:
        result = call_groq(
            f"Farmer found: {pred.replace('_', ' ')} — {risk} risk\n"
            f"Crop: {crop or 'not specified'}\n"
            f"Location: {location or 'not specified'}\n"
            "Give: 1) Brief ecological context 2) Key action 3) Any urgent warning. Max 150 words."
        )
        if result:
            ai_advice = f"### AI Advice (Groq)\n\n{result}"
    else:
        ai_advice = "*Add GROQ_API_KEY as a Space Secret for AI-powered advice (free at console.groq.com)*"

    map_html = ""
    if lat_str and lon_str:
        try:
            lat, lon = float(lat_str), float(lon_str)
            sightings.append({
                "role":       pred,
                "confidence": round(conf * 100, 1),
                "lat":        lat,
                "lon":        lon,
                "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
            map_html = build_sightings_map(lat, lon)
        except ValueError:
            pass

    return class_md, f"{conf*100:.1f}%", risk_display, recommendation, ai_advice, map_html


def predict_expert(image, crop, lat_str, lon_str, notes, farm_type, organic, lang_str):
    lang  = lang_code(lang_str)
    roles = ROLE_LABELS.get(lang, ROLE_LABELS["en"])

    if image is None:
        return "Please upload an insect photograph.", "", "", "", ""

    pred, conf, top5 = predict(image)
    if pred is None:
        return "Prediction failed.", "", "", "", ""

    conf_note = (
        "High — confident result"   if conf >= 0.70 else
        "Moderate — verify result"  if conf >= 0.45 else
        "Low — retake photograph"
    )

    report = (
        f"## Expert Analysis Report\n\n"
        f"**{datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}**\n\n---\n\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| **Predicted Role** | {roles.get(pred, pred)} |\n"
        f"| **Confidence** | {conf*100:.2f}% — {conf_note} |\n"
        f"| **Model** | {MODEL_CHOICE.upper()} |\n"
        f"| **Risk Level** | {RISK_LEVELS.get(pred, '—')} |\n"
        f"| **Crop** | {crop or '—'} |\n"
        f"| **Farm Type** | {farm_type or '—'} |\n"
        f"| **Organic** | {'Yes' if organic else 'No'} |\n\n"
        f"**Field Notes:** {notes.strip() if notes.strip() else '—'}"
    )

    top5_lines = [
        "## Top-5 Model Predictions\n",
        "| Rank | Role | Confidence |",
        "|------|------|------------|",
    ]
    for rank, (cls, c) in enumerate(top5, 1):
        bold = "**" if rank == 1 else ""
        top5_lines.append(
            f"| {rank} | {bold}{roles.get(cls, cls)}{bold} "
            f"| {bold}{c*100:.2f}%{bold} |"
        )
    if len(top5) >= 2 and (top5[0][1] - top5[1][1]) < 0.05:
        top5_lines.append(
            "\n> Top two predictions are very close. "
            "A clearer photograph would improve accuracy."
        )
    top5_md = "\n".join(top5_lines)

    species = ROLE_SPECIES.get(pred) if conf >= 0.55 else None
    bio_lines = ["## Biodiversity Profile\n"]
    if species:
        inat = inat_lookup(species)
        if inat and inat.get("name"):
            bio_lines += [
                "### iNaturalist",
                f"- **Scientific name:** *{inat['name']}*",
                f"- **Common name:** {inat.get('common', '—')}",
                f"- **Verified observations worldwide:** {inat.get('obs', 0):,}",
            ]
            if inat.get("wiki"):
                bio_lines.append(f"- [View Wikipedia entry]({inat['wiki']})")
        gbif = gbif_lookup(species)
        if gbif and gbif.get("order"):
            bio_lines += [
                "\n### GBIF Taxonomy",
                f"- **Order:** {gbif.get('order', '—')}",
                f"- **Family:** {gbif.get('family', '—')}",
            ]
    else:
        bio_lines.append(
            f"> Confidence is {conf*100:.1f}% — "
            "species-level data not shown below 55% confidence."
        )
    bio_md = "\n".join(bio_lines)

    ai_plan = ""
    if GROQ_AVAILABLE:
        result = call_groq(
            f"Expert analysis: {pred} found in {crop or 'unspecified crop'}.\n"
            f"Confidence: {conf*100:.0f}%. Risk: {RISK_LEVELS.get(pred, '?')}.\n"
            "Give: 1) Ecological context 2) IPM action plan 3) Monitoring advice. Max 180 words."
        )
        if result:
            ai_plan = f"### AI Management Plan\n\n{result}"
    if not ai_plan:
        ai_plan = get_recommendation(pred, conf, lang)

    map_html = ""
    try:
        lat, lon = float(lat_str), float(lon_str)
        sightings.append({
            "role":       pred,
            "confidence": round(conf * 100, 1),
            "lat":        lat,
            "lon":        lon,
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M"),
        })
        map_html = build_sightings_map(lat, lon)
    except Exception:
        pass

    return report, top5_md, bio_md, ai_plan, map_html


def identify_from_text(description, crop, location, lang_str):
    if not description.strip():
        return "*Please describe the insect.*"
    if not GROQ_AVAILABLE:
        return (
            "**Groq AI is not configured.**\n\n"
            "To enable this tab:\n"
            "1. Go to console.groq.com → Sign up free\n"
            "2. Create an API key\n"
            "3. In your Hugging Face Space: Settings → Variables and secrets\n"
            "4. Add: Name = `GROQ_API_KEY`, Value = your key"
        )
    result = call_groq(
        f'Identify this insect: "{description}"\n'
        f"Crop: {crop or 'not specified'}\n"
        f"Location: {location or 'not specified'}\n\n"
        "Provide:\n"
        "1. Most likely identification (common + scientific name)\n"
        "2. Agricultural role (pest/pollinator/predator/etc.)\n"
        "3. Most important action for the farmer\n"
        "4. Any invasive species concern\n"
        "Max 200 words.",
        max_tokens=350,
    )
    return result or "*Identification failed — please try again.*"


def scan_for_invasives(lat_str, lon_str, lang_str):
    try:
        lat, lon = float(lat_str), float(lon_str)
    except ValueError:
        return "Please enter valid decimal coordinates (e.g. 31.5204 for latitude)."

    results = inat_nearby_invasives(lat, lon, radius_km=100)
    if not results:
        return (
            "**No invasive insect sightings found** within 100 km of your location "
            "in iNaturalist research-grade data.\n\n"
            "Continue regular field scouting and report any unusual insects to your extension officer."
        )

    lines = [
        f"## Invasive Species Detected Within 100 km ({len(results)} records)\n",
        "*Source: iNaturalist — research-grade, community-verified observations*\n",
        "| Species | Common Name | Last Observed | Location |",
        "|---------|------------|--------------|----------|",
    ]
    for a in results[:12]:
        lines.append(
            f"| *{a['taxon']}* | {a['common']} | {a['date']} | {a['place']} |"
        )
    lines.append(
        "\n\n**Action Required:** If any of these match insects on your farm, "
        "contact your National Plant Protection Organization (NPPO) or "
        "agricultural extension office immediately."
    )
    return "\n".join(lines)


def refresh_map():
    return build_sightings_map()


def export_sightings_csv():
    if not sightings:
        return None
    df  = pd.DataFrame(sightings)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    df.to_csv(tmp.name, index=False)
    return tmp.name


# ════════════════════════════════════════════════════════════
#  GRADIO UI
# ════════════════════════════════════════════════════════════

HEADER_HTML = f"""
<div style="background:#FFFFFF;border-bottom:3px solid #1A7A4A;
     padding:18px 24px;display:flex;align-items:flex-start;
     justify-content:space-between;flex-wrap:wrap;gap:16px;
     font-family:'Segoe UI',system-ui,sans-serif;">
  <div style="flex:1;min-width:260px;">
    <span style="background:#1A7A4A;color:#fff;font-size:12px;
          font-weight:700;padding:3px 9px;border-radius:4px;">
      EcoGuard.AI
    </span>
    <h1 style="font-size:19px;font-weight:600;color:#1A1C1E;margin:6px 0 3px;">
      Smart Pest &amp; Pollinator Identification System
    </h1>
    <p style="font-size:12px;color:#6C757D;margin:0 0 10px;line-height:1.5;">
      AI-powered insect identification for farmers, extension workers, and agronomists.
      Upload an insect photo to identify its agricultural role and get evidence-based IPM advice.
    </p>
    <div style="display:flex;flex-wrap:wrap;gap:5px;">
      <span style="background:#E8F5EE;color:#1A7A4A;border:1px solid #B7DFC8;
            font-size:10px;font-weight:600;padding:2px 8px;border-radius:3px;">SDG 2 — Zero Hunger</span>
      <span style="background:#E8F5EE;color:#1A7A4A;border:1px solid #B7DFC8;
            font-size:10px;font-weight:600;padding:2px 8px;border-radius:3px;">SDG 12 — Responsible Production</span>
      <span style="background:#E8F5EE;color:#1A7A4A;border:1px solid #B7DFC8;
            font-size:10px;font-weight:600;padding:2px 8px;border-radius:3px;">SDG 15 — Life on Land</span>
      <span style="background:#E8F5EE;color:#1A7A4A;border:1px solid #B7DFC8;
            font-size:10px;font-weight:600;padding:2px 8px;border-radius:3px;">SDG 1 — No Poverty</span>
    </div>
  </div>
  <div style="background:#F8F9FA;border:1px solid #DEE2E6;border-radius:8px;
              padding:12px 16px;min-width:210px;">
    <div style="font-size:10px;font-weight:700;color:#6C757D;
                text-transform:uppercase;letter-spacing:.06em;margin-bottom:8px;">
      System Status
    </div>
    <div style="font-size:12px;display:flex;justify-content:space-between;
                padding:4px 0;border-bottom:1px solid #DEE2E6;">
      <span style="color:#495057;">Model</span>
      <span style="color:#1A1C1E;font-family:monospace;">{MODEL_CHOICE.upper()}</span>
    </div>
    <div style="font-size:12px;display:flex;justify-content:space-between;
                padding:4px 0;border-bottom:1px solid #DEE2E6;">
      <span style="color:#495057;">Val Accuracy</span>
      <span style="color:{'#1A7A4A' if SAVED_ACC >= 60 else '#D97706'};font-weight:600;">
        {SAVED_ACC:.1f}%
      </span>
    </div>
    <div style="font-size:12px;display:flex;justify-content:space-between;
                padding:4px 0;border-bottom:1px solid #DEE2E6;">
      <span style="color:#495057;">Classes</span>
      <span style="color:#1A1C1E;">{NUM_CLASSES} roles</span>
    </div>
    <div style="font-size:12px;display:flex;justify-content:space-between;
                padding:4px 0;border-bottom:1px solid #DEE2E6;">
      <span style="color:#495057;">Groq AI</span>
      <span style="color:{'#1A7A4A' if GROQ_AVAILABLE else '#D97706'};">
        {'Active' if GROQ_AVAILABLE else 'Add key in Settings'}
      </span>
    </div>
    <div style="font-size:12px;display:flex;justify-content:space-between;padding:4px 0;">
      <span style="color:#495057;">Data APIs</span>
      <span style="color:#1A7A4A;">iNat · GBIF ✓</span>
    </div>
  </div>
</div>
"""

APP_CSS = """
body, .gradio-container {
    background-color: #F1F3F5 !important;
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}
.tab-nav {
    background-color: #FFFFFF !important;
    border-bottom: 2px solid #1A7A4A !important;
    padding: 0 14px !important;
}
.tab-nav button {
    color: #6C757D !important;
    font-size: 13px !important;
    padding: 11px 15px !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    background: transparent !important;
    border-radius: 0 !important;
}
.tab-nav button.selected {
    color: #1A7A4A !important;
    border-bottom: 3px solid #1A7A4A !important;
    font-weight: 600 !important;
}
.tabitem {
    background-color: #F8F9FA !important;
    padding: 20px !important;
}
.block {
    background-color: #FFFFFF !important;
    border: 1px solid #DEE2E6 !important;
    border-radius: 8px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,.05) !important;
}
label span, .label-wrap span {
    color: #6C757D !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: .05em !important;
}
input[type=text], textarea, select {
    background-color: #FFFFFF !important;
    border: 1px solid #DEE2E6 !important;
    border-radius: 6px !important;
    color: #1A1C1E !important;
    font-size: 14px !important;
}
input[type=text]:focus, textarea:focus {
    border-color: #1A7A4A !important;
    box-shadow: 0 0 0 3px rgba(26,122,74,.12) !important;
}
button.primary {
    background-color: #1A7A4A !important;
    border: none !important;
    border-radius: 6px !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}
button.secondary {
    background-color: #FFFFFF !important;
    border: 1px solid #1A7A4A !important;
    border-radius: 6px !important;
    color: #1A7A4A !important;
}
.prose p, .markdown-body p,
.prose li, .markdown-body li { color: #1A1C1E !important; }
.prose h1, .prose h2, .prose h3,
.markdown-body h1, .markdown-body h2, .markdown-body h3 { color: #1A7A4A !important; }
.prose strong, .markdown-body strong { color: #1A1C1E !important; }
.prose table th, .markdown-body table th {
    background: #F1F3F5 !important;
    color: #1A1C1E !important;
}
.prose table td, .markdown-body table td { color: #1A1C1E !important; }
.prose blockquote, .markdown-body blockquote {
    border-left: 4px solid #1A7A4A !important;
    background: #E8F5EE !important;
    color: #1A1C1E !important;
}
"""

LANG_CHOICES = [f"{name} ({code.upper()})" for code, name in LANG_NAMES.items()]

with gr.Blocks(
    title="EcoGuard.AI — Insect Identification System",
    css=APP_CSS,
    theme=gr.themes.Default(
        primary_hue=gr.themes.colors.green,
        font=gr.themes.GoogleFont("Inter"),
    ),
) as app:

    gr.HTML(HEADER_HTML)

    lang_selector = gr.Dropdown(
        label="Language / زبان / 语言 / भाषा / 언어",
        choices=LANG_CHOICES,
        value="English (EN)",
    )

    with gr.Tabs():

        # ── Tab 1: Field Identification ──────────────────────
        with gr.Tab("Field Identification"):
            gr.Markdown(
                "Upload a clear photo of the insect found in your field. "
                "The system identifies it and provides management recommendations."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t1_img    = gr.Image(label="Insect photograph", type="numpy", height=275)
                    t1_crop   = gr.Textbox(label="Crop affected (optional)",
                                           placeholder="e.g. Maize, Tomato, Cotton, Wheat")
                    t1_loc    = gr.Textbox(label="Location (optional)",
                                           placeholder="e.g. Punjab, Pakistan")
                    t1_season = gr.Textbox(label="Growing season (optional)",
                                           placeholder="e.g. Kharif, Rabi, dry season")
                    with gr.Row():
                        t1_lat = gr.Textbox(label="Latitude (optional)",
                                            placeholder="e.g. 31.5204")
                        t1_lon = gr.Textbox(label="Longitude (optional)",
                                            placeholder="e.g. 74.3587")
                    t1_btn = gr.Button("Identify Insect", variant="primary", size="lg")
                with gr.Column(scale=1):
                    t1_class = gr.Markdown("*Upload a photo and click Identify Insect*")
                    t1_conf  = gr.Textbox(label="Model confidence",    interactive=False)
                    t1_risk  = gr.Textbox(label="Agricultural risk",   interactive=False)
            t1_rec = gr.Markdown(label="Management Recommendation (IPM-Based)")
            t1_ai  = gr.Markdown(label="AI Ecological Advice")
            t1_map = gr.HTML(label="Sighting Map")
            t1_btn.click(
                fn=predict_field,
                inputs=[t1_img, t1_crop, t1_loc, t1_season,
                        t1_lat, t1_lon, lang_selector],
                outputs=[t1_class, t1_conf, t1_risk, t1_rec, t1_ai, t1_map],
            )

        # ── Tab 2: Expert Analysis ────────────────────────────
        with gr.Tab("Expert Analysis"):
            gr.Markdown(
                "Detailed diagnostic report with top-5 predictions, "
                "biodiversity data from iNaturalist and GBIF, and AI management plan."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    t2_img   = gr.Image(label="Insect photograph", type="numpy", height=275)
                    t2_crop  = gr.Textbox(label="Crop", placeholder="e.g. Wheat")
                    t2_farm  = gr.Dropdown(
                        label="Farming system",
                        choices=["Mixed farming", "Arable / grain crops", "Orchard / fruit",
                                 "Protected horticulture", "Rice paddy", "Vegetables",
                                 "Maize", "Cotton", "Pasture"],
                        value="Mixed farming",
                    )
                    t2_org   = gr.Checkbox(label="Certified organic farm", value=False)
                    with gr.Row():
                        t2_lat = gr.Textbox(label="Latitude",  placeholder="e.g. 31.5204")
                        t2_lon = gr.Textbox(label="Longitude", placeholder="e.g. 74.3587")
                    t2_notes = gr.Textbox(
                        label="Field observation notes",
                        placeholder="Infestation level, damage symptoms, previous treatments...",
                        lines=4,
                    )
                    t2_btn = gr.Button("Run Expert Analysis", variant="primary", size="lg")
                with gr.Column(scale=1):
                    t2_report = gr.Markdown(label="Diagnostic Report")
            with gr.Row():
                t2_top5 = gr.Markdown(label="Top-5 Model Predictions")
                t2_ai   = gr.Markdown(label="AI Management Plan")
            t2_bio = gr.Markdown(label="Biodiversity Profile (iNaturalist + GBIF)")
            t2_map = gr.HTML(label="Geographic Map")
            t2_btn.click(
                fn=predict_expert,
                inputs=[t2_img, t2_crop, t2_lat, t2_lon,
                        t2_notes, t2_farm, t2_org, lang_selector],
                outputs=[t2_report, t2_top5, t2_bio, t2_ai, t2_map],
            )

        # ── Tab 3: Describe & Identify ────────────────────────
        with gr.Tab("Describe & Identify"):
            gr.Markdown(
                "No clear photograph? Describe the insect's appearance and "
                "behaviour and the AI will suggest what it might be. "
                "Requires Groq API key (free at console.groq.com)."
            )
            with gr.Row():
                with gr.Column():
                    t3_desc = gr.Textbox(
                        label="Describe the insect",
                        placeholder=(
                            "Example: Small green soft-bodied insect, approximately 2 mm, "
                            "in dense clusters on the undersides of tomato leaves. "
                            "Sticky residue visible. Leaves curling and yellowing."
                        ),
                        lines=6,
                    )
                    t3_crop = gr.Textbox(label="Crop affected",
                                         placeholder="e.g. Tomato")
                    t3_loc  = gr.Textbox(label="Location / region",
                                         placeholder="e.g. Punjab, Pakistan")
                    t3_btn  = gr.Button("Identify from Description", variant="primary")
                with gr.Column():
                    t3_result = gr.Markdown(label="AI Identification Result")
            t3_btn.click(
                fn=identify_from_text,
                inputs=[t3_desc, t3_crop, t3_loc, lang_selector],
                outputs=t3_result,
            )

        # ── Tab 4: Invasive Species Surveillance ──────────────
        with gr.Tab("Invasive Surveillance"):
            gr.Markdown(
                "Check for invasive insect species reported within 100 km of your farm. "
                "Data sourced from iNaturalist — free, no login needed."
            )
            with gr.Row():
                t4_lat = gr.Textbox(label="Farm latitude",  placeholder="e.g. 31.5204")
                t4_lon = gr.Textbox(label="Farm longitude", placeholder="e.g. 74.3587")
            t4_btn = gr.Button("Scan for Nearby Invasives", variant="primary")
            t4_out = gr.Markdown()
            t4_btn.click(
                fn=scan_for_invasives,
                inputs=[t4_lat, t4_lon, lang_selector],
                outputs=t4_out,
            )

        # ── Tab 5: Sightings Map ──────────────────────────────
        with gr.Tab("Sightings Map"):
            gr.Markdown(
                "All geo-tagged insect sightings recorded during this session. "
                "Invasive species show a 5 km risk zone. Export as CSV for field records."
            )
            with gr.Row():
                t5_refresh = gr.Button("Refresh Map",  variant="secondary")
                t5_export  = gr.Button("Export CSV",   variant="secondary")
            t5_map = gr.HTML(
                "<p style='color:#6C757D;padding:20px;font-family:sans-serif;'>"
                "Click <strong>Refresh Map</strong> to display sightings.</p>"
            )
            t5_csv = gr.File(label="Download sightings as CSV")
            t5_refresh.click(fn=refresh_map,             outputs=t5_map)
            t5_export.click(fn=export_sightings_csv,     outputs=t5_csv)

        # ── Tab 6: About ──────────────────────────────────────
        with gr.Tab("About"):
            gr.Markdown(f"""
## EcoGuard.AI — Smart Pest & Pollinator Identification System

### Model Information

| Property | Value |
|----------|-------|
| Architecture | {MODEL_CHOICE.upper()} |
| Validation Accuracy | {SAVED_ACC:.1f}% |
| Number of Classes | {NUM_CLASSES} insect roles |
| Languages | English \\| اردو \\| 中文 \\| हिंदी \\| 한국어 |

### Insect Roles Classified

| Role | Risk Level | Recommended Action |
|------|-----------|-------------------|
| Pest | High | IPM: monitor → biological → cultural → chemical (last resort) |
| Pollinator | Low | Protect: no-spray zones, habitat, water |
| Predator | Low | Never spray — provide habitat, be patient |
| Biological Control Agent | Low | Absolute no-spray — support with insectary plants |
| Decomposer | Low | Minimise tillage, maintain organic matter |
| Invasive Species | Critical | Report to NPPO immediately |
| Neutral | Low | Monitor only — no treatment needed |

### APIs Used (all free)

| API | Cost | Purpose |
|-----|------|---------|
| iNaturalist | Free | Species data, invasive species radar |
| GBIF | Free | Taxonomy verification |
| Groq LLM | Free | AI ecological advice (add key in Space Settings) |

### UN Sustainable Development Goals
**SDG 1** · **SDG 2** · **SDG 12** · **SDG 13** · **SDG 15**

### How to Add Groq AI Key
1. Go to [console.groq.com](https://console.groq.com) — sign up free
2. Click **API Keys** → **Create API Key** → copy the key
3. In this Hugging Face Space: **Settings** → **Variables and secrets**
4. Add: Name = `GROQ_API_KEY`, Value = your key
5. The Space will restart automatically — AI advice will be active
            """)


if __name__ == "__main__":
    app.launch()
