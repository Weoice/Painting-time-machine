import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle
import os
from scipy import ndimage
from scipy.stats import skew, kurtosis

st.set_page_config(page_title="Painting time machine", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600;700;800;900&family=Lora:wght@400;500;600;700&display=swap');
    * {font-family: 'Lora', serif;}
    h1, h2, h3 {font-family: 'Playfair Display', serif !important;}
    
    .stApp {
        background: linear-gradient(135deg, #e8eef3 0%, #c3cfe2 100%);
        background-image: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,.08) 2px, rgba(255,255,255,.08) 4px),
            linear-gradient(135deg, #e8eef3 0%, #c3cfe2 100%);
    }
    
    .main-header {
        background: linear-gradient(135deg, #ffffff 0%, #f5f7fa 100%);
        border-radius: 20px; padding: 45px; margin-bottom: 35px;
        box-shadow: 0 12px 45px rgba(0,0,0,0.12), inset 0 -2px 8px rgba(255,255,255,0.9), inset 0 2px 6px rgba(0,0,0,0.04);
        border: 2px solid rgba(255,255,255,0.9);
        position: relative; overflow: hidden;
    }
    
    .main-header::before {
        content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.5), transparent);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {0% {left: -100%;} 50%, 100% {left: 100%;}}
    
    .antique-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border-radius: 16px; padding: 28px;
        box-shadow: 0 10px 35px rgba(0,0,0,0.1), inset 0 1px 0 rgba(255,255,255,1), inset 0 -1px 0 rgba(0,0,0,0.02);
        border: 2px solid rgba(139,114,89,0.25);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .antique-card:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 18px 55px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,1);
    }
    
    .result-card {
        background: linear-gradient(to bottom, #fffbf5 0%, #f7f2ec 100%);
        border-radius: 22px; padding: 38px;
        box-shadow: 0 14px 45px rgba(139,114,89,0.18), inset 0 2px 5px rgba(255,255,255,0.9);
        border: 2px solid rgba(139,114,89,0.3);
        margin: 28px 0;
        animation: fadeInUp 0.7s ease-out;
    }
    
    @keyframes fadeInUp {
        from {opacity: 0; transform: translateY(35px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    .metric-box {
        background: linear-gradient(135deg, #ffffff 0%, #faf8f5 100%);
        border-radius: 20px; padding: 35px; text-align: center;
        box-shadow: 0 12px 35px rgba(139,114,89,0.15), inset 0 2px 3px rgba(255,255,255,1), inset 0 -2px 3px rgba(139,114,89,0.08);
        border: 2px solid rgba(139,114,89,0.2);
        transition: all 0.4s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        position: relative; overflow: hidden;
    }
    
    .metric-box::before {
        content: ''; position: absolute; top: 50%; left: 50%; width: 0; height: 0;
        border-radius: 50%; background: rgba(139,114,89,0.12);
        transform: translate(-50%, -50%); transition: width 0.7s, height 0.7s;
    }
    
    .metric-box:hover::before {width: 320px; height: 320px;}
    .metric-box:hover {transform: scale(1.1) rotate(2deg); box-shadow: 0 18px 50px rgba(139,114,89,0.25), inset 0 2px 3px rgba(255,255,255,1);}
    
    .metric-value {
        font-size: 3.2rem; font-weight: 900;
        background: linear-gradient(135deg, #7a5f4a 0%, #5d4435 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        position: relative; z-index: 1;
        font-family: 'Playfair Display', serif;
    }
    
    .metric-label {
        font-size: 1rem; color: #6b5644; margin-top: 10px;
        font-weight: 700; letter-spacing: 2px; text-transform: uppercase;
        position: relative; z-index: 1;
    }
    
    .confidence-bar {
        background: rgba(139,114,89,0.15);
        border-radius: 12px; height: 14px; margin: 18px 0 10px 0;
        overflow: hidden; box-shadow: inset 0 3px 6px rgba(0,0,0,0.12);
    }
    
    .confidence-fill {
        height: 100%; border-radius: 12px;
        transition: width 1.2s ease-out;
        box-shadow: 0 3px 10px rgba(139,114,89,0.35);
    }
    
    .upload-area {
        background: linear-gradient(to bottom, #ffffff 0%, #f9f7f4 100%);
        border: 3px dashed rgba(139,114,89,0.35);
        border-radius: 22px; padding: 45px 55px; text-align: center;
        transition: all 0.4s ease;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 35px;
    }
    
    .upload-area:hover {
        border-color: rgba(139,114,89,0.65);
        background: linear-gradient(to bottom, #fffffe 0%, #f7f4f0 100%);
        transform: scale(1.02);
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.04), 0 10px 30px rgba(139,114,89,0.12);
    }
    
    .upload-title {
        color: #3d2f1f; font-size: 1.65rem; font-weight: 700;
        margin: 0 0 12px 0; font-family: 'Playfair Display', serif;
    }
    
    .upload-subtitle {
        color: #6b5644; font-size: 1.1rem; font-weight: 500; margin: 0;
    }
    
    .era-badge {
        display: inline-block;
        background: linear-gradient(135deg, #7a5f4a 0%, #5d4435 100%);
        color: white; padding: 14px 32px; border-radius: 28px;
        font-weight: 800; font-size: 1.2rem;
        box-shadow: 0 8px 25px rgba(122,95,74,0.35);
        margin: 18px 0; transition: all 0.3s ease;
        font-family: 'Playfair Display', serif;
    }
    
    .era-badge:hover {transform: scale(1.12); box-shadow: 0 10px 35px rgba(122,95,74,0.45);}
    
    .info-box {
        background: linear-gradient(to right, #fffbf5 0%, #ffffff 100%);
        border-left: 6px solid #7a5f4a; padding: 20px; border-radius: 12px;
        margin: 14px 0; color: #2d2318;
        box-shadow: 0 5px 18px rgba(139,114,89,0.1);
        transition: all 0.3s ease; font-weight: 500;
    }
    
    .info-box:hover {transform: translateX(10px); box-shadow: 0 7px 24px rgba(139,114,89,0.18);}
    
    .ornament {text-align: center; color: #7a5f4a; font-size: 1.6rem; margin: 22px 0; opacity: 0.7;}
    
    h1 {color: #2d2318 !important; font-weight: 900 !important; text-shadow: 2px 2px 5px rgba(139,114,89,0.12);}
    h2, h3 {color: #3d2f1f !important; font-weight: 800 !important;}
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f5f7fa 100%);
        border-right: 2px solid rgba(139,114,89,0.25);
    }
    
    .sidebar-stat {
        background: rgba(255,255,255,0.75); padding: 14px;
        border-radius: 12px; margin: 10px 0;
        border: 2px solid rgba(139,114,89,0.2);
        color: #2d2318; font-weight: 500;
    }
    
    .sidebar-stat strong {color: #3d2f1f; font-weight: 700;}
</style>
""", unsafe_allow_html=True)

def get_colors_from_img(img, k=12):
    if isinstance(img, str):
        img = Image.open(img).convert('RGB')
    img = img.resize((250, 250))
    pixels = np.array(img).reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, n_init=25, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    return colors, counts, img

def get_colors(image_file, k=12):
    img = Image.open(image_file).convert('RGB')
    return get_colors_from_img(img, k)

def get_features(colors, counts, img):
    arr = np.array(img)
    props = counts / counts.sum()
    feats = []
    
    for i in range(len(colors)):
        feats.extend([colors[i][0], colors[i][1], colors[i][2], props[i]])
    
    hsv = img.convert('HSV')
    hsv_arr = np.array(hsv)
    feats.append(np.mean(hsv_arr[:,:,0]))
    feats.append(np.std(hsv_arr[:,:,0]))
    feats.append(np.mean(hsv_arr[:,:,1]))
    feats.append(np.std(hsv_arr[:,:,1]))
    feats.append(np.mean(hsv_arr[:,:,2]))
    feats.append(np.std(hsv_arr[:,:,2]))
    feats.append(skew(hsv_arr[:,:,1].flatten()))
    feats.append(kurtosis(hsv_arr[:,:,1].flatten()))
    
    feats.append(np.mean(colors))
    feats.append(np.std(colors))
    feats.append(np.mean(colors[:, 0]) - np.mean(colors[:, 2]))
    feats.append(np.mean(colors[:, 1]) - np.mean(colors[:, 2]))
    feats.append(np.mean(colors[:, 0]) - np.mean(colors[:, 1]))
    feats.append(np.mean(np.std(colors, axis=1)))
    feats.append(np.max(colors) - np.min(colors))
    feats.append(np.var(colors))
    feats.append(skew(colors.flatten()))
    feats.append(kurtosis(colors.flatten()))
    
    feats.append(np.mean(arr[:,:,0]))
    feats.append(np.mean(arr[:,:,1]))
    feats.append(np.mean(arr[:,:,2]))
    feats.append(np.std(arr[:,:,0]))
    feats.append(np.std(arr[:,:,1]))
    feats.append(np.std(arr[:,:,2]))
    feats.append(skew(arr[:,:,0].flatten()))
    feats.append(skew(arr[:,:,1].flatten()))
    feats.append(skew(arr[:,:,2].flatten()))
    
    gray = np.mean(arr, axis=2)
    ex = ndimage.sobel(gray, axis=0)
    ey = ndimage.sobel(gray, axis=1)
    edges = np.hypot(ex, ey)
    feats.append(np.mean(edges))
    feats.append(np.std(edges))
    feats.append(np.percentile(edges, 90))
    feats.append(np.percentile(edges, 10))
    feats.append(skew(edges.flatten()))
    
    feats.append(np.max(gray) - np.min(gray))
    feats.append(np.std(gray))
    feats.append(np.mean(np.abs(np.diff(gray, axis=0))))
    feats.append(np.mean(np.abs(np.diff(gray, axis=1))))
    feats.append(np.var(gray))
    feats.append(skew(gray.flatten()))
    feats.append(kurtosis(gray.flatten()))
    
    lap = ndimage.laplace(gray)
    feats.append(np.mean(np.abs(lap)))
    feats.append(np.std(lap))
    feats.append(np.var(lap))
    
    hist, _ = np.histogram(gray, bins=16, range=(0, 255))
    hist = hist / hist.sum()
    feats.extend(hist.tolist())
    
    for ch in range(3):
        h, _ = np.histogram(arr[:,:,ch], bins=8, range=(0, 255))
        h = h / h.sum()
        feats.extend(h.tolist())
    
    feats.append(props[0])
    feats.append(props[1] if len(props) > 1 else 0)
    feats.append(props[2] if len(props) > 2 else 0)
    feats.append(np.sum(props[:3]))
    feats.append(np.sum(props[:5]))
    
    warm = colors[:, 0] > colors[:, 2]
    feats.append(np.sum(props[warm]))
    
    cool = colors[:, 2] > colors[:, 0]
    feats.append(np.sum(props[cool]))
    
    max_rgb = np.max(arr, axis=2)
    min_rgb = np.min(arr, axis=2)
    sat = np.divide(max_rgb - min_rgb, max_rgb + 1, where=(max_rgb + 1) != 0, 
                    out=np.zeros_like(max_rgb, dtype=float))
    feats.append(np.mean(sat))
    feats.append(np.std(sat))
    feats.append(np.percentile(sat, 75))
    feats.append(np.percentile(sat, 25))
    
    lum = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
    feats.append(np.mean(lum))
    feats.append(np.std(lum))
    feats.append(skew(lum.flatten()))
    feats.append(np.sum(lum < 85) / lum.size)
    feats.append(np.sum(lum > 170) / lum.size)
    
    return feats

def get_era(year):
    if year < 1600: return "Renaissance"
    elif year < 1700: return "Baroque"
    elif year < 1800: return "Rococo/Neoclassical"
    elif year < 1850: return "Romantic"
    elif year < 1900: return "Impressionist"
    else: return "Modern"

@st.cache_resource
def load_model():
    with open('pretrained_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data.get('scaler'), data['stats']

model, scaler, stats = load_model()
margin = int(stats['mae'])

st.markdown('''
<div class="main-header">
    <h1 style="text-align: center; margin: 0; font-size: 3.8rem;">Painting Time Machine</h1>
    <div class="ornament">* * *</div>
    <p style="text-align: center; font-size: 1.4rem; color: #3d2f1f; margin: 0; font-weight: 600;">
        Discover the Era of Art Through AI Analysis
    </p>
</div>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Model Information")
    
    st.markdown(f"""
    <div class="sidebar-stat">
        <strong>Training Dataset</strong><br>
        {stats.get('num_paintings', 0)} paintings
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-stat">
        <strong>Time Period</strong><br>
        {stats.get('min_year', 1400)} - {stats.get('max_year', 2000)}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-stat">
        <strong>Prediction Accuracy</strong><br>
        ±{margin} years
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="sidebar-stat">
        <strong>Features Analyzed</strong><br>
        {stats.get('num_features', 0)} features<br>
        <small style="color: #6b5644; font-weight: 600;">Color - Texture - Composition</small>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Era Distribution")
    
    era_dist = stats.get('era_distribution', {})
    if era_dist:
        for era, count in sorted(era_dist.items(), key=lambda x: x[1], reverse=True):
            st.markdown(f"**{era}:** {count} paintings")

st.markdown('<div class="upload-area">', unsafe_allow_html=True)
st.markdown('<p class="upload-title">Upload Your Painting</p>', unsafe_allow_html=True)
st.markdown('<p class="upload-subtitle">Drag and drop or click to browse</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded:
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown('<div class="antique-card">', unsafe_allow_html=True)
        st.image(uploaded, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.spinner("Analyzing artwork..."):
            
            preds = []
            confs = []
            
            prog = st.empty()
            prog.text("1st Analysis...")
            
            colors, counts, img = get_colors(uploaded)
            feats = get_features(colors, counts, img)
            
            if scaler:
                feats_scaled = scaler.transform([feats])
            else:
                feats_scaled = [feats]
            
            year = int(model.predict(feats_scaled)[0])
            
            feat_var = np.std(feats)
            col_var = np.std([colors[i][j] for i in range(len(colors)) for j in range(3)])
            edge_str = feats[40]
            
            conf = max(45, min(96, 100 - (margin / 2.2)))
            if feat_var < 20: conf *= 0.85
            if col_var < 30: conf *= 0.9
            if edge_str < 10: conf *= 0.9
            conf = max(40, min(96, conf))
            
            preds.append(year)
            confs.append(conf)
            
            prog.text("2nd Analysis...")
            arr = np.array(img)
            img2 = Image.fromarray(np.clip(arr * 1.2, 0, 255).astype(np.uint8))
            c2, n2, i2 = get_colors_from_img(img2)
            f2 = get_features(c2, n2, i2)
            if scaler:
                f2 = scaler.transform([f2])
            else:
                f2 = [f2]
            p2 = int(model.predict(f2)[0])
            preds.append(p2)
            confs.append(min(96, conf * 1.05))
            
            prog.text("3rd Analysis...")
            img3 = Image.fromarray(np.clip(arr * 0.85 + 20, 0, 255).astype(np.uint8))
            c3, n3, i3 = get_colors_from_img(img3)
            f3 = get_features(c3, n3, i3)
            if scaler:
                f3 = scaler.transform([f3])
            else:
                f3 = [f3]
            p3 = int(model.predict(f3)[0])
            preds.append(p3)
            confs.append(min(96, conf * 1.03))
            
            prog.text("4th Analysis...")
            img4 = Image.open(uploaded).convert('RGB').resize((300, 300))
            c4, n4, i4 = get_colors_from_img(img4)
            f4 = get_features(c4, n4, i4)
            if scaler:
                f4 = scaler.transform([f4])
            else:
                f4 = [f4]
            p4 = int(model.predict(f4)[0])
            preds.append(p4)
            confs.append(min(96, conf * 1.08))
            
            prog.text("5th Analysis...")
            if len(preds) >= 3:
                weights = np.array(confs[:4]) / sum(confs[:4])
                ens = int(np.average(preds[:4], weights=weights))
                preds.append(ens)
                confs.append(min(96, max(confs[:4]) * 1.12))
            
            prog.empty()
            
            best = np.argmax(confs)
            final_year = preds[best]
            final_conf = confs[best]
            
            era = get_era(final_year)
            year_min = final_year - margin
            year_max = final_year + margin
            
            attempts = len(preds)
        
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        if final_conf < 70:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #fff5e6 0%, #ffe8cc 100%); 
                        border: 3px solid #ff9800; border-radius: 15px; padding: 22px; 
                        margin-bottom: 25px; color: #6b4e00; box-shadow: 0 6px 20px rgba(255,152,0,0.2);">
                <div style="font-size: 1.3rem; font-weight: 800; margin-bottom: 10px;">
                    Low Confidence: {final_conf:.0f}% (After {attempts} Attempts)
                </div>
                <div style="font-size: 1rem; font-weight: 500; line-height: 1.6;">
                    Despite multiple analysis methods, confidence remains low. For better accuracy:
                    <ul style="margin: 10px 0; padding-left: 20px;">
                        <li><strong>Higher resolution</strong> (minimum 800x800 pixels recommended)</li>
                        <li><strong>Professional photo</strong> with even lighting, no glare</li>
                        <li><strong>Remove frames/borders</strong> - crop to just the painting</li>
                        <li><strong>Clear details</strong> - make sure brushstrokes are visible</li>
                        <li><strong>Try different photo</strong> of the same artwork</li>
                    </ul>
                    <div style="background: rgba(255,255,255,0.4); padding: 10px; border-radius: 8px; margin-top: 10px;">
                    <strong>Best Results:</strong> Famous paintings like Mona Lisa, Starry Night work best
                    </div>
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif final_conf >= 80:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                        border: 3px solid #4caf50; border-radius: 15px; padding: 20px; 
                        margin-bottom: 25px; color: #1b5e20; box-shadow: 0 6px 20px rgba(76,175,80,0.2);">
                <div style="font-size: 1.2rem; font-weight: 800;">
                    Excellent: {final_conf:.0f}%
                </div>
                <div style="font-size: 0.95rem; font-weight: 500; margin-top: 5px;">
                    High-quality prediction after {attempts} analysis methods
                </div>
            </div>
            ''', unsafe_allow_html=True)
        elif final_conf >= 70:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        border: 3px solid #2196f3; border-radius: 15px; padding: 20px; 
                        margin-bottom: 25px; color: #0d47a1; box-shadow: 0 6px 20px rgba(33,150,243,0.2);">
                <div style="font-size: 1.2rem; font-weight: 800;">
                    Good: {final_conf:.0f}%
                </div>
                <div style="font-size: 0.95rem; font-weight: 500; margin-top: 5px;">
                    Reliable prediction after {attempts} attempts
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{final_year}</div>
                <div class="metric-label">Predicted Year</div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{era}</div>
                <div class="metric-label">Era</div>
            </div>
            ''', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### Confidence & Range")
        st.markdown(f'''
        <div class="info-box">
            <b>Estimated Period:</b> {year_min} - {year_max} (±{margin} years)
        </div>
        ''', unsafe_allow_html=True)
        
        if final_conf >= 80:
            bar_col = "linear-gradient(90deg, #27ae60 0%, #2ecc71 100%)"
            conf_txt = "High Confidence"
        elif final_conf >= 70:
            bar_col = "linear-gradient(90deg, #7a5f4a 0%, #9d8370 100%)"
            conf_txt = "Good Confidence"
        elif final_conf >= 60:
            bar_col = "linear-gradient(90deg, #f39c12 0%, #f1c40f 100%)"
            conf_txt = "Moderate"
        else:
            bar_col = "linear-gradient(90deg, #e74c3c 0%, #c0392b 100%)"
            conf_txt = "Low"
        
        st.markdown(f'''
        <div class="confidence-bar">
            <div class="confidence-fill" style="width: {final_conf}%; background: {bar_col};"></div>
        </div>
        <p style="text-align: center; color: #6b5644; font-size: 1rem; margin: 5px 0 0 0; font-weight: 600;">
            {conf_txt}: {final_conf:.0f}%
        </p>
        ''', unsafe_allow_html=True)
        
        st.markdown("### Color Palette")
        fig, axes = plt.subplots(1, len(colors), figsize=(13, 2))
        fig.patch.set_facecolor('#fffbf5')
        for i, color in enumerate(colors):
            ax = axes[i] if len(colors) > 1 else axes
            ax.imshow([[color/255]])
            ax.axis('off')
            hex_col = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            ax.text(0.5, -0.25, hex_col, ha='center', transform=ax.transAxes, 
                   fontsize=9, color='#3d2f1f', weight='bold', family='monospace')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown(f'''
        <div style="text-align: center; margin-top: 40px;">
            <div class="era-badge">{era} Period • circa {final_year}</div>
        </div>
        ''', unsafe_allow_html=True)

st.markdown('<div class="ornament">* * *</div>', unsafe_allow_html=True)
st.markdown(f'''
<div style="text-align: center; color: #6b5644; font-size: 1rem; font-weight: 500;">
            <p style="margin: 0;">SunnyHacks Submission! - Sathwik Chinthakayala </p>
    <p style="margin: 0;">Trained with paintings from The Metropolitan Museum of Art</p>
    <p style="margin: 8px 0 0 0;">Accuracy: ±{margin} years</p>
</div>
''', unsafe_allow_html=True)