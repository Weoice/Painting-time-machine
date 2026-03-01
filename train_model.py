import requests
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import time
from scipy import ndimage
from scipy.stats import skew, kurtosis

def get_dominant_colors(image_url, k=12):
    try:
        response = requests.get(image_url, timeout=10)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((250, 250))
        pixels = np.array(img).reshape(-1, 3)
        kmeans = KMeans(n_clusters=k, n_init=25, random_state=42)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        return colors, counts, img
    except:
        return None, None, None

def extract_features(colors, counts, img):
    if colors is None:
        return None
    
    img_array = np.array(img)
    proportions = counts / counts.sum()
    features = []
    
    for i in range(len(colors)):
        features.extend([colors[i][0], colors[i][1], colors[i][2], proportions[i]])
    
    hsv_img = img.convert('HSV')
    hsv_array = np.array(hsv_img)
    features.append(np.mean(hsv_array[:,:,0]))
    features.append(np.std(hsv_array[:,:,0]))
    features.append(np.mean(hsv_array[:,:,1]))
    features.append(np.std(hsv_array[:,:,1]))
    features.append(np.mean(hsv_array[:,:,2]))
    features.append(np.std(hsv_array[:,:,2]))
    features.append(skew(hsv_array[:,:,1].flatten()))
    features.append(kurtosis(hsv_array[:,:,1].flatten()))
    features.append(np.mean(colors))
    features.append(np.std(colors))
    features.append(np.mean(colors[:, 0]) - np.mean(colors[:, 2]))
    features.append(np.mean(colors[:, 1]) - np.mean(colors[:, 2]))
    features.append(np.mean(colors[:, 0]) - np.mean(colors[:, 1]))
    features.append(np.mean(np.std(colors, axis=1)))
    features.append(np.max(colors) - np.min(colors))
    features.append(np.var(colors))
    features.append(skew(colors.flatten()))
    features.append(kurtosis(colors.flatten()))

    features.append(np.mean(img_array[:,:,0]))
    features.append(np.mean(img_array[:,:,1]))
    features.append(np.mean(img_array[:,:,2]))
    features.append(np.std(img_array[:,:,0]))
    features.append(np.std(img_array[:,:,1]))
    features.append(np.std(img_array[:,:,2]))

    features.append(skew(img_array[:,:,0].flatten()))
    features.append(skew(img_array[:,:,1].flatten()))
    features.append(skew(img_array[:,:,2].flatten()))
    
    gray = np.mean(img_array, axis=2)
    edges_x = ndimage.sobel(gray, axis=0)
    edges_y = ndimage.sobel(gray, axis=1)
    edges = np.hypot(edges_x, edges_y)
    features.append(np.mean(edges))
    features.append(np.std(edges))
    features.append(np.percentile(edges, 90))
    features.append(np.percentile(edges, 10))
    features.append(skew(edges.flatten()))
    
    features.append(np.max(gray) - np.min(gray))
    features.append(np.std(gray))
    features.append(np.mean(np.abs(np.diff(gray, axis=0))))
    features.append(np.mean(np.abs(np.diff(gray, axis=1))))
    features.append(np.var(gray))
    features.append(skew(gray.flatten()))
    features.append(kurtosis(gray.flatten()))
    
    laplacian = ndimage.laplace(gray)
    features.append(np.mean(np.abs(laplacian)))
    features.append(np.std(laplacian))
    features.append(np.var(laplacian))
    
    hist, _ = np.histogram(gray, bins=16, range=(0, 255))
    hist = hist / hist.sum()
    features.extend(hist.tolist())
    
    for channel in range(3):
        hist_c, _ = np.histogram(img_array[:,:,channel], bins=8, range=(0, 255))
        hist_c = hist_c / hist_c.sum()
        features.extend(hist_c.tolist())
    
    features.append(proportions[0])
    features.append(proportions[1] if len(proportions) > 1 else 0)
    features.append(proportions[2] if len(proportions) > 2 else 0)
    features.append(np.sum(proportions[:3]))
    features.append(np.sum(proportions[:5]))
    warm_colors = colors[:, 0] > colors[:, 2]
    features.append(np.sum(proportions[warm_colors]))
    
    cool_colors = colors[:, 2] > colors[:, 0]
    features.append(np.sum(proportions[cool_colors]))
    
    max_rgb = np.max(img_array, axis=2)
    min_rgb = np.min(img_array, axis=2)
    saturation = np.divide(max_rgb - min_rgb, max_rgb + 1, where=(max_rgb + 1) != 0, 
                          out=np.zeros_like(max_rgb, dtype=float))
    features.append(np.mean(saturation))
    features.append(np.std(saturation))
    features.append(np.percentile(saturation, 75))
    features.append(np.percentile(saturation, 25))
    
    luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    features.append(np.mean(luminance))


    features.append(np.std(luminance))
    features.append(skew(luminance.flatten()))
    features.append(np.sum(luminance < 85) / luminance.size)
    features.append(np.sum(luminance > 170) / luminance.size)
    
    return features

search_url = "https://collectionapi.metmuseum.org/public/collection/v1/search?hasImages=true&q=painting"
search_res = requests.get(search_url, timeout=25).json()
all_ids = search_res.get('objectIDs', [])[:2000]
rows = []
for i, obj_id in enumerate(all_ids):
    try:
        obj = requests.get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{obj_id}", 
                          timeout=10).json()
        img_url = obj.get('primaryImageSmall')
        year = obj.get('objectBeginDate')
        
        if img_url and year and 1400 <= year <= 2000:
            colors, counts, img = get_dominant_colors(img_url)
            if colors is not None:
                feature_row = extract_features(colors, counts, img)
                if feature_row and len(feature_row) > 80:
                    feature_row.append(year)
                    rows.append(feature_row)
        
        if len(rows) >= 400:
            break
        
        time.sleep(0.1)
    except:
        continue

if len(rows) < 60:
    exit()

df = pd.DataFrame(rows)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

gb_model = GradientBoostingRegressor(
    n_estimators=600,
    learning_rate=0.04,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    subsample=0.85,
    max_features='sqrt',
    random_state=42
)
gb_model.fit(X_train, y_train)
gb_mae = np.mean(np.abs(gb_model.predict(X_test) - y_test))

rf_model = RandomForestRegressor(
    n_estimators=500,
    max_depth=25,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)
rf_mae = np.mean(np.abs(rf_model.predict(X_test) - y_test))

if gb_mae < rf_mae:
    best_model = gb_model
    best_mae = gb_mae
    model_type = "Gradient Boosting"
else:
    best_model = rf_model
    best_mae = rf_mae
    model_type = "Random Forest"

best_model.fit(X_scaled, y)

era_counts = {}
for year in y:
    if year < 1600: era = 'Renaissance'
    elif year < 1700: era = 'Baroque'
    elif year < 1800: era = 'Rococo/Neoclassical'
    elif year < 1850: era = 'Romantic'
    elif year < 1900: era = 'Impressionist'
    else: era = 'Modern'
    era_counts[era] = era_counts.get(era, 0) + 1

stats = {
    'num_paintings': len(df),
    'min_year': int(y.min()),
    'max_year': int(y.max()),
    'era_distribution': era_counts,
    'mae': float(best_mae),
    'model_type': model_type,
    'num_features': X_scaled.shape[1]
}

with open('pretrained_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'stats': stats}, f)