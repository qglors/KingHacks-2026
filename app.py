import streamlit as st
import cv2
import numpy as np
import base64
import os

BACKGROUND_IMAGE_FILENAME = 'background.jpg' 

#set bg
def set_background(image_file):
    if not os.path.exists(image_file):
        st.error(f"Error: Could not find image named '{image_file}'.")
        return

    with open(image_file, "rb") as f:
        bin_str = f.read()
    
    b64_val = base64.b64encode(bin_str).decode()
    
    #ui
    static_css = '''
    <style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Source+Code+Pro:wght@400;600&display=swap');

    /* Global Font */
    html, body, [class*="css"] {
        font-family: 'Source Code Pro', monospace !important; 
    }
    
    /* Header Font */
    h1, h2, h3, .navbar-title {
        font-family: 'Share Tech Mono', monospace !important;
        letter-spacing: 0px;
    }

    /* Dark Overlay */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.4); 
        z-index: -1;
    }

    /* Navbar */
    .navbar {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 70px;
        background: rgba(15, 12, 41, 0.95);
        border-bottom: 1px solid rgba(166, 136, 250, 0.3);
        z-index: 99999;
        display: flex;
        align-items: center;
        padding-left: 30px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }
    
    .navbar-title {
        font-size: 26px;
        color: #E0E0E0;
        text-shadow: 0px 0px 8px rgba(180, 150, 255, 0.6);
    }

    /* Main Content Box */
    .block-container {
        background: linear-gradient(135deg, rgba(20, 10, 40, 0.9) 0%, rgba(50, 20, 80, 0.9) 100%);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5);
        
        padding-top: 1rem !important;
        padding-bottom: 3rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
        
        margin-top: 5rem;
        max-width: 850px;
    }
    
    /* UI Elements */
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    h1, h2, h3, p, label, span, div { color: #FFFFFF !important; }
    
    .stFileUploader {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed #A688FA;
        border-radius: 15px;
        padding: 20px;
    }
    
    div.stButton > button {
        background: linear-gradient(90deg, #6a11cb 0%, #2575fc 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        border-radius: 30px;
        font-size: 16px;
        font-weight: bold;
        font-family: 'Source Code Pro', monospace;
        transition: all 0.3s ease;
        box-shadow: 0px 0px 15px rgba(100, 100, 255, 0.5);
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0px 0px 25px rgba(100, 100, 255, 0.8);
    }
    
    .stSuccess, .stInfo, .stWarning {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
        border-radius: 10px;
        font-family: 'Source Code Pro', monospace;
    }
    </style>
    
    <div class="navbar">
        <div class="navbar-title"> SEROLENS </div>
    </div>
    '''

    #bg
    background_css = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_val}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    
    st.markdown(static_css + background_css, unsafe_allow_html=True)

#medical advice
def get_immunity_advice(score):
    if score > 75:
        return ("Strong Protection", "Your antibody levels are high. You likely have strong protection. No immediate action is required.", "success")
    elif score > 40:
        return ("Moderate Protection", "Your antibody levels are visible, but fading. You have some immunity, but you should research local health guidelines regarding booster shots.", "warning")
    else:
        return ("Low/No Protection", "Antibody levels are very low or undetectable. If you have been vaccinated previously, your immunity may have weakened significantly. You may want to consider consulting a healthcare provider about a booster shot.", "error")

#img analysis
def analyze_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, white_thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    test_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 20000: 
            if area > max_area:
                max_area = area
                test_contour = cnt
                
    if test_contour is None:
        return None, "Error: Could not find the test strip."

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [test_contour], -1, 255, -1)
    isolated_test = cv2.bitwise_and(img, img, mask=mask)
    
    hsv = cv2.cvtColor(isolated_test, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    target_contour = None
    max_area = 0
    for cnt in red_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = float(w) / h
        is_not_square = (aspect_ratio > 1.5) or (aspect_ratio < 0.8)
        if area > 50 and is_not_square: 
            if area > max_area:
                max_area = area
                target_contour = cnt

    if target_contour is None:
        return None, "Error: No antibody line detected (Result Negative or Invalid)."

    x, y, w, h = cv2.boundingRect(target_contour)
    result_img = img.copy()
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 4)

    roi = gray[y:y+h, x:x+w]
    mean_brightness = np.mean(roi)
    score = 255 - mean_brightness
    max_expected = 120.0 
    percentage = (score / max_expected) * 100
    if percentage > 100: percentage = 100
    
    return result_img, percentage

#app
st.set_page_config(page_title="SeroLens AI", page_icon="", layout="centered")

#apply bg
set_background(BACKGROUND_IMAGE_FILENAME)

#heading
st.markdown("### Antibody Analysis")
st.write("Upload your test strip to analyze your antibody levels.")

uploaded_file = st.file_uploader("Upload Test Image", type=["jpg", "jpeg", "png"], key="test_uploader")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, channels="BGR", caption="Original Scan", use_container_width=True)
    
    with col2:
        if st.button("Analyze Sample"):
            with st.spinner("Processing data..."):
                processed_img, result = analyze_image(img)
                if processed_img is None:
                    st.error(result)
                else:
                    st.image(processed_img, channels="BGR", caption="Test Line Detection", use_container_width=True)
                    
                    st.success(f"**Scan Complete**")
                    st.metric(label="Opacity Score", value=f"{round(result, 1)}%")
                    st.progress(int(result))

                    # medical interpretation
                    st.markdown("---")
                    title, advice, status_color = get_immunity_advice(result)
                    
                    if status_color == "success":
                        st.success(f"**{title}**\n\n{advice}")
                    elif status_color == "warning":
                        st.warning(f"**{title}**\n\n{advice}")
                    else:
                        st.error(f"**{title}**\n\n{advice}")