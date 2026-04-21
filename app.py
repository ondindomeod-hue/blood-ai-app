
import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt

st.set_page_config(page_title="Blood AI", layout="centered")

st.title("🩸 Blood Cell Analyzer")
st.caption("Upload image to analyze blood cells")

# 🎨 ตั้งค่า
st.set_page_config(page_title="Blood AI", layout="wide")

# 🌙 Dark theme + style
st.markdown("""
<style>
body {background-color: #0E1117; color: white;}
.big-title {font-size:40px; font-weight:bold;}
.card {
    padding:20px;
    border-radius:15px;
    background-color:#1E1E1E;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# 🧬 Header
st.markdown('<div class="big-title">🩸 Blood Cell AI Analysis</div>', unsafe_allow_html=True)
st.write("AI วิเคราะห์เม็ดเลือดแดงและประเมินแนวโน้มโรค")

# 📤 Upload
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

import streamlit as st
import torch
from ultralytics import YOLO

@st.cache_resource
def load_model():
    model = YOLO("best.pt")   # 👈 ใช้ตรง ๆ
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file:
    col1, col2 = st.columns(2)   # 👈 เพิ่มตรงนี้

    with col1:
        img = Image.open(uploaded_file)
        img = img.resize((640,640))
        st.image(img, caption="Uploaded Image")

    with col2:
        results = model(img)
        st.image(results.render()[0], caption="Detection Result")
        df = results.pandas().xyxy[0]
        st.write(df)

        counts = df['name'].value_counts()

        echino = counts.get('Echinocyte', 0)
        acantho = counts.get('Acanthocyte', 0)
        normal = counts.get('Normal cell', 0)

        total = echino + acantho + normal

        st.metric("Echinocyte", f"{round(echino/total*100,1)}%")
        st.metric("Acanthocyte", f"{round(acantho/total*100,1)}%")
        st.metric("Normal", f"{round(normal/total*100,1)}%")

if total > 0:
    st.metric("Echinocyte", f"{round(echino/total*100,1)}%")
    st.metric("Acanthocyte", f"{round(acantho/total*100,1)}%")
    st.metric("Normal", f"{round(normal/total*100,1)}%")
else:
    st.warning("⚠️ No cells detected")

if acantho > 5:
    st.error("⚠️ Abnormal cells detected")
else:
    st.success("✅ Mostly normal cells")
    
    with col2:
        st.markdown("### 📊 Results")
        st.metric("Echinocyte", f"{e_p:.1f}%")
        st.metric("Acanthocyte", f"{a_p:.1f}%")
        st.metric("Normal", f"{n_p:.1f}%")

    # 📈 กราฟ
    st.markdown("### 📈 Distribution")
    fig, ax = plt.subplots()
    ax.pie([e_p, a_p, n_p],
           labels=['Echinocyte','Acanthocyte','Normal'],
           autopct='%1.1f%%')
    st.pyplot(fig)

    # 🩺 Diagnosis
    st.markdown("### 🩺 Diagnosis")

    if a_p > 20:
        st.error("🔴 High Risk: Liver Disease")
    elif e_p > 20:
        st.warning("🔵 Possible Kidney Issue")
    else:
        st.success("✅ Normal Condition")

    # 🖼 Result
    st.markdown("### 🔍 Detection Result")
    results.render()
    st.image(results.ims[0])
