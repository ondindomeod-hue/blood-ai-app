import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt

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

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file)
        st.image(img, caption="🖼 Uploaded Image", use_column_width=True)

    # โหลดโมเดล
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path='runs/train/exp/weights/best.pt')

    results = model(img)
    df = results.pandas().xyxy[0]
    counts = df['name'].value_counts()

    echino = counts.get('Echinocyte', 0)
    acantho = counts.get('Acanthocyte', 0)
    normal = counts.get('Normal cell', 0)

    total = echino + acantho + normal

    if total > 0:
        e_p = echino/total*100
        a_p = acantho/total*100
        n_p = normal/total*100
    else:
        e_p = a_p = n_p = 0

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
