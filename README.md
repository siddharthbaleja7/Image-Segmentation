# 🖼️ Interactive Image Segmentation using DFS

An interactive web application that allows users to upload **binary** or **grayscale** images and perform **connected component segmentation** using the **Depth-First Search (DFS)** algorithm.

---

## 🚀 Demo

👉 [Try it Live on Streamlit](https://image-segmentation-8qhgnts6zi8e2x9qjrew3z.streamlit.app/)

---
## 🚀 Video Link

👉 [Video Link](https://drive.google.com/file/d/1eyMHhJXaWzHbS9qgI_MAWu1B9TAAMnjT/view?usp=share_link)

---

## 🔍 What It Does

- Upload an image (binary or grayscale).
- Preprocess it (resize, threshold, invert, clean using morphology).
- Detect connected components using **DFS traversal**.
- Assign distinct colors to each detected region.
- Visualize component statistics, bounding boxes, and advanced plots.
- Download results and data.

---

## 💡 Applications

| Domain               | Use Case                               |
|----------------------|-----------------------------------------|
| 🏥 Medical Imaging    | Segment cells or detect tumor regions   |
| 📝 OCR Processing     | Separate text characters                |
| 🤖 Robotics           | Object recognition for navigation       |
| 📹 Surveillance       | Motion or object detection              |
| 🌾 Agriculture        | Analyze crop health from aerial images |

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](w)
- **Image Processing**: [OpenCV](w), [Pillow](w), [NumPy](w)
- **Visualization**: [Matplotlib](w), [Plotly](w)
- **Language**: Python 3.x

---

## ▶️ Run Locally

Make sure Python and pip are installed. Then:

```bash
git clone https://github.com/siddharthbaleja7/image-segmentation.git
cd image-segmentation

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run main.py
```
---
