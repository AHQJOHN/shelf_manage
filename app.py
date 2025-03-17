import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from ultralytics import YOLO

def load_model():
    model = YOLO("v3_v11_withoutaugmentation.pt", task = 'detect')  # Load YOLOv11n model
    return model

def get_color_for_class(class_id):
    np.random.seed(class_id)
    return tuple(np.random.randint(0, 255, size=3).tolist())

def run_yolo(model, image):
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data
    
    # Annotate image
    annotated_img = np.array(image.copy())
    class_counts = Counter()
    for det in detections:
        x1, y1, x2, y2, conf, cls = map(int, det[:6])
        label = f"{model.names[cls]}" #({conf:.2f})"
        color = get_color_for_class(cls)  # Get a unique color for each class
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 6)
        cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 6)
        class_counts[model.names[cls]] += 1
    
    return Image.fromarray(annotated_img), dict(class_counts)

def main():
    st.set_page_config(layout="wide")  # Responsive layout
    st.title("YOLOv11n Object Detection App")
    
    model = load_model()
    st.sidebar.success("YOLOv11n Model Loaded Successfully")
    
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        if st.sidebar.button("Run YOLO Detection"):
            annotated_img, class_counts = run_yolo(model, image)
            
            st.sidebar.subheader("Detection Summary")
            for class_name, count in class_counts.items():
                st.sidebar.write(f"**{class_name}**: {count}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Uploaded Image", width=300)
            with col2:
                st.image(annotated_img, caption="Detected Objects", width=800)

if __name__ == "__main__":
    main()










# import streamlit as st
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from collections import Counter
# from ultralytics import YOLO

# def load_model():
#     model = YOLO("shelf.v3i.yolov11/runs/detect/train/weights/best.pt", task = 'detect')  # Load YOLOv11n model
#     return model

# def run_yolo(model, image):
#     results = model(image)
#     detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data
    
#     # Annotate image
#     annotated_img = np.array(image.copy())
#     class_counts = Counter()
#     for det in detections:
#         x1, y1, x2, y2, conf, cls = map(int, det[:6])
#         label = f"{model.names[cls]} ({conf:.2f})"
#         cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 255), 6)
#         cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 6)
#         class_counts[model.names[cls]] += 1
    
#     return Image.fromarray(annotated_img), dict(class_counts)

# def main():
#     st.set_page_config(layout="wide")  # Responsive layout
#     st.title("YOLOv11n Object Detection App")
    
#     model = load_model()
#     st.sidebar.success("YOLOv11n Model Loaded Successfully")
    
#     uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "jfif"])
    
#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
        
#         if st.sidebar.button("Run YOLO Detection"):
#             annotated_img, class_counts = run_yolo(model, image)
            
#             st.sidebar.subheader("Detection Summary")
#             for class_name, count in class_counts.items():
#                 st.sidebar.write(f"**{class_name}**: {count}")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.image(image, caption="Uploaded Image", width=300)
#             with col2:
#                 st.image(annotated_img, caption="Detected Objects", width=800)

# if __name__ == "__main__":
#     main()



















# import streamlit as st
# import torch
# import cv2
# import numpy as np
# from PIL import Image
# from collections import Counter
# from ultralytics import YOLO

# def load_model():
#     model = YOLO("yolo11n.pt", task = "detect")  # Load YOLOv11n model
#     return model

# def run_yolo(model, image):
#     results = model(image)
#     detections = results[0].boxes.data.cpu().numpy()  # Extract bounding box data
    
#     # Annotate image
#     annotated_img = np.array(image.copy())
#     class_counts = Counter()
#     for det in detections:
#         x1, y1, x2, y2, conf, cls = map(int, det[:6])
#         label = f"{model.names[cls]} ({conf:.2f})"
#         cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         class_counts[model.names[cls]] += 1
    
#     return Image.fromarray(annotated_img), dict(class_counts)

# def main():
#     st.set_page_config(layout="wide")  # Responsive layout
#     st.title("YOLOv11n Object Detection App")
    
#     model = load_model()
#     st.sidebar.success("YOLOv11n Model Loaded Successfully")
    
#     uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
#     if uploaded_image is not None:
#         image = Image.open(uploaded_image)
#         st.image(image, caption="Uploaded Image", use_column_width=True)
        
#         if st.button("Run YOLO Detection"):
#             annotated_img, class_counts = run_yolo(model, image)
            
#             st.image(annotated_img, caption="Detected Objects", use_column_width=True)
            
#             st.subheader("Detection Summary")
#             for class_name, count in class_counts.items():
#                 st.write(f"**{class_name}**: {count}")

# if __name__ == "__main__":
#     main()
