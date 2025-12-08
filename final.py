import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import time
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import json
import tempfile
import os
import zipfile
from io import BytesIO

# Configure page
st.set_page_config(
    page_title="Agentic AI Object Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main {
        background-color: #1a1a1a;
        color: #e0e0e0;
    }
    .stApp {
        background-color: #1a1a1a;
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #e0e0e0 !important;
    }
    .stMetric {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3d3d3d;
    }
    .stMetric label {
        color: #a0a0a0 !important;
    }
    .stMetric .metric-value {
        color: #4CAF50 !important;
    }
    .agent-card {
        background-color: #2d2d2d;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 10px 0;
    }
    .feedback-box {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border: 1px solid #3d3d3d;
    }
    .detection-box {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 3px solid #2196F3;
    }
    .stImage {
        border-radius: 10px;
    }
    .batch-image {
        border: 2px solid #3d3d3d;
        border-radius: 8px;
        padding: 5px;
        margin: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detections_history' not in st.session_state:
    st.session_state.detections_history = []
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = defaultdict(int)
if 'model' not in st.session_state:
    st.session_state.model = None
if 'agent_logs' not in st.session_state:
    st.session_state.agent_logs = []
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []
if 'current_model_size' not in st.session_state:
    st.session_state.current_model_size = 'yolov8n.pt'

# Agent Classes
class DetectionAgent:
    """Agent responsible for object detection using YOLOv8"""
    def __init__(self, model_path='yolov8n.pt'):
        self.name = "Detection Agent"
        self.status = "Idle"
        self.model = None
        self.model_path = model_path
        
    def load_model(self, model_path='yolov8n.pt'):
        """Load YOLOv8 model"""
        self.status = "Loading Model"
        self.model_path = model_path
        try:
            self.model = YOLO(model_path)
            self.status = "Ready"
            return True
        except Exception as e:
            self.status = f"Error: {str(e)}"
            return False
    
    def detect(self, image, confidence_threshold=0.25, iou_threshold=0.45, image_size=640, augment=False):
        """Perform object detection with enhanced parameters"""
        self.status = "Detecting Objects"
        try:
            # Convert image to RGB if needed (remove alpha channel)
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run detection with enhanced parameters
            results = self.model(
                image,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=image_size,
                augment=augment,
                verbose=False,
                max_det=300  # Allow more detections
            )
            self.status = "Detection Complete"
            return results
        except Exception as e:
            self.status = f"Error: {str(e)}"
            return None
    
    def batch_detect(self, images, confidence_threshold=0.25, iou_threshold=0.45, image_size=640):
        """Detect objects in multiple images"""
        self.status = "Batch Processing"
        results_list = []
        
        for idx, image in enumerate(images):
            self.status = f"Processing image {idx+1}/{len(images)}"
            result = self.detect(image, confidence_threshold, iou_threshold, image_size)
            if result:
                results_list.append(result)
        
        self.status = "Batch Complete"
        return results_list

class AnalysisAgent:
    """Agent responsible for analyzing detection results"""
    def __init__(self):
        self.name = "Analysis Agent"
        self.status = "Idle"
    
    def analyze_detections(self, results):
        """Analyze detection results and generate insights"""
        self.status = "Analyzing Results"
        try:
            analytics = {
                'total_objects': 0,
                'class_distribution': {},
                'confidence_stats': {},
                'spatial_analysis': {},
                'detection_details': []
            }
            
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue
                    
                analytics['total_objects'] = len(boxes)
                
                # Class distribution
                classes = boxes.cls.cpu().numpy()
                class_names = [result.names[int(cls)] for cls in classes]
                analytics['class_distribution'] = dict(Counter(class_names))
                
                # Confidence statistics
                confidences = boxes.conf.cpu().numpy()
                analytics['confidence_stats'] = {
                    'mean': float(np.mean(confidences)),
                    'min': float(np.min(confidences)),
                    'max': float(np.max(confidences)),
                    'std': float(np.std(confidences))
                }
                
                # Spatial analysis
                xyxy = boxes.xyxy.cpu().numpy()
                areas = [(box[2]-box[0])*(box[3]-box[1]) for box in xyxy]
                analytics['spatial_analysis'] = {
                    'avg_area': float(np.mean(areas)),
                    'total_area': float(np.sum(areas)),
                    'min_area': float(np.min(areas)),
                    'max_area': float(np.max(areas))
                }
                
                # Detailed detection info
                for i, (cls, conf, box) in enumerate(zip(classes, confidences, xyxy)):
                    analytics['detection_details'].append({
                        'id': i,
                        'class': result.names[int(cls)],
                        'confidence': float(conf),
                        'bbox': box.tolist(),
                        'area': float((box[2]-box[0])*(box[3]-box[1]))
                    })
            
            self.status = "Analysis Complete"
            return analytics
        except Exception as e:
            self.status = f"Error: {str(e)}"
            return {}
    
    def analyze_batch(self, batch_results):
        """Analyze multiple detection results"""
        self.status = "Batch Analysis"
        batch_analytics = {
            'total_images': len(batch_results),
            'total_objects': 0,
            'class_distribution': defaultdict(int),
            'avg_confidence': [],
            'images_with_detections': 0
        }
        
        for results in batch_results:
            analytics = self.analyze_detections(results)
            if analytics['total_objects'] > 0:
                batch_analytics['images_with_detections'] += 1
                batch_analytics['total_objects'] += analytics['total_objects']
                
                for cls, count in analytics['class_distribution'].items():
                    batch_analytics['class_distribution'][cls] += count
                
                batch_analytics['avg_confidence'].append(analytics['confidence_stats']['mean'])
        
        batch_analytics['class_distribution'] = dict(batch_analytics['class_distribution'])
        batch_analytics['overall_avg_confidence'] = np.mean(batch_analytics['avg_confidence']) if batch_analytics['avg_confidence'] else 0
        
        self.status = "Batch Analysis Complete"
        return batch_analytics

class LearningAgent:
    """Agent responsible for learning from user feedback"""
    def __init__(self):
        self.name = "Learning Agent"
        self.status = "Idle"
        self.feedback_storage = []
    
    def process_feedback(self, detection_id, feedback_type, comments, corrections):
        """Process and store user feedback"""
        self.status = "Processing Feedback"
        try:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'detection_id': detection_id,
                'feedback_type': feedback_type,
                'comments': comments,
                'corrections': corrections
            }
            self.feedback_storage.append(feedback_entry)
            self.status = "Feedback Processed"
            return feedback_entry
        except Exception as e:
            self.status = f"Error: {str(e)}"
            return None
    
    def get_improvement_suggestions(self):
        """Analyze feedback and suggest improvements"""
        self.status = "Generating Suggestions"
        if not self.feedback_storage:
            return []
        
        suggestions = []
        negative_feedback = [f for f in self.feedback_storage if f['feedback_type'] == 'incorrect']
        
        if len(negative_feedback) > len(self.feedback_storage) * 0.3:
            suggestions.append("⚠️ High number of incorrect detections. Consider:")
            suggestions.append("  • Using a larger model (yolov8m or yolov8l)")
            suggestions.append("  • Lowering confidence threshold to 0.20-0.25")
            suggestions.append("  • Enabling image augmentation")
        
        if len(self.feedback_storage) > 10:
            suggestions.append("✅ Great job! You've provided substantial feedback to improve the system.")
        
        # Analyze missing objects from corrections
        corrections = [f['corrections'] for f in self.feedback_storage if f['corrections']]
        if len(corrections) > 3:
            suggestions.append("📝 Multiple corrections noted. System is learning common misdetections.")
        
        self.status = "Suggestions Ready"
        return suggestions

# Initialize agents
@st.cache_resource
def initialize_agents():
    detection_agent = DetectionAgent()
    analysis_agent = AnalysisAgent()
    learning_agent = LearningAgent()
    return detection_agent, analysis_agent, learning_agent

detection_agent, analysis_agent, learning_agent = initialize_agents()

def draw_boxes_on_image(image, results):
    """Draw bounding boxes on image with proper scaling"""
    try:
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        if len(img_array.shape) == 3 and img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        for result in results:
            img_array = result.plot(
                line_width=3,
                font_size=14,
                labels=True,
                boxes=True,
                conf=True
            )
        
        return img_array
    except Exception as e:
        st.error(f"Error drawing boxes: {str(e)}")
        return image

def process_uploaded_files(uploaded_files):
    """Process multiple uploaded files"""
    images = []
    filenames = []
    
    for uploaded_file in uploaded_files:
        try:
            image = Image.open(uploaded_file)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            images.append(np.array(image))
            filenames.append(uploaded_file.name)
        except Exception as e:
            st.warning(f"Could not process {uploaded_file.name}: {str(e)}")
    
    return images, filenames

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.markdown("### 🎯 Detection Settings")
    
    # Confidence threshold
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Lower values detect more objects but may include false positives. Recommended: 0.20-0.30 for small objects"
    )
    
    # IOU threshold
    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
        help="Controls overlapping box suppression. Lower = fewer overlaps"
    )
    
    # Image size
    image_size = st.select_slider(
        "Input Image Size",
        options=[320, 480, 640, 800, 1024, 1280],
        value=640,
        help="Larger sizes detect smaller objects better but are slower"
    )
    
    # Augmentation
    use_augmentation = st.checkbox(
        "Enable Test-Time Augmentation (TTA)",
        value=False,
        help="Improves accuracy but slower. Good for difficult images"
    )
    
    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    
    model_info = {
        'yolov8n.pt': '⚡ Nano (Fastest)',
        'yolov8s.pt': '🚀 Small (Balanced)',
        'yolov8m.pt': '💪 Medium (Accurate)',
        'yolov8l.pt': '🎯 Large (Very Accurate)',
        'yolov8x.pt': '🏆 Extra Large (Best)'
    }
    
    model_size = st.selectbox(
        "YOLOv8 Model",
        list(model_info.keys()),
        format_func=lambda x: model_info[x],
        index=2,  # Default to medium
        help="Larger models detect small objects better (pen, phone, etc.)"
    )
    
    if st.button("🔄 Load Model", use_container_width=True, type="primary"):
        with st.spinner(f"Loading {model_info[model_size]}..."):
            if detection_agent.load_model(model_size):
                st.success(f"✅ {model_info[model_size]} loaded successfully!")
                st.session_state.model = detection_agent.model
                st.session_state.current_model_size = model_size
            else:
                st.error("❌ Failed to load model. Installing ultralytics...")
                st.code("pip install ultralytics", language="bash")
    
    # Show current model
    if st.session_state.model:
        st.info(f"📦 Current: {model_info.get(st.session_state.current_model_size, 'Unknown')}")
    
    st.markdown("---")
    st.markdown("### 🤖 Agent Status")
    
    agents_info = [
        (detection_agent, "🔍", "#4CAF50"),
        (analysis_agent, "📊", "#2196F3"),
        (learning_agent, "🧠", "#FF9800")
    ]
    
    for agent, icon, color in agents_info:
        st.markdown(f"""
        <div style='background-color: #2d2d2d; padding: 10px; border-radius: 8px; 
                    border-left: 4px solid {color}; margin: 5px 0;'>
            <strong>{icon} {agent.name}</strong><br>
            <small>Status: {agent.status}</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📈 Quick Stats")
    st.metric("Total Detections", len(st.session_state.detections_history))
    st.metric("Feedback Received", len(st.session_state.feedback_data))
    st.metric("Batch Processed", len(st.session_state.batch_results))

# Main content
st.title("🤖 Agentic AI Object Detection System")
st.markdown("**Powered by YOLOv8 with Intelligent Agents | Enhanced for Small Object Detection**")

# Tips banner
st.info("""
💡 **Tips for Better Detection:**
- Use **Medium (yolov8m)** or **Large (yolov8l)** models for small objects like pens, phones
- Lower confidence threshold to **0.20-0.25** for detecting more objects
- Increase image size to **800 or 1024** for small object detection
- Enable **TTA** for difficult images
""")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Detection", "📁 Batch Processing", "📊 Analytics", "💬 Feedback", "🧠 Agent Logs"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Image or Use Webcam")
        
        input_method = st.radio(
            "Select Input Method",
            ["Upload Image", "Use Webcam"],
            horizontal=True
        )
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                help="Upload an image for object detection"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                    
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                if st.button("🚀 Run Detection", use_container_width=True, type="primary"):
                    if st.session_state.model is None:
                        st.error("⚠️ Please load the model first from the sidebar!")
                    else:
                        with st.spinner("🔍 Detection Agent is processing..."):
                            img_array = np.array(image)
                            
                            results = detection_agent.detect(
                                img_array, 
                                confidence_threshold,
                                iou_threshold,
                                image_size,
                                use_augmentation
                            )
                            
                            if results:
                                img_with_boxes = draw_boxes_on_image(img_array, results)
                                st.session_state.processed_image = img_with_boxes
                                
                                analytics = analysis_agent.analyze_detections(results)
                                
                                detection_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.session_state.detections_history.append({
                                    'id': detection_id,
                                    'timestamp': datetime.now(),
                                    'results': results,
                                    'analytics': analytics,
                                    'image': img_with_boxes,
                                    'filename': uploaded_file.name
                                })
                                
                                for cls, count in analytics['class_distribution'].items():
                                    st.session_state.analytics_data[cls] += count
                                
                                st.success(f"✅ Detection complete! Found {analytics['total_objects']} objects")
                                st.rerun()
        
        else:  # Webcam
            st.markdown("#### 📷 Webcam Capture")
            camera_photo = st.camera_input("Take a picture")
            
            if camera_photo is not None:
                image = Image.open(camera_photo)
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                elif image.mode != 'RGB':
                    image = image.convert('RGB')
                
                if st.button("🚀 Detect from Webcam", use_container_width=True, type="primary"):
                    if st.session_state.model is None:
                        st.error("⚠️ Please load the model first from the sidebar!")
                    else:
                        with st.spinner("🔍 Detection Agent is processing webcam image..."):
                            img_array = np.array(image)
                            
                            results = detection_agent.detect(
                                img_array,
                                confidence_threshold,
                                iou_threshold,
                                image_size,
                                use_augmentation
                            )
                            
                            if results:
                                img_with_boxes = draw_boxes_on_image(img_array, results)
                                st.session_state.processed_image = img_with_boxes
                                
                                analytics = analysis_agent.analyze_detections(results)
                                
                                detection_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                                st.session_state.detections_history.append({
                                    'id': detection_id,
                                    'timestamp': datetime.now(),
                                    'results': results,
                                    'analytics': analytics,
                                    'image': img_with_boxes,
                                    'filename': 'webcam_capture'
                                })
                                
                                for cls, count in analytics['class_distribution'].items():
                                    st.session_state.analytics_data[cls] += count
                                
                                st.success(f"✅ Detection complete! Found {analytics['total_objects']} objects")
                                st.rerun()
    
    with col2:
        st.markdown("### Detection Results")
        
        if st.session_state.processed_image is not None:
            st.image(
                st.session_state.processed_image,
                caption="Detected Objects",
                use_column_width=True,
                channels="RGB"
            )
            
            if st.session_state.detections_history:
                latest = st.session_state.detections_history[-1]
                analytics = latest['analytics']
                
                # Summary metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Objects", analytics['total_objects'])
                with col_b:
                    st.metric("Classes", len(analytics['class_distribution']))
                with col_c:
                    st.metric("Avg Conf", f"{analytics['confidence_stats'].get('mean', 0):.1%}")
                
                st.markdown("### 📋 Detected Objects")
                
                detection_container = st.container()
                with detection_container:
                    results = latest['results']
                    for result in results:
                        boxes = result.boxes
                        if len(boxes) == 0:
                            st.info("No objects detected above confidence threshold.")
                            continue
                        
                        # Group by class
                        class_groups = defaultdict(list)
                        for i, box in enumerate(boxes):
                            cls = int(box.cls[0])
                            conf = float(box.conf[0])
                            label = result.names[cls]
                            xyxy = box.xyxy[0].cpu().numpy()
                            
                            class_groups[label].append({
                                'id': i,
                                'conf': conf,
                                'bbox': xyxy
                            })
                        
                        # Display grouped
                        for label, detections in sorted(class_groups.items()):
                            with st.expander(f"**{label}** ({len(detections)} found)", expanded=True):
                                for det in detections:
                                    conf_color = "#4CAF50" if det['conf'] > 0.7 else "#FFC107" if det['conf'] > 0.4 else "#FF5722"
                                    st.markdown(f"""
                                    <div style='background-color: #2d2d2d; padding: 8px; border-radius: 5px; 
                                                margin: 3px 0; border-left: 3px solid {conf_color};'>
                                        <strong>Detection #{det['id']+1}</strong><br>
                                        Confidence: <strong>{det['conf']:.1%}</strong><br>
                                        <small>BBox: ({int(det['bbox'][0])}, {int(det['bbox'][1])}) → 
                                        ({int(det['bbox'][2])}, {int(det['bbox'][3])})</small>
                                    </div>
                                    """, unsafe_allow_html=True)
        else:
            st.info("🎯 No detections yet. Upload an image or use webcam and run detection!")

with tab2:
    st.markdown("### 📁 Batch Processing - Multiple Images")
    st.markdown("Upload multiple images or a COCO-style dataset for batch detection")
    
    uploaded_files = st.file_uploader(
        "Choose multiple images...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
        accept_multiple_files=True,
        help="Select multiple images for batch processing"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Process All Images", use_container_width=True, type="primary"):
                if st.session_state.model is None:
                    st.error("⚠️ Please load the model first from the sidebar!")
                else:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    images, filenames = process_uploaded_files(uploaded_files)
                    
                    batch_results_temp = []
                    batch_images_with_boxes = []
                    
                    for idx, (image, filename) in enumerate(zip(images, filenames)):
                        status_text.text(f"Processing {idx+1}/{len(images)}: {filename}")
                        progress_bar.progress((idx + 1) / len(images))
                        
                        results = detection_agent.detect(
                            image,
                            confidence_threshold,
                            iou_threshold,
                            image_size,
                            use_augmentation
                        )
                        
                        if results:
                            img_with_boxes = draw_boxes_on_image(image, results)
                            analytics = analysis_agent.analyze_detections(results)
                            
                            batch_results_temp.append({
                                'filename': filename,
                                'results': results,
                                'analytics': analytics,
                                'image': img_with_boxes
                            })
                            
                            for cls, count in analytics['class_distribution'].items():
                                st.session_state.analytics_data[cls] += count
                    
                    st.session_state.batch_results = batch_results_temp
                    
                    # Batch analytics
                    batch_analytics = analysis_agent.analyze_batch([r['results'] for r in batch_results_temp])
                    
                    status_text.empty()
                    progress_bar.empty()
                    
                    st.success(f"✅ Batch processing complete! Processed {len(batch_results_temp)} images")
                    st.balloons()
                    
                    # Display batch summary
                    st.markdown("### 📊 Batch Summary")
                    col_a, col_b, col_c, col_d = st.columns(4)
                    with col_a:
                        st.metric("Total Images", batch_analytics['total_images'])
                    with col_b:
                        st.metric("Total Objects", batch_analytics['total_objects'])
                    with col_c:
                        st.metric("Avg Confidence", f"{batch_analytics['overall_avg_confidence']:.1%}")
                    with col_d:
                        st.metric("Detection Rate", f"{batch_analytics['images_with_detections']}/{batch_analytics['total_images']}")
        
        with col2:
            if st.button("🗑️ Clear Batch", use_container_width=True):
                st.session_state.batch_results = []
                st.rerun()
    
    # Display batch results
    if st.session_state.batch_results:
        st.markdown("---")
        st.markdown("### 📸 Batch Results Gallery")
        
        # Image grid
        cols_per_row = 3
        for i in range(0, len(st.session_state.batch_results), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(st.session_state.batch_results):
                    result = st.session_state.batch_results[idx]
                    with col:
                        st.image(
                            result['image'],
                            caption=f"{result['filename']}\n{result['analytics']['total_objects']} objects",
                            use_column_width=True,
                            channels="RGB"
                        )
                        
                        with st.expander("Details"):
                            st.write(f"**Objects:** {result['analytics']['total_objects']}")
                            st.write(f"**Classes:** {', '.join(result['analytics']['class_distribution'].keys())}")
                            st.write(f"**Avg Conf:** {result['analytics']['confidence_stats'].get('mean', 0):.1%}")

with tab3:
    st.markdown("### 📊 Detection Analytics")
    
    if st.session_state.detections_history or st.session_state.batch_results:
        
        # Choose analysis source
        analysis_source = st.radio(
            "Analysis Source",
            ["Latest Detection", "Batch Results", "All Time"],
            horizontal=True
        )
        
        if analysis_source == "Latest Detection" and st.session_state.detections_history:
            latest = st.session_state.detections_history[-1]
            analytics = latest['analytics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Objects", analytics['total_objects'])
            with col2:
                st.metric("Unique Classes", len(analytics['class_distribution']))
            with col3:
                st.metric("Avg Confidence", f"{analytics['confidence_stats'].get('mean', 0):.1%}")
            with col4:
                st.metric("Max Confidence", f"{analytics['confidence_stats'].get('max', 0):.1%}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Class Distribution")
                if analytics['class_distribution']:
                    df = pd.DataFrame(
                        list(analytics['class_distribution'].items()),
                        columns=['Class', 'Count']
                    ).sort_values('Count', ascending=False)
                    
                    fig = px.bar(
                        df,
                        x='Class',
                        y='Count',
                        color='Count',
                        color_continuous_scale='Viridis',
                        text='Count'
                    )
                    fig.update_layout(
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d',
                        font_color='#e0e0e0',
                        xaxis_title="Object Class",
                        yaxis_title="Count",
                        showlegend=False
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Confidence Distribution")
                if analytics['detection_details']:
                    df_conf = pd.DataFrame(analytics['detection_details'])
                    fig = px.histogram(
                        df_conf,
                        x='confidence',
                        nbins=20,
                        color_discrete_sequence=['#4CAF50']
                    )
                    fig.update_layout(
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d',
                        font_color='#e0e0e0',
                        xaxis_title="Confidence Score",
                        yaxis_title="Count",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_source == "Batch Results" and st.session_state.batch_results:
            batch_analytics = analysis_agent.analyze_batch([r['results'] for r in st.session_state.batch_results])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Images", batch_analytics['total_images'])
            with col2:
                st.metric("Total Objects", batch_analytics['total_objects'])
            with col3:
                st.metric("Avg Objects/Image", f"{batch_analytics['total_objects']/batch_analytics['total_images']:.1f}")
            with col4:
                st.metric("Detection Rate", f"{batch_analytics['images_with_detections']}/{batch_analytics['total_images']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Batch Class Distribution")
                if batch_analytics['class_distribution']:
                    df = pd.DataFrame(
                        list(batch_analytics['class_distribution'].items()),
                        columns=['Class', 'Total Count']
                    ).sort_values('Total Count', ascending=False).head(15)
                    
                    fig = px.bar(
                        df,
                        x='Class',
                        y='Total Count',
                        color='Total Count',
                        color_continuous_scale='Plasma',
                        text='Total Count'
                    )
                    fig.update_layout(
                        paper_bgcolor='#1a1a1a',
                        plot_bgcolor='#2d2d2d',
                        font_color='#e0e0e0'
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("#### Top Detected Classes")
                if batch_analytics['class_distribution']:
                    df_pie = pd.DataFrame(
                        list(batch_analytics['class_distribution'].items()),
                        columns=['Class', 'Count']
                    ).sort_values('Count', ascending=False).head(10)
                    
                    fig = px.pie(
                        df_pie,
                        values='Count',
                        names='Class',
                        hole=0.4
                    )
                    fig.update_layout(
                        paper_bgcolor='#1a1a1a',
                        font_color='#e0e0e0'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:  # All Time
            if st.session_state.analytics_data:
                total_objects = sum(st.session_state.analytics_data.values())
                unique_classes = len(st.session_state.analytics_data)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Objects (All Time)", total_objects)
                with col2:
                    st.metric("Unique Classes", unique_classes)
                with col3:
                    st.metric("Total Sessions", len(st.session_state.detections_history))
                
                st.markdown("#### All-Time Class Distribution")
                df_total = pd.DataFrame(
                    list(st.session_state.analytics_data.items()),
                    columns=['Class', 'Total Count']
                ).sort_values('Total Count', ascending=False).head(20)
                
                fig = px.bar(
                    df_total,
                    x='Class',
                    y='Total Count',
                    color='Total Count',
                    color_continuous_scale='Turbo'
                )
                fig.update_layout(
                    paper_bgcolor='#1a1a1a',
                    plot_bgcolor='#2d2d2d',
                    font_color='#e0e0e0',
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run a detection to see analytics!")

with tab4:
    st.markdown("### 💬 Feedback System")
    
    if st.session_state.detections_history:
        st.markdown("#### Provide Feedback on Latest Detection")
        
        latest_id = st.session_state.detections_history[-1]['id']
        
        col1, col2 = st.columns(2)
        
        with col1:
            feedback_type = st.radio(
                "Detection Accuracy",
                ['correct', 'partially_correct', 'incorrect'],
                format_func=lambda x: {
                    'correct': '✅ Correct',
                    'partially_correct': '⚠️ Partially Correct',
                    'incorrect': '❌ Incorrect'
                }[x]
            )
        
        with col2:
            comments = st.text_area(
                "Comments",
                placeholder="Any specific feedback or observations...",
                height=100
            )
        
        corrections = st.text_input(
            "Corrections (if any)",
            placeholder="E.g., 'Missed: pen on desk, book on shelf' or 'False: detected person as car'"
        )
        
        if st.button("Submit Feedback", type="primary", use_container_width=True):
            feedback = learning_agent.process_feedback(
                latest_id,
                feedback_type,
                comments,
                corrections
            )
            st.session_state.feedback_data.append(feedback)
            st.success("✅ Thank you for your feedback! The Learning Agent will use this to improve.")
            time.sleep(1)
            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Feedback History")
        
        if st.session_state.feedback_data:
            for fb in reversed(st.session_state.feedback_data[-5:]):
                icon = {'correct': '✅', 'partially_correct': '⚠️', 'incorrect': '❌'}
                st.markdown(f"""
                <div class='feedback-box'>
                    <strong>{icon[fb['feedback_type']]} {fb['timestamp'][:19]}</strong><br>
                    <small>Detection ID: {fb['detection_id']}</small><br>
                    {fb['comments'] if fb['comments'] else 'No comments'}<br>
                    {('<small><strong>Corrections:</strong> ' + fb['corrections'] + '</small>') if fb['corrections'] else ''}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feedback yet!")
        
        st.markdown("---")
        st.markdown("#### 🧠 Learning Agent Suggestions")
        suggestions = learning_agent.get_improvement_suggestions()
        
        if suggestions:
            for suggestion in suggestions:
                if suggestion.startswith('⚠️'):
                    st.warning(suggestion)
                elif suggestion.startswith('✅'):
                    st.success(suggestion)
                else:
                    st.info(suggestion)
        else:
            st.success("✨ System performance is good! Keep providing feedback to help improve.")
    
    else:
        st.info("Run a detection first to provide feedback!")

with tab5:
    st.markdown("### 🧠 Agent Activity Logs")
    
    st.markdown("#### Agent Workflow")
    st.code("""
    1. 🔍 Detection Agent
       ↓ Loads YOLOv8 model (nano/small/medium/large/x)
       ↓ Processes image with optimized parameters
       ↓ Applies confidence & IoU thresholds
       ↓ Returns bounding boxes, classes & scores
    
    2. 📊 Analysis Agent
       ↓ Analyzes detection results
       ↓ Generates class distribution
       ↓ Calculates confidence statistics
       ↓ Creates spatial analysis & insights
    
    3. 🧠 Learning Agent
       ↓ Collects user feedback
       ↓ Identifies detection patterns
       ↓ Analyzes missed objects
       ↓ Suggests model/parameter improvements
    """, language="text")
    
    st.markdown("#### Current Agent States")
    
    agents_data = {
        'Agent': ['Detection Agent', 'Analysis Agent', 'Learning Agent'],
        'Status': [detection_agent.status, analysis_agent.status, learning_agent.status],
        'Icon': ['🔍', '📊', '🧠']
    }
    
    df_agents = pd.DataFrame(agents_data)
    st.dataframe(df_agents, use_container_width=True, hide_index=True)
    
    st.markdown("#### Activity Timeline")
    
    if st.session_state.detections_history or st.session_state.feedback_data:
        timeline_data = []
        
        for detection in st.session_state.detections_history[-15:]:
            timeline_data.append({
                'Time': detection['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                'Event': f"Detection: {detection['analytics']['total_objects']} objects - {detection.get('filename', 'N/A')}",
                'Agent': 'Detection Agent',
                'Type': 'Detection'
            })
        
        for fb in st.session_state.feedback_data[-10:]:
            timeline_data.append({
                'Time': fb['timestamp'][:19],
                'Event': f"Feedback: {fb['feedback_type']}",
                'Agent': 'Learning Agent',
                'Type': 'Feedback'
            })
        
        df_timeline = pd.DataFrame(timeline_data).sort_values('Time', ascending=False)
        st.dataframe(df_timeline, use_container_width=True, hide_index=True)
    else:
        st.info("No activity yet!")
    
    st.markdown("#### System Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_detections = sum(d['analytics']['total_objects'] for d in st.session_state.detections_history)
        st.metric("Total Objects Detected", total_detections)
    
    with col2:
        avg_conf = np.mean([d['analytics']['confidence_stats'].get('mean', 0) 
                           for d in st.session_state.detections_history]) if st.session_state.detections_history else 0
        st.metric("Avg Confidence Score", f"{avg_conf:.1%}")
    
    with col3:
        feedback_accuracy = len([f for f in st.session_state.feedback_data if f['feedback_type'] == 'correct']) / len(st.session_state.feedback_data) * 100 if st.session_state.feedback_data else 0
        st.metric("Feedback Accuracy", f"{feedback_accuracy:.1f}%")
    
    with col4:
        st.metric("Model Size", model_info.get(st.session_state.current_model_size, 'Not Loaded'))

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888;'>
    <p>🤖 Agentic AI Object Detection System | Powered by YOLOv8 & Streamlit</p>
    <p><small>💡 For best results with small objects: Use yolov8m/yolov8l, confidence 0.20-0.25, image size 800+</small></p>
</div>
""", unsafe_allow_html=True)