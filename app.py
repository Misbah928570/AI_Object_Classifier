import streamlit as st
import cv2
import numpy as np
import pyttsx3
import threading
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
import logging
import json
import requests
from typing import List, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="YOLOv8 AI Agent Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling (keeping original styles)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .agent-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .feedback-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .stat-card {
        background: linear-gradient(135deg, #8B7BAF 0%, #6B5B95 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.15);
        transition: transform 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
    }
    
    .stat-card h3 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #E8E4F3 0%, #D3CCE3 100%);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 5px 20px rgba(0,0,0,0.15);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .improvement-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        margin: 0.3rem;
        font-size: 0.9rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
</style>
""", unsafe_allow_html=True)

class AIAgent:
    """AI Agent for intelligent analysis and feedback"""
    
    def __init__(self):
        self.api_url = "https://api.anthropic.com/v1/messages"
        
    def analyze_detections(self, detections: List[Dict], image_context: str = "") -> Dict:
        """Analyze detections using Claude AI"""
        try:
            # Prepare detection summary
            detection_summary = self._prepare_detection_summary(detections)
            
            prompt = f"""You are an expert computer vision and object detection analyst with deep knowledge of YOLOv8 architecture, COCO dataset classes, and real-world detection challenges.

DETECTION RESULTS:
{detection_summary}

IMAGE CONTEXT: {image_context}

Perform a comprehensive expert-level analysis. Consider:
- YOLOv8's strengths and limitations with these object classes
- COCO dataset training distribution and class imbalances
- Lighting, occlusion, scale, and perspective issues
- Inter-object relationships and scene composition
- Detection confidence patterns and what they indicate

Provide expert analysis in JSON format with these fields:
1. "overall_assessment": Detailed professional assessment of detection quality and reliability
2. "confidence_analysis": Deep dive into why confidence scores are what they are, considering model training and scene factors
3. "scene_understanding": Comprehensive scene interpretation including object relationships, spatial layout, and context
4. "recommendations": 5-7 highly specific, actionable recommendations prioritized by impact
5. "potential_issues": Detailed technical issues including model limitations, dataset biases, and environmental factors
6. "detection_insights": Expert insights about what the model is seeing vs what might actually be present
7. "class_specific_notes": Notes about specific object classes detected and their typical detection challenges

Provide expert-level, technical analysis suitable for computer vision professionals.
Return ONLY valid JSON, no other text."""

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['content'][0]['text']
                
                # Clean and parse JSON
                content = content.replace('```json', '').replace('```', '').strip()
                analysis = json.loads(content)
                return analysis
            else:
                logger.error(f"API error: {response.status_code}")
                return self._get_fallback_analysis(detections)
                
        except Exception as e:
            logger.error(f"Agent analysis error: {str(e)}")
            return self._get_fallback_analysis(detections)
    
    def feedback_loop_analysis(self, low_confidence_detections: List[Dict], 
                               image_description: str = "") -> Dict:
        """Provide feedback for low confidence detections"""
        try:
            low_conf_summary = "\n".join([
                f"- {d['object']}: {d['confidence']*100:.1f}% confidence (Class ID: {d.get('class_id', 'N/A')})"
                for d in low_confidence_detections
            ])
            
            prompt = f"""You are a senior computer vision engineer specializing in YOLOv8 optimization and detection debugging. Analyze these LOW CONFIDENCE detections that require improvement:

{low_conf_summary}

CONTEXT: {image_description}

Perform deep technical analysis considering:
- YOLOv8 architecture limitations and anchor box configurations
- COCO dataset class distribution and training biases
- Common causes: poor lighting, occlusion, scale issues, angle/perspective problems
- Background complexity and texture similarity
- Object aspect ratios and unusual poses
- Model confidence calibration issues

Provide expert technical feedback in JSON format:
1. "issues_identified": Detailed list of specific technical problems causing low confidence, with root cause analysis
2. "improvement_strategies": Concrete, prioritized steps to improve detection with expected impact levels
3. "alternative_interpretations": What these objects might actually be, considering visual similarity and confusion matrices
4. "data_recommendations": Specific training data augmentation strategies, including quantity, variety, and annotation requirements
5. "preprocessing_suggestions": Advanced image preprocessing techniques with parameters (e.g., CLAHE settings, denoising methods)
6. "model_tuning_advice": Specific model hyperparameters to adjust (confidence threshold, NMS IOU, augmentation)
7. "environmental_factors": Lighting, camera settings, and scene composition improvements
8. "quick_wins": Immediate actionable steps ranked by ease of implementation

Provide graduate-level computer vision expertise suitable for ML engineers.
Return ONLY valid JSON."""

            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 2000,
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data['content'][0]['text']
                content = content.replace('```json', '').replace('```', '').strip()
                feedback = json.loads(content)
                return feedback
            else:
                return self._get_fallback_feedback(low_confidence_detections)
                
        except Exception as e:
            logger.error(f"Feedback loop error: {str(e)}")
            return self._get_fallback_feedback(low_confidence_detections)
    
    def _prepare_detection_summary(self, detections: List[Dict]) -> str:
        """Prepare detection summary for AI analysis"""
        if not detections:
            return "No objects detected"
        
        summary = f"Total detections: {len(detections)}\n\n"
        for i, det in enumerate(detections[:15], 1):
            summary += f"{i}. {det['object']}: {det['confidence']*100:.1f}% confidence\n"
        
        high_conf = sum(1 for d in detections if d['confidence'] >= 0.7)
        medium_conf = sum(1 for d in detections if 0.5 <= d['confidence'] < 0.7)
        low_conf = sum(1 for d in detections if d['confidence'] < 0.5)
        
        summary += f"\nConfidence distribution:\n"
        summary += f"- High (≥70%): {high_conf}\n"
        summary += f"- Medium (50-70%): {medium_conf}\n"
        summary += f"- Low (<50%): {low_conf}"
        
        return summary
    
    def _get_fallback_analysis(self, detections: List[Dict]) -> Dict:
        """Fallback analysis when API fails"""
        avg_conf = np.mean([d['confidence'] for d in detections]) * 100
        low_conf_count = sum(1 for d in detections if d['confidence'] < 0.5)
        
        return {
            "overall_assessment": f"Detected {len(detections)} objects with {avg_conf:.1f}% average confidence",
            "confidence_analysis": f"{low_conf_count} detections have low confidence and may need review",
            "scene_understanding": "Multiple objects detected in the scene",
            "recommendations": [
                "Review low confidence detections manually",
                "Consider adjusting confidence threshold",
                "Ensure good lighting conditions"
            ],
            "potential_issues": ["Some detections may require verification"]
        }
    
    def _get_fallback_feedback(self, detections: List[Dict]) -> Dict:
        """Fallback feedback when API fails"""
        return {
            "issues_identified": ["Low confidence scores indicate uncertainty"],
            "improvement_strategies": [
                "Improve image quality and lighting",
                "Use higher resolution images",
                "Ensure objects are clearly visible"
            ],
            "alternative_interpretations": ["Objects may be partially occluded or unclear"],
            "data_recommendations": ["Additional training data for these object types"],
            "preprocessing_suggestions": ["Try contrast enhancement", "Noise reduction"]
        }


class YOLOv8DetectionSystem:
    def __init__(self):
        """Initialize YOLOv8 Detection System with AI Agent"""
        self.model = None
        self.voice_engine = None
        self.confidence_threshold = 0.25
        self.ai_agent = AIAgent()
        self.initialize_model()
        self.initialize_voice()
    
    def initialize_model(self):
        """Load YOLOv8 model"""
        try:
            with st.spinner("🔄 Loading YOLOv8 AI Model..."):
                # Try to load a larger model for better detection
                try:
                    self.model = YOLO('yolov8m.pt')  # Medium model - better accuracy
                    st.success("✅ YOLOv8 Medium Model Loaded Successfully!")
                except:
                    self.model = YOLO('yolov8n.pt')  # Fallback to nano
                    st.success("✅ YOLOv8 Nano Model Loaded Successfully!")
                logger.info("YOLOv8 initialized")
        except Exception as e:
            st.error(f"❌ Model Loading Failed: {str(e)}")
            logger.error(f"Model error: {str(e)}")
    
    def initialize_voice(self):
        """Initialize voice engine"""
        try:
            self.voice_engine = pyttsx3.init()
            self.voice_engine.setProperty('rate', 150)
            self.voice_engine.setProperty('volume', 1.0)
            logger.info("Voice engine ready")
        except Exception as e:
            logger.warning(f"Voice unavailable: {str(e)}")
            self.voice_engine = None
    
    def speak(self, text):
        """Text to speech"""
        if self.voice_engine:
            def async_speak():
                try:
                    self.voice_engine.say(text)
                    self.voice_engine.runAndWait()
                except:
                    pass
            threading.Thread(target=async_speak, daemon=True).start()
    
    def detect_objects(self, image):
        """Detect objects using YOLOv8"""
        try:
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Run inference with lower confidence for initial detection
            results = self.model(image, conf=self.confidence_threshold, iou=0.45, max_det=300)
            
            detections = []
            annotated_image = image.copy()
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = result.names[cls]
                    
                    detections.append({
                        'object': label.title(),
                        'confidence': conf,
                        'box': (x1, y1, x2, y2),
                        'class_id': cls
                    })
                    
                    # Draw boxes with color coding
                    color = (0, 255, 0) if conf > 0.7 else (255, 165, 0) if conf > 0.5 else (255, 0, 0)
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
                    
                    label_text = f"{label}: {conf:.2f}"
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated_image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                    cv2.putText(annotated_image, label_text, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            return detections, annotated_image
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return [], image
    
    def get_low_confidence_detections(self, detections: List[Dict], 
                                     threshold: float = 0.5) -> List[Dict]:
        """Extract low confidence detections for feedback loop"""
        return [d for d in detections if d['confidence'] < threshold]


def create_confidence_table(detections):
    """Create styled confidence table"""
    if not detections:
        return None
    
    df_data = []
    for idx, det in enumerate(detections, 1):
        conf_percent = det['confidence'] * 100
        if conf_percent >= 70:
            status = "🟢 High"
        elif conf_percent >= 50:
            status = "🟡 Medium"
        else:
            status = "🔴 Low"
        
        df_data.append({
            'Rank': f"#{idx}",
            'Object': det['object'],
            'Confidence': f"{conf_percent:.2f}%",
            'Status': status
        })
    
    return pd.DataFrame(df_data)


def create_summary_table(detections: List[Dict], analysis: Dict, feedback: Dict = None) -> pd.DataFrame:
    """Create comprehensive summary table of entire detection session"""
    
    # Calculate statistics
    total_detections = len(detections)
    unique_objects = len(set(d['object'] for d in detections))
    avg_confidence = np.mean([d['confidence'] for d in detections]) * 100 if detections else 0
    
    high_conf = sum(1 for d in detections if d['confidence'] >= 0.7)
    medium_conf = sum(1 for d in detections if 0.5 <= d['confidence'] < 0.7)
    low_conf = sum(1 for d in detections if d['confidence'] < 0.5)
    
    # Object frequency
    object_counts = {}
    for det in detections:
        obj = det['object']
        object_counts[obj] = object_counts.get(obj, 0) + 1
    
    top_objects = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Create summary data
    summary_data = {
        'Metric': [
            'Total Objects Detected',
            'Unique Object Types',
            'Average Confidence',
            'High Confidence Detections (≥70%)',
            'Medium Confidence Detections (50-70%)',
            'Low Confidence Detections (<50%)',
            'Most Detected Object',
            'Detection Quality Score',
            'AI Assessment',
            'Primary Recommendation'
        ],
        'Value': [
            str(total_detections),
            str(unique_objects),
            f"{avg_confidence:.2f}%",
            f"{high_conf} ({high_conf/total_detections*100:.1f}%)" if total_detections > 0 else "0",
            f"{medium_conf} ({medium_conf/total_detections*100:.1f}%)" if total_detections > 0 else "0",
            f"{low_conf} ({low_conf/total_detections*100:.1f}%)" if total_detections > 0 else "0",
            f"{top_objects[0][0]} ({top_objects[0][1]}x)" if top_objects else "N/A",
            f"{'Excellent' if avg_confidence >= 80 else 'Good' if avg_confidence >= 60 else 'Fair' if avg_confidence >= 40 else 'Needs Improvement'} ({avg_confidence:.1f}%)",
            analysis.get('overall_assessment', 'N/A')[:100] + '...' if len(analysis.get('overall_assessment', '')) > 100 else analysis.get('overall_assessment', 'N/A'),
            analysis.get('recommendations', ['N/A'])[0] if analysis.get('recommendations') else 'N/A'
        ]
    }
    
    return pd.DataFrame(summary_data)


def create_object_frequency_table(detections: List[Dict]) -> pd.DataFrame:
    """Create table showing frequency of each detected object type"""
    if not detections:
        return None
    
    object_counts = {}
    confidence_sums = {}
    
    for det in detections:
        obj = det['object']
        object_counts[obj] = object_counts.get(obj, 0) + 1
        confidence_sums[obj] = confidence_sums.get(obj, 0) + det['confidence']
    
    freq_data = []
    for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
        avg_conf = (confidence_sums[obj] / count) * 100
        freq_data.append({
            'Object Type': obj,
            'Count': count,
            'Percentage': f"{(count/len(detections)*100):.1f}%",
            'Avg Confidence': f"{avg_conf:.2f}%",
            'Status': '🟢' if avg_conf >= 70 else '🟡' if avg_conf >= 50 else '🔴'
        })
    
    return pd.DataFrame(freq_data)


def display_summary_section(detections: List[Dict], analysis: Dict, feedback: Dict = None):
    """Display comprehensive summary section at the end"""
    st.markdown("---")
    st.markdown("## 📋 Comprehensive Detection Summary")
    
    # Summary Table
    st.markdown("### 📊 Session Summary")
    summary_df = create_summary_table(detections, analysis, feedback)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Object Frequency Table
    st.markdown("### 🔢 Object Frequency Analysis")
    freq_df = create_object_frequency_table(detections)
    if freq_df is not None:
        st.dataframe(freq_df, use_container_width=True, hide_index=True)
    
    # Key Insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        most_common = freq_df.iloc[0]['Object Type'] if freq_df is not None and len(freq_df) > 0 else "N/A"
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="margin: 0;">🎯 Most Common</h4>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">{most_common}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        detection_quality = "Excellent" if np.mean([d['confidence'] for d in detections]) >= 0.8 else "Good" if np.mean([d['confidence'] for d in detections]) >= 0.6 else "Fair"
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="margin: 0;">⭐ Quality</h4>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">{detection_quality}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        reliability = f"{sum(1 for d in detections if d['confidence'] >= 0.7) / len(detections) * 100:.0f}%"
        st.markdown(f"""
        <div class="stat-card">
            <h4 style="margin: 0;">✅ Reliability</h4>
            <p style="font-size: 1.3rem; margin: 0.5rem 0;">{reliability}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Final Recommendations
    st.markdown("### 💡 Final Recommendations")
    recommendations = analysis.get('recommendations', [])[:5]
    for i, rec in enumerate(recommendations, 1):
        st.markdown(f"**{i}.** {rec}")
    
    # Export Summary
    st.markdown("### 💾 Export Complete Summary")
    
    summary_text = f"""
{'='*80}
YOLOV8 AI AGENT DETECTION SYSTEM - COMPREHENSIVE SUMMARY
{'='*80}

DETECTION STATISTICS:
--------------------
Total Objects Detected: {len(detections)}
Unique Object Types: {len(set(d['object'] for d in detections))}
Average Confidence: {np.mean([d['confidence'] for d in detections])*100:.2f}%

CONFIDENCE DISTRIBUTION:
-----------------------
High (≥70%): {sum(1 for d in detections if d['confidence'] >= 0.7)}
Medium (50-70%): {sum(1 for d in detections if 0.5 <= d['confidence'] < 0.7)}
Low (<50%): {sum(1 for d in detections if d['confidence'] < 0.5)}

DETAILED DETECTIONS:
-------------------
"""
    for i, det in enumerate(detections, 1):
        summary_text += f"{i}. {det['object']}: {det['confidence']*100:.2f}%\n"
    
    summary_text += f"""

AI AGENT ANALYSIS:
-----------------
{analysis.get('overall_assessment', 'N/A')}

KEY RECOMMENDATIONS:
-------------------
"""
    for i, rec in enumerate(recommendations, 1):
        summary_text += f"{i}. {rec}\n"
    
    summary_text += f"\n{'='*80}\n"
    
    st.download_button(
        "📥 Download Complete Summary Report",
        summary_text,
        file_name=f"detection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


def display_agent_analysis(analysis: Dict, detections: List[Dict]):
    """Display AI agent analysis with confidence table"""
    st.markdown("### 🤖 AI Agent Analysis")
    
    # Overall Assessment
    st.markdown(f"""
    <div class="agent-card">
        <h3 style="margin-top: 0;">🎯 Overall Assessment</h3>
        <p style="font-size: 1.1rem; line-height: 1.6;">{analysis.get('overall_assessment', 'No assessment available')}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence Table
    st.markdown("#### 📊 Confidence Levels Table")
    conf_table = create_confidence_table(detections)
    if conf_table is not None:
        st.dataframe(conf_table, use_container_width=True, hide_index=True)
    
    # Recommendations
    recommendations = analysis.get('recommendations', [])
    if recommendations:
        st.markdown("#### 💡 AI Recommendations")
        cols = st.columns(min(len(recommendations), 3))
        for i, rec in enumerate(recommendations):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="improvement-badge">
                    {rec}
                </div>
                """, unsafe_allow_html=True)
    
    # Issues
    issues = analysis.get('potential_issues', [])
    if issues:
        st.warning("⚠️ Potential Issues: " + ", ".join(issues))


def display_feedback_loop(feedback: Dict, low_conf_detections: List[Dict]):
    """Display feedback loop analysis"""
    st.markdown("### 🔄 Feedback Loop: Low Confidence Analysis")
    
    # Issues Identified
    st.markdown("#### 🔍 Issues Identified")
    issues = feedback.get('issues_identified', [])
    for issue in issues:
        st.error(f"❌ {issue}")
    
    # Improvement Strategies
    st.markdown("#### 🚀 Improvement Strategies")
    strategies = feedback.get('improvement_strategies', [])
    for strategy in strategies:
        st.success(f"✅ {strategy}")
    
    # Quick Wins
    quick_wins = feedback.get('quick_wins', [])
    if quick_wins:
        st.markdown("#### ⚡ Quick Wins (Easy to Implement)")
        for win in quick_wins:
            st.info(f"💡 {win}")
    
    # Alternative Interpretations
    st.markdown("#### 🤔 Alternative Interpretations")
    alternatives = feedback.get('alternative_interpretations', [])
    for alt in alternatives:
        st.info(f"💭 {alt}")
    
    # Show low confidence detections
    st.markdown("#### 📉 Low Confidence Detections Requiring Review")
    low_conf_df = pd.DataFrame([
        {
            'Object': d['object'],
            'Confidence': f"{d['confidence']*100:.2f}%",
            'Class ID': d.get('class_id', 'N/A'),
            'Status': '⚠️ Needs Review'
        }
        for d in low_conf_detections
    ])
    
    if not low_conf_df.empty:
        st.dataframe(low_conf_df, use_container_width=True, hide_index=True)
    
    # Technical recommendations
    with st.expander("🔧 Advanced Technical Recommendations"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📚 Data Recommendations:**")
            for rec in feedback.get('data_recommendations', []):
                st.write(f"• {rec}")
            
            st.markdown("**🎛️ Model Tuning Advice:**")
            for advice in feedback.get('model_tuning_advice', []):
                st.write(f"• {advice}")
        
        with col2:
            st.markdown("**🖼️ Preprocessing Suggestions:**")
            for sug in feedback.get('preprocessing_suggestions', []):
                st.write(f"• {sug}")
            
            st.markdown("**🌟 Environmental Factors:**")
            for factor in feedback.get('environmental_factors', []):
                st.write(f"• {factor}")


def create_confidence_graph(detections):
    """Create interactive confidence graph"""
    if not detections:
        return None
    
    objects = [d['object'] for d in detections[:10]]
    confidences = [d['confidence'] * 100 for d in detections[:10]]
    
    colors = ['#11998e' if c >= 70 else '#f2994a' if c >= 50 else '#eb3349' for c in confidences]
    
    fig = go.Figure(data=[
        go.Bar(
            x=objects,
            y=confidences,
            text=[f"{c:.1f}%" for c in confidences],
            textposition='outside',
            marker=dict(color=colors, line=dict(color='white', width=2)),
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={'text': "Object Detection Confidence Analysis", 'x': 0.5, 'xanchor': 'center',
               'font': {'size': 24, 'color': '#667eea', 'family': 'Poppins'}},
        xaxis_title="Detected Objects",
        yaxis_title="Confidence Score (%)",
        yaxis_range=[0, 110],
        height=500,
        template="plotly_white",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Poppins', size=12)
    )
    
    fig.update_xaxes(tickangle=45, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='lightgray', gridwidth=0.5)
    
    return fig


# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 YOLOv8 AI Agent Detection System</h1>
        <p>Advanced AI-Powered Object Recognition with Intelligent Feedback Loop</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize system
    if 'detector' not in st.session_state:
        st.session_state.detector = YOLOv8DetectionSystem()
    
    detector = st.session_state.detector
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>⚙️ Control Panel</h2></div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Detection Settings")
        confidence = st.slider("Confidence Threshold", 0.05, 1.0, 0.15, 0.05,
                              help="Lower values detect more objects (try 0.15 for books)")
        detector.confidence_threshold = confidence
        
        st.info("💡 **Tip for Books**: Try lowering confidence to 0.10-0.20")
        
        # Model selection
        model_options = {
            "YOLOv8 Nano (Fast)": "yolov8n.pt",
            "YOLOv8 Small (Balanced)": "yolov8s.pt", 
            "YOLOv8 Medium (Accurate)": "yolov8m.pt"
        }
        selected_model = st.selectbox("Model Selection", list(model_options.keys()), 
                                     help="Larger models detect better but are slower")
        
        if st.button("🔄 Load Selected Model"):
            with st.spinner("Loading model..."):
                try:
                    detector.model = YOLO(model_options[selected_model])
                    st.success(f"✅ Loaded {selected_model}!")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.markdown("### 🤖 AI Agent Features")
        enable_agent = st.checkbox("🧠 Enable AI Agent Analysis", value=True)
        enable_feedback = st.checkbox("🔄 Enable Feedback Loop", value=True)
        feedback_threshold = st.slider("Feedback Threshold", 0.1, 0.9, 0.5, 0.05,
                                      help="Detections below this confidence trigger feedback loop")
        
        st.markdown("### 🎙️ Output Settings")
        voice_enabled = st.checkbox("🔊 Enable Voice Output", value=True)
        
        st.markdown("### 📊 Display Options")
        show_boxes = st.checkbox("📦 Show Bounding Boxes", value=True)
        show_graph = st.checkbox("📈 Show Confidence Graph", value=True)
        
        st.info("💡 The AI agent analyzes detections and provides intelligent feedback for improvements")
    
    # Main Content
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Image Input")
        
        uploaded_file = st.file_uploader(
            "Upload Image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image for object detection"
        )
        
        camera_image = st.camera_input("📷 Or Capture Photo")
        
        image_to_process = None
        if uploaded_file:
            image_to_process = Image.open(uploaded_file)
        elif camera_image:
            image_to_process = Image.open(camera_image)
        
        if image_to_process:
            st.image(image_to_process, caption="🖼️ Original Image", use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Detection Results")
        
        if image_to_process:
            with st.spinner("🔍 AI is analyzing the image..."):
                detections, annotated = detector.detect_objects(image_to_process)
                
                if show_boxes and detections:
                    st.image(annotated, caption="✅ Detected Objects", use_container_width=True)
                
                if detections:
                    # Statistics Cards
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    
                    with stat_col1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <h3>{len(detections)}</h3>
                            <p>Objects Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stat_col2:
                        avg_conf = np.mean([d['confidence'] for d in detections]) * 100
                        st.markdown(f"""
                        <div class="stat-card">
                            <h3>{avg_conf:.1f}%</h3>
                            <p>Avg Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with stat_col3:
                        low_conf_count = len([d for d in detections if d['confidence'] < feedback_threshold])
                        st.markdown(f"""
                        <div class="stat-card">
                            <h3>{low_conf_count}</h3>
                            <p>Low Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Primary Detection
                    primary = detections[0]
                    st.markdown(f"""
                    <div class="result-card">
                        <h2 style="color: #667eea; margin: 0;">🎯 Primary Detection</h2>
                        <h1 style="margin: 0.5rem 0; color: #333;">{primary['object']}</h1>
                        <h3 style="color: #764ba2;">{primary['confidence']*100:.2f}% Confidence</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Voice Output
                    if voice_enabled:
                        speech = f"Detected {len(detections)} objects. Primary object is {primary['object']} with {primary['confidence']*100:.0f} percent confidence."
                        if st.button("🔊 Play Voice Summary"):
                            detector.speak(speech)
                            st.success("🎵 Playing audio...")
                else:
                    st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
        else:
            st.info("📤 Upload or capture an image to begin detection")
    
    # AI Agent Analysis Section
    if image_to_process and detections:
        st.markdown("---")
        
        # Agent Analysis
        if enable_agent:
            with st.spinner("🤖 AI Agent is analyzing detections..."):
                image_desc = f"Image contains {len(detections)} detected objects"
                analysis = detector.ai_agent.analyze_detections(detections, image_desc)
                display_agent_analysis(analysis, detections)
        
        # Feedback Loop for Low Confidence
        if enable_feedback:
            low_conf_detections = detector.get_low_confidence_detections(detections, feedback_threshold)
            
            if low_conf_detections:
                st.markdown("---")
                with st.spinner("🔄 Running feedback loop analysis..."):
                    feedback = detector.ai_agent.feedback_loop_analysis(
                        low_conf_detections,
                        f"Scene with {len(detections)} total detections"
                    )
                    display_feedback_loop(feedback, low_conf_detections)
            else:
                st.success(f"✅ All detections are above {feedback_threshold*100:.0f}% confidence threshold!")
        
        # Confidence Graph
        if show_graph:
            st.markdown("---")
            st.markdown("### 📊 Confidence Analysis")
            fig_bar = create_confidence_graph(detections)
            if fig_bar:
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # COMPREHENSIVE SUMMARY SECTION - NEW
        if enable_agent:
            display_summary_section(detections, analysis, feedback if enable_feedback and low_conf_detections else None)
        
        # Download Results
        st.markdown("---")
        st.markdown("### 💾 Export Results")
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            result_text = f"YOLOv8 AI Agent Detection Results\n{'='*50}\n\n"
            for i, det in enumerate(detections, 1):
                result_text += f"{i}. {det['object']}: {det['confidence']*100:.2f}%\n"
            
            if enable_agent and 'analysis' in locals():
                result_text += f"\n\nAI Agent Analysis:\n"
                result_text += f"{analysis.get('overall_assessment', '')}\n"
            
            st.download_button(
                "📄 Download Results (TXT)",
                result_text,
                file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    # Footer
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-top: 3rem;">
        <h3>🤖 YOLOv8 AI Agent Detection System</h3>
        <p style="opacity: 0.8; font-size: 0.9rem;">
            Advanced AI • Intelligent Feedback Loop • Real-time Analysis • Professional Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()