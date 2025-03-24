import streamlit as st
import os
import time
import torch
import tempfile
from PIL import Image
from dotenv import load_dotenv
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
CACHE_DIR = os.getenv("CACHE_DIR", os.path.join(tempfile.gettempdir(), "smoldocling_cache"))

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

# Import for Transformers approach
try:
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from huggingface_hub import login
    transformers_available = True
except ImportError:
    transformers_available = False

try:
    from docling_core.types.doc import DoclingDocument
    from docling_core.types.doc.document import DocTagsDocument
    docling_available = True
except ImportError:
    docling_available = False

# Global variables for model caching
processor = None
model = None

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    if not transformers_available:
        missing.append("transformers huggingface_hub")
    if not docling_available:
        missing.append("docling-core")
    
    return missing

def get_available_devices():
    """Get available processing devices"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        for i in range(cuda_count):
            devices.append(f"cuda:{i} ({torch.cuda.get_device_name(i)})")
    return devices

def get_device_from_selection(selection):
    """Convert user-friendly device selection to torch device"""
    if selection.startswith("cuda:"):
        return selection.split(" ")[0]  # Extract just the "cuda:X" part
    return "cpu"

@st.cache_resource
def load_model(_device):
    """Load and cache the model to avoid reloading"""
    global processor, model
    
    # Authenticate with Hugging Face
    if HF_TOKEN:
        login(token=HF_TOKEN)
    
    try:
        logger.info(f"Loading SmolDocling model on {_device}...")
        processor = AutoProcessor.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            cache_dir=CACHE_DIR
        )
        model = AutoModelForVision2Seq.from_pretrained(
            "ds4sd/SmolDocling-256M-preview",
            torch_dtype=torch.float16 if _device.startswith("cuda") else torch.float32,
            cache_dir=CACHE_DIR
        ).to(_device)
        logger.info("Model loaded successfully")
        return processor, model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def optimize_image(image, max_size=1600):
    """Optimize image size while maintaining aspect ratio"""
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def process_single_image(image, prompt_text="Convert this page to docling.", device="cpu", show_progress=None):
    """Process a single image"""
    global processor, model
    
    # Optimize image
    image = optimize_image(image)
    
    start_time = time.time()
    
    # Load the model if not already loaded
    processor, model = load_model(device)
    
    # Create input messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        },
    ]
    
    # Prepare inputs
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    inputs = inputs.to(device)
    
    # Generate outputs
    with torch.no_grad():  # Add this to save memory
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=1500,  # Increased for better results
            do_sample=False,      # Deterministic generation
            num_beams=1,          # Simple beam search
            temperature=1.0,      # No temperature scaling
        )
    
    prompt_length = inputs.input_ids.shape[1]
    trimmed_generated_ids = generated_ids[:, prompt_length:]
    doctags = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=False,
    )[0].lstrip()
    
    # Clean the output
    doctags = doctags.replace("<end_of_utterance>", "").strip()
    
    # Populate document
    doctags_doc = DocTagsDocument.from_doctags_and_image_pairs([doctags], [image])
    
    # Create a docling document
    doc = DoclingDocument(name="Document")
    doc.load_from_doctags(doctags_doc)
    
    # Export as markdown
    md_content = doc.export_to_markdown()
    
    # Export as HTML
    html_content = doc.export_to_html()
    
    # Get plain text
    plain_text = doc.export_to_text()
    
    processing_time = time.time() - start_time
    
    return {
        "doctags": doctags,
        "markdown": md_content,
        "html": html_content,
        "text": plain_text,
        "processing_time": processing_time
    }

def process_batch(images, prompt_text, device, progress_bar=None):
    """Process a batch of images with progress tracking"""
    results = []
    total = len(images)
    
    for idx, image in enumerate(images):
        if progress_bar:
            progress_bar.progress((idx) / total, text=f"Processing image {idx+1}/{total}")
        
        result = process_single_image(image, prompt_text, device)
        results.append(result)
        
        if progress_bar:
            progress_bar.progress((idx + 1) / total, text=f"Processed {idx+1}/{total} images")
    
    return results

def save_session_history(results):
    """Save processing results to session history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for idx, result in enumerate(results):
        st.session_state.history.append({
            "id": len(st.session_state.history) + 1,
            "timestamp": timestamp,
            "type": "Image " + str(idx + 1),
            "processing_time": result["processing_time"],
            "result": result
        })

def display_history():
    """Display session history"""
    if 'history' not in st.session_state or not st.session_state.history:
        st.info("No processing history available")
        return
    
    st.subheader("Processing History")
    
    for item in reversed(st.session_state.history):
        with st.expander(f"#{item['id']} - {item['type']} ({item['timestamp']})"):
            st.write(f"Processing time: {item['processing_time']:.2f} seconds")
            tabs = st.tabs(["Markdown", "Text", "DocTags", "HTML"])
            
            with tabs[0]:
                st.markdown(item['result']['markdown'])
                st.download_button(
                    "Download Markdown", 
                    item['result']['markdown'], 
                    file_name=f"output_{item['id']}.md"
                )
            
            with tabs[1]:
                st.text_area("Plain Text", item['result']['text'], height=200)
                st.download_button(
                    "Download Text", 
                    item['result']['text'], 
                    file_name=f"output_{item['id']}.txt"
                )
            
            with tabs[2]:
                st.text_area("DocTags", item['result']['doctags'], height=200)
                st.download_button(
                    "Download DocTags", 
                    item['result']['doctags'], 
                    file_name=f"output_{item['id']}.dt"
                )
            
            with tabs[3]:
                st.code(item['result']['html'], language="html")
                st.download_button(
                    "Download HTML", 
                    item['result']['html'], 
                    file_name=f"output_{item['id']}.html"
                )

def main():
    # App configuration
    st.set_page_config(
        page_title="SmolDocling OCR App",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom theme
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        background-color: #f0f2f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f0ff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App header
    st.markdown('<p class="main-header">SmolDocling OCR App</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Extract text from images using SmolDocling AI</p>', unsafe_allow_html=True)
    
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}. Please install them to use this app.")
        st.info("Install with: pip install " + " ".join(missing_deps))
        st.stop()
    
    # Initialize session state
    if 'results' not in st.session_state:
        st.session_state.results = []
    
    # Create sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Device selection
        st.subheader("Processing Device")
        available_devices = get_available_devices()
        selected_device = st.selectbox(
            "Select processing device",
            available_devices,
            index=0 if len(available_devices) == 1 else 1,  # Default to CUDA if available
            help="Choose the device for model inference. GPU (CUDA) is recommended for faster processing."
        )
        device = get_device_from_selection(selected_device)
        
        # Model info
        st.info(f"Selected device: {selected_device}")
        
        if device == "cpu":
            st.warning("⚠️ CPU processing may be slow. Select a GPU device if available for faster performance.")
        
        # Memory management
        if device.startswith("cuda"):
            with st.expander("GPU Memory Management"):
                st.write("Current GPU Memory Usage:")
                if torch.cuda.is_available():
                    gpu_idx = int(device.split(":")[1]) if ":" in device else 0
                    allocated = torch.cuda.memory_allocated(gpu_idx) / (1024 ** 3)
                    reserved = torch.cuda.memory_reserved(gpu_idx) / (1024 ** 3)
                    st.progress(allocated / (torch.cuda.get_device_properties(gpu_idx).total_memory / (1024 ** 3)))
                    st.write(f"Allocated: {allocated:.2f} GB")
                    st.write(f"Reserved: {reserved:.2f} GB")
                    
                    if st.button("Clear GPU Cache"):
                        torch.cuda.empty_cache()
                        st.success("GPU cache cleared")
        
        # Upload options
        st.subheader("Upload Options")
        upload_option = st.radio("Choose upload option:", ["Single Image", "Multiple Images"])
        
        # Advanced options
        with st.expander("Advanced Options"):
            task_type = st.selectbox(
                "Select task type",
                [
                    "Convert this page to docling.",
                    "Convert this table to OTSL.",
                    "Convert code to text.",
                    "Convert formula to latex.",
                    "Convert chart to OTSL.",
                    "Extract all section header elements on the page."
                ]
            )
            
            custom_prompt = st.text_area(
                "Custom prompt (optional)", 
                value="", 
                help="Provide a custom prompt if needed. Leave empty to use the selected task type."
            )
            
            max_image_size = st.slider(
                "Max image dimension (pixels)", 
                min_value=800, 
                max_value=3200, 
                value=1600,
                step=100,
                help="Larger values may improve OCR quality but use more memory"
            )
            
            final_prompt = custom_prompt if custom_prompt else task_type
        
        # Upload controls
        st.subheader("Upload Image(s)")
        if upload_option == "Single Image":
            uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "pdf"])
            
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    st.image(image, caption="Uploaded Image", width=250)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
        else:
            uploaded_files = st.file_uploader(
                "Upload multiple images", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.success(f"{len(uploaded_files)} images uploaded")
        
        # Process button
        if (upload_option == "Single Image" and 'uploaded_file' in locals() and uploaded_file is not None) or \
           (upload_option == "Multiple Images" and 'uploaded_files' in locals() and uploaded_files):
            process_button = st.button("Process Image(s)", type="primary")
        
        # History button
        st.subheader("History")
        if st.button("Show Processing History"):
            st.session_state.show_history = True
        
        # About section
        with st.expander("About SmolDocling OCR"):
            st.write("""
            This app uses SmolDocling, a powerful OCR model for document understanding from Hugging Face Hub.
            
            The app extracts DocTags format and converts it to Markdown, HTML, and plain text for easy reading.
            
            Available tasks:
            - Convert pages to DocTags (general OCR)
            - Convert tables to OTSL
            - Convert code snippets to text
            - Convert formulas to LaTeX
            - Convert charts to OTSL
            - Extract section headers
            """)
    
    # Main content area
    if 'show_history' in st.session_state and st.session_state.show_history:
        display_history()
        st.session_state.show_history = False
    elif upload_option == "Single Image" and 'uploaded_file' in locals() and uploaded_file is not None and process_button:
        with st.spinner("Processing image..."):
            try:
                progress_bar = st.progress(0, text="Preparing to process...")
                
                # Update global optimization settings
                optimize_image.func_defaults = (max_image_size,)
                
                result = process_single_image(image, final_prompt, device)
                st.session_state.results = [result]
                
                # Save to history
                save_session_history(st.session_state.results)
                
                progress_bar.progress(1.0, text="Processing complete!")
                
                # Display results
                tabs = st.tabs(["Markdown", "Text", "DocTags", "HTML"])
                
                with tabs[0]:
                    st.subheader("Markdown Output")
                    st.markdown(result["markdown"])
                    st.download_button(
                        "Download Markdown", 
                        result["markdown"], 
                        file_name="output.md"
                    )
                
                with tabs[1]:
                    st.subheader("Plain Text Output")
                    st.text_area("Extracted Text", result["text"], height=300)
                    st.download_button(
                        "Download Text", 
                        result["text"], 
                        file_name="output.txt"
                    )
                
                with tabs[2]:
                    st.subheader("DocTags Output")
                    st.text_area("DocTags", result["doctags"], height=300)
                    st.download_button(
                        "Download DocTags", 
                        result["doctags"], 
                        file_name="output.dt"
                    )
                
                with tabs[3]:
                    st.subheader("HTML Output")
                    st.code(result["html"], language="html")
                    st.download_button(
                        "Download HTML", 
                        result["html"], 
                        file_name="output.html"
                    )
                
                st.success(f"Processing completed in {result['processing_time']:.2f} seconds on {selected_device}")
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                logger.error(f"Error processing image: {str(e)}", exc_info=True)
    
    elif upload_option == "Multiple Images" and 'uploaded_files' in locals() and uploaded_files and process_button:
        try:
            images = [Image.open(file).convert("RGB") for file in uploaded_files]
            
            if len(images) > 0:
                with st.spinner(f"Processing {len(images)} images..."):
                    progress_bar = st.progress(0, text="Preparing to process...")
                    
                    # Update global optimization settings
                    optimize_image.func_defaults = (max_image_size,)
                    
                    results = process_batch(images, final_prompt, device, progress_bar)
                    st.session_state.results = results
                    
                    # Save to history
                    save_session_history(results)
                    
                    progress_bar.progress(1.0, text="Processing complete!")
                    
                    # Display results
                    st.subheader("Processing Results")
                    
                    total_time = sum(result["processing_time"] for result in results)
                    avg_time = total_time / len(results)
                    
                    st.write(f"Total processing time: {total_time:.2f} seconds on {selected_device}")
                    st.write(f"Average processing time: {avg_time:.2f} seconds per image")
                    
                    # Create tabs for each image
                    for idx, (result, image) in enumerate(zip(results, images)):
                        with st.expander(f"Image {idx+1} Results"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.image(image, caption=f"Image {idx+1}", width=250)
                                st.write(f"Processing time: {result['processing_time']:.2f} seconds")
                            
                            with col2:
                                inner_tabs = st.tabs(["Markdown", "Text", "DocTags", "HTML"])
                                
                                with inner_tabs[0]:
                                    st.markdown(result["markdown"])
                                    st.download_button(
                                        f"Download Markdown", 
                                        result["markdown"], 
                                        file_name=f"output_{idx+1}.md"
                                    )
                                
                                with inner_tabs[1]:
                                    st.text_area("Plain Text", result["text"], height=200)
                                    st.download_button(
                                        f"Download Text", 
                                        result["text"], 
                                        file_name=f"output_{idx+1}.txt"
                                    )
                                
                                with inner_tabs[2]:
                                    st.text_area("DocTags", result["doctags"], height=200)
                                    st.download_button(
                                        f"Download DocTags", 
                                        result["doctags"], 
                                        file_name=f"output_{idx+1}.dt"
                                    )
                                
                                with inner_tabs[3]:
                                    st.code(result["html"], language="html")
                                    st.download_button(
                                        f"Download HTML", 
                                        result["html"], 
                                        file_name=f"output_{idx+1}.html"
                                    )
                    
                    st.success(f"All images processed successfully")
        except Exception as e:
            st.error(f"Error processing images: {str(e)}")
            logger.error(f"Error processing images: {str(e)}", exc_info=True)
    
    # Display a welcome message if no image has been uploaded
    if ('uploaded_file' not in locals() or uploaded_file is None) and \
       ('uploaded_files' not in locals() or not uploaded_files):
        st.info("👈 Upload an image using the sidebar to get started")


if __name__ == "__main__":
    main()