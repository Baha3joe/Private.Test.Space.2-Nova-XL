# ===== 必须首先导入spaces =====
try:
    import spaces
    SPACES_AVAILABLE = True
    print("✅ Spaces available - ZeroGPU mode")
except ImportError:
    SPACES_AVAILABLE = False
    print("⚠️ Spaces not available - running in regular mode")

# ===== 其他导入 =====
import os
import uuid
from datetime import datetime
import random
import torch
import gradio as gr
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from PIL import Image
import traceback
import numpy as np

# ===== 长提示词处理 =====
try:
    from compel import Compel, ReturnedEmbeddingsType
    COMPEL_AVAILABLE = True
    print("✅ Compel available for long prompt processing")
except ImportError:
    COMPEL_AVAILABLE = False
    print("⚠️ Compel not available - using standard prompt processing")

# ===== 优化后的配置 =====
# Nova XL Anime风格核心关键词 - 使用Danbooru标签风格
STYLE_KEYWORDS = {
    "None": {
        "prefix": "",
        "suffix": ""
    },
    "Anime": {
        "prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, anime style, vibrant colors, detailed anime, high resolution, detailed face, beautiful eyes",
        "suffix": "cel shading, clean linework, vibrant anime colors, detailed anime eyes, smooth anime skin, professional illustration"
    },
    "Artistic": {
        "prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, artistic, illustration, detailed artwork, professional illustration",
        "suffix": "vibrant colors, expressive, detailed composition, artistic rendering, beautiful lighting"
    },
    "High Detail": {
        "prefix": "masterpiece, best quality, high resolution, ultra-detailed, absurdres, newest, colorful, detailed, professional quality",
        "suffix": "sharp focus, detailed background, professional illustration, high quality art"
    }
}

# 通用质量增强词
QUALITY_TAGS = "very awa, masterpiece, best quality, high resolution, highly detailed, professional"

# Nova XL Anime模型 - 使用from_single_file加载
NOVA_MODEL_URL = "https://huggingface.co/AriaVale2/nova-anime-xl-weights/resolve/main/novaAnimeXL_ilV190.safetensors"

SAVE_DIR = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 模型相关变量 =====
pipeline = None
compel_processor = None
device = None
model_loaded = False

def initialize_model():
    """优化的模型初始化 - 使用from_single_file加载Nova XL"""
    global pipeline, compel_processor, device, model_loaded
    
    if model_loaded and pipeline is not None:
        print("✅ Model already loaded, skipping initialization")
        return True
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        
        print(f"📦 Loading Nova XL Anime model from custom repository...")
        
        # 使用from_single_file直接加载HuggingFace模型URL
        pipeline = StableDiffusionXLPipeline.from_single_file(
            NOVA_MODEL_URL,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        print(f"📥 Nova XL model loaded successfully")
        
        # 优化调度器 - 使用Euler适合Nova XL
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        
        pipeline = pipeline.to(device)
        
        # GPU优化 - 适配ZeroGPU环境
        if torch.cuda.is_available():
            try:
                pipeline.enable_vae_slicing()
                pipeline.enable_vae_tiling()
                
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers enabled")
                except:
                    print("⚠️ xFormers not available, using default attention")
                
                print("ℹ️ Skipping torch.compile for ZeroGPU compatibility")
                
            except Exception as opt_error:
                print(f"⚠️ Optimization warning: {opt_error}")
        
        # 初始化Compel用于长提示词
        if COMPEL_AVAILABLE:
            try:
                compel_processor = Compel(
                    tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
                    text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    truncate_long_prompts=False
                )
                print("✅ Compel processor initialized")
            except Exception as compel_error:
                print(f"⚠️ Compel initialization failed: {compel_error}")
                compel_processor = None
        
        model_loaded = True
        print("✅ Nova XL model initialization complete")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        print(traceback.format_exc())
        model_loaded = False
        return False

def enhance_prompt(prompt: str, style: str) -> str:
    """优化的提示词增强 - 适配Nova XL的Danbooru标签风格"""
    if not prompt or prompt.strip() == "":
        return ""
    
    style_config = STYLE_KEYWORDS.get(style, STYLE_KEYWORDS["None"])
    parts = []
    
    if style_config["prefix"]:
        parts.append(style_config["prefix"])
    
    parts.append(prompt.strip())
    
    if style_config["suffix"]:
        parts.append(style_config["suffix"])
    
    parts.append(QUALITY_TAGS)
    
    enhanced = ", ".join(parts)
    return enhanced

def build_negative_prompt(style: str, custom_negative: str = "") -> str:
    """根据风格构建负面提示词 - 适配Nova XL"""
    base_negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    style_negatives = {
        "Anime": ", (realistic:1.2), (photorealistic:1.2), (photo:1.1), (3d:1.1)",
        "Artistic": ", (photo:1.1), (photorealistic:1.1)",
        "High Detail": ""
    }
    
    negative = base_negative
    if style in style_negatives:
        negative += style_negatives[style]
    
    if custom_negative.strip():
        negative += f", {custom_negative.strip()}"
    
    return negative

def process_with_compel(prompt, negative_prompt):
    if not compel_processor:
        return None, None
    try:
        conditioning, pooled = compel_processor([prompt, negative_prompt])
        return conditioning, pooled
    except Exception as e:
        print(f"⚠️ Compel processing failed: {e}")
        return None, None

def apply_spaces_decorator(func):
    if SPACES_AVAILABLE:
        return spaces.GPU(duration=45)(func)
    return func

def create_metadata_content(prompt, enhanced_prompt, seed, steps, cfg_scale, width, height, style):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""Generated Image Metadata
======================
Timestamp: {timestamp}
Original Prompt: {prompt}
Seed: {seed}
Steps: {steps}
CFG Scale: {cfg_scale}
Dimensions: {width}x{height}
Style: {style}
Model: Nova XL Anime
"""

def cleanup_pipeline():
    global pipeline
    if pipeline is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

@apply_spaces_decorator
def generate_image(prompt: str, style: str, negative_prompt: str = "",
                   steps: int = 20, cfg_scale: float = 6.0,
                   seed: int = -1, width: int = 896, height: int = 1152,
                   progress=gr.Progress()):
    
    if not prompt or prompt.strip() == "":
        return None, "", "❌ Please enter a prompt"
    
    progress(0.05, desc="Initializing...")
    if not initialize_model():
        return None, "", "❌ Failed to load model"
    
    cleanup_pipeline()
    progress(0.1, desc="Processing prompt...")
    
    try:
        if seed == -1:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        generator = torch.Generator(device).manual_seed(seed)
        enhanced_prompt = enhance_prompt(prompt, style)
        final_negative = build_negative_prompt(style, negative_prompt)
        
        progress(0.2, desc="Generating image...")
        
        prompt_length = len(enhanced_prompt.split())
        use_compel = prompt_length > 50 and compel_processor is not None
        
        if use_compel:
            conditioning, pooled = process_with_compel(enhanced_prompt, final_negative)
            if conditioning is not None:
                result = pipeline(
                    prompt_embeds=conditioning[0:1],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="pil"
                ).images[0]
            else:
                result = pipeline(
                    prompt=enhanced_prompt,
                    negative_prompt=final_negative,
                    num_inference_steps=steps,
                    guidance_scale=cfg_scale,
                    width=width,
                    height=height,
                    generator=generator,
                    output_type="pil"
                ).images[0]
        else:
            result = pipeline(
                prompt=enhanced_prompt,
                negative_prompt=final_negative,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator,
                output_type="pil"
            ).images[0]
        
        progress(0.95, desc="Finalizing...")
        
        if not isinstance(result, Image.Image):
            if isinstance(result, np.ndarray):
                if result.dtype != np.uint8:
                    result = (result * 255).astype(np.uint8)
                result = Image.fromarray(result)
        
        metadata = create_metadata_content(
            prompt, enhanced_prompt, seed, steps, cfg_scale, width, height, style
        )
        generation_info = f"Style: {style} | Seed: {seed} | Size: {width}×{height} | Steps: {steps} | CFG: {cfg_scale}"
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress(1.0, desc="Complete!")
        return result, generation_info, metadata
        
    except Exception as e:
        error_msg = str(e)
        print(traceback.format_exc())
        return None, "", f"❌ Generation failed: {error_msg}"

# ===== CSS样式 =====
css = """
.gradio-container {
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    min-height: 100vh !important;
    font-family: 'Segoe UI', Arial, sans-serif !important;
}
.main-content {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin: 15px !important;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2) !important;
    min-height: calc(100vh - 30px) !important;
    color: #3e3e3e !important;
    backdrop-filter: blur(10px) !important;
}
.title {
    text-align: center !important;
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-size: 2rem !important;
    margin-bottom: 15px !important;
    font-weight: bold !important;
}
.warning-box {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    color: white !important;
    padding: 8px !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    text-align: center !important;
    font-weight: bold !important;
    font-size: 14px !important;
}
.prompt-box textarea, .prompt-box input {
    border-radius: 10px !important;
    border: 2px solid #667eea !important;
    padding: 15px !important;
    font-size: 18px !important;
    background: linear-gradient(135deg, rgba(245, 243, 255, 0.9), rgba(237, 233, 254, 0.9)) !important;
    color: #2d2d2d !important;
}
.controls-section {
    background: linear-gradient(135deg, rgba(224, 218, 255, 0.8), rgba(196, 181, 253, 0.8)) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 8px !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
}
.generate-btn {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    padding: 15px 25px !important;
    border-radius: 25px !important;
    font-size: 16px !important;
    font-weight: bold !important;
    width: 100% !important;
    cursor: pointer !important;
}
.image-output {
    border-radius: 15px !important;
    overflow: hidden !important;
    border: 3px solid #764ba2 !important;
}
"""

# ===== 创建UI =====
def create_interface():
    with gr.Blocks(css=css, title="ADULT AI Image Generator") as interface:
        with gr.Column(elem_classes=["main-content"]):
            gr.HTML('<div class="title">🎨 ADULT AI Image Generator</div>')
            gr.HTML('<div class="warning-box">⚠️ 18+ CONTENT WARNING ⚠️</div>')
            
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(
                        label="Detailed Prompt (Use Danbooru tags style)",
                        placeholder="1boy, solo, messy hair, blue eyes, detailed face, handsome...",
                        lines=15,
                        elem_classes=["prompt-box"]
                    )
                    
                    negative_prompt_input = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        placeholder="Additional things you don't want...",
                        lines=4,
                        elem_classes=["prompt-box"]
                    )
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["controls-section"]):
                        style_input = gr.Radio(
                            label="Style Preset",
                            choices=list(STYLE_KEYWORDS.keys()),
                            value="None"  # FIX: Set to a valid key from dictionary
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        seed_input = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        width_input = gr.Slider(label="Width", minimum=512, maximum=2048, value=896, step=64)
                        height_input = gr.Slider(label="Height", minimum=512, maximum=2048, value=1152, step=64)
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        steps_input = gr.Slider(label="Steps", minimum=10, maximum=50, value=20, step=1)
                        cfg_input = gr.Slider(label="CFG Scale", minimum=1.0, maximum=15.0, value=6.0, step=0.1)
                    
                    generate_button = gr.Button("GENERATE", elem_classes=["generate-btn"], variant="primary")
            
            image_output = gr.Image(label="Generated Image", elem_classes=["image-output"], show_label=False)
            
            with gr.Row():
                generation_info = gr.Textbox(label="Generation Info", interactive=False, visible=False)
            
            with gr.Row():
                metadata_display = gr.Textbox(label="Image Metadata", interactive=True, lines=15, visible=False)
            
            def on_generate(prompt, style, neg_prompt, steps, cfg, seed, width, height):
                image, info, metadata = generate_image(
                    prompt, style, neg_prompt, steps, cfg, seed, width, height
                )
                if image is not None:
                    return image, info, metadata, gr.update(visible=True, value=info), gr.update(visible=True, value=metadata)
                return None, info, "", gr.update(visible=False), gr.update(visible=False)
            
            generate_button.click(
                fn=on_generate,
                inputs=[prompt_input, style_input, negative_prompt_input, steps_input, cfg_input, seed_input, width_input, height_input],
                outputs=[image_output, generation_info, metadata_display, generation_info, metadata_display]
            )
        return interface

if __name__ == "__main__":
    app = create_interface()
    app.queue(max_size=10, default_concurrency_limit=2)
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)