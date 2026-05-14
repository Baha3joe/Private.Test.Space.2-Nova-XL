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
# Kageillustrious风格核心关键词 - 使用Danbooru标签风格
STYLE_KEYWORDS = {
    "None": {
        "prefix": "",
        "suffix": ""
    },
    "Standard Quality": {
        "prefix": "(RAW photo:1.3), (photorealistic:1.4), (hyperrealistic:1.3), 8k uhd, (ultra realistic skin texture:1.2), cinematic lighting, vibrant colors,masterpiece, realistic skin texture, detailed anatomy, professional photography",
        "suffix": "sharp focus, (everything in focus:1.3), (no bokeh:1.2), realistic skin texture, subsurface scattering, detailed anatomy, (perfect anatomy:1.2),detailed face, detailed background, lifelike, professional photography, realistic proportions,  (detailed face:1.1), natural pose,expressive eyes, 8k resolution"
    },
    "High Detail": {
        "prefix": "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, colorful, rim light, backlit, highest detailed",
        "suffix": ""
    },
    "Realistic": {
        "prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, (photorealistic:1.3), (realistic:1.4), detailed skin texture, cinematic lighting",
        "suffix": "sharp focus, detailed anatomy, realistic proportions, detailed face, natural pose, expressive eyes, 8k resolution"
    },
    "Anime": {
        "prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, anime style, vibrant colors, detailed anime",
        "suffix": "cel shading, clean linework, vibrant anime colors, detailed anime eyes, smooth anime skin"
    },
    "Artistic": {
        "prefix": "masterpiece, best quality, amazing quality, very aesthetic, absurdres, artistic, illustration, detailed artwork",
        "suffix": "vibrant colors, expressive, detailed composition, artistic rendering"
    }
}

# 通用质量增强词
QUALITY_TAGS = "very awa, masterpiece, best quality, high resolution, highly detailed, professional"

# 修改为Kageillustrious模型 - 使用from_single_file加载
FIXED_MODEL_REPO = "PutiLeslie/kageillustrious_v60NLXLVersion"
FIXED_MODEL_FILE = "kageillustrious_v60NLXLVersion.safetensors"

# LoRA 配置 - 保留原有的LoRA(可能需要测试兼容性)
LORA_CONFIGS = [
    {
        "repo_id": "artificialguybr/LogoRedmond-LogoLoraForSDXL-V2",
        "weight_name": "LogoRedAF.safetensors",
        "adapter_name": "logo_lora",
        "scale": 0.8
    }
]

SAVE_DIR = "generated_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===== 模型相关变量 =====
pipeline = None
compel_processor = None
device = None
model_loaded = False

def initialize_model():
    """优化的模型初始化 - 使用from_single_file加载Kageillustrious"""
    global pipeline, compel_processor, device, model_loaded
    
    if model_loaded and pipeline is not None:
        print("✅ Model already loaded, skipping initialization")
        return True
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ Using device: {device}")
        
        print(f"📦 Loading Kageillustrious model from: {FIXED_MODEL_REPO}")
        
        # 使用from_single_file加载单个safetensors文件
        from huggingface_hub import hf_hub_download
        
        # 下载模型文件
        model_path = hf_hub_download(
            repo_id=FIXED_MODEL_REPO,
            filename=FIXED_MODEL_FILE
        )
        
        print(f"📥 Model downloaded to: {model_path}")
        
        # 使用from_single_file加载
        pipeline = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        
        # 优化调度器 - 使用Euler适合Illustrious系列
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config,
            timestep_spacing="trailing"
        )
        
        pipeline = pipeline.to(device)
        
        # 加载 LoRA
        print("🎨 Loading LoRA models...")
        adapter_names = []
        adapter_scales = []
        
        for lora_config in LORA_CONFIGS:
            try:
                pipeline.load_lora_weights(
                    lora_config["repo_id"],
                    weight_name=lora_config["weight_name"],
                    adapter_name=lora_config["adapter_name"]
                )
                adapter_names.append(lora_config["adapter_name"])
                adapter_scales.append(lora_config.get("scale", 0.8))
                print(f"✅ LoRA loaded: {lora_config['adapter_name']} (scale: {lora_config.get('scale', 0.8)})")
            except Exception as lora_error:
                print(f"⚠️ Failed to load LoRA {lora_config['adapter_name']}: {lora_error}")
        
        # 设置 LoRA 强度
        if adapter_names:
            try:
                pipeline.set_adapters(adapter_names, adapter_weights=adapter_scales)
                print(f"✅ LoRA adapters activated with scales: {adapter_scales}")
            except Exception as e:
                print(f"⚠️ Failed to set adapter scales: {e}")
        
        # GPU优化 - 适配ZeroGPU环境
        if torch.cuda.is_available():
            try:
                # VAE优化
                pipeline.enable_vae_slicing()
                pipeline.enable_vae_tiling()
                
                # 尝试启用xformers
                try:
                    pipeline.enable_xformers_memory_efficient_attention()
                    print("✅ xFormers enabled")
                except:
                    print("⚠️ xFormers not available, using default attention")
                
                # 不使用torch.compile，因为它在ZeroGPU环境中不稳定
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
        print("✅ Kageillustrious model initialization complete")
        return True
        
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        print(traceback.format_exc())
        model_loaded = False
        return False

def enhance_prompt(prompt: str, style: str) -> str:
    """优化的提示词增强 - 适配Kageillustrious的Danbooru标签风格"""
    if not prompt or prompt.strip() == "":
        return ""
    
    # 获取风格关键词
    style_config = STYLE_KEYWORDS.get(style, STYLE_KEYWORDS["None"])
    
    # 组合顺序:风格前缀 → 用户提示词 → 风格后缀 → 质量标签
    parts = []
    
    if style_config["prefix"]:
        parts.append(style_config["prefix"])
    
    parts.append(prompt.strip())
    
    if style_config["suffix"]:
        parts.append(style_config["suffix"])
    
    parts.append(QUALITY_TAGS)
    
    enhanced = ", ".join(parts)
    
    print(f"\n🎨 Style: {style}")
    print(f"📝 User prompt: {prompt[:100]}...")
    print(f"✨ Enhanced: {enhanced[:200]}...\n")
    
    return enhanced

def build_negative_prompt(style: str, custom_negative: str = "") -> str:
    """根据风格构建负面提示词 - 适配Illustrious系列"""
    # Illustrious系列推荐的负面提示词
    base_negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    
    # 风格特定的负面词
    style_negatives = {
        "Standard Quality": ", (cartoon:1.3), (anime:1.3), (3d render:1.2), (illustration:1.2), (painting:1.2), (drawing:1.2), (art:1.2), (sketch:1.2), artificial, unrealistic, (depth of field:1.2), (bokeh:1.2)",
        "Realistic": ", (cartoon:1.3), (anime:1.3), (3d render:1.2), (illustration:1.2)",
        "Anime": ", (realistic:1.3), (photorealistic:1.3), (photo:1.2)",
        "Artistic": ", (photo:1.2), (photorealistic:1.2)"
    }
    
    negative = base_negative
    if style in style_negatives:
        negative += style_negatives[style]
    
    # 添加用户自定义负面词
    if custom_negative.strip():
        negative += f", {custom_negative.strip()}"
    
    return negative

def process_with_compel(prompt, negative_prompt):
    """使用Compel处理长提示词"""
    if not compel_processor:
        return None, None
    
    try:
        # Compel会自动处理超过77 tokens的提示词
        conditioning, pooled = compel_processor([prompt, negative_prompt])
        print("✅ Long prompt processed with Compel")
        return conditioning, pooled
    except Exception as e:
        print(f"⚠️ Compel processing failed: {e}")
        return None, None

def apply_spaces_decorator(func):
    """应用spaces装饰器"""
    if SPACES_AVAILABLE:
        return spaces.GPU(duration=45)(func)
    return func

def create_metadata_content(prompt, enhanced_prompt, seed, steps, cfg_scale, width, height, style):
    """创建元数据"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 获取 LoRA 信息
    lora_info = ", ".join([f"{lora['adapter_name']}({lora.get('scale', 1.0)})" for lora in LORA_CONFIGS])
    
    return f"""Generated Image Metadata
======================
Timestamp: {timestamp}
Original Prompt: {prompt}
Seed: {seed}
Steps: {steps}
CFG Scale: {cfg_scale}
Dimensions: {width}x{height}
Style: {style}
"""

def cleanup_pipeline():
    """清理 pipeline 状态，防止污染"""
    global pipeline
    
    if pipeline is None:
        return
    
    try:
        # 清理 CUDA 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # 清理 pipeline 的内部缓存
        if hasattr(pipeline, 'unet'):
            # 清空 UNet 的注意力缓存
            if hasattr(pipeline.unet, 'set_attn_processor'):
                try:
                    from diffusers.models.attention_processor import AttnProcessor
                    pipeline.unet.set_attn_processor(AttnProcessor())
                except:
                    pass
        
        # 清理 VAE 缓存
        if hasattr(pipeline, 'vae'):
            pipeline.vae.to('cpu')
            pipeline.vae.to(device)
        
        print("🧹 Pipeline cleaned")
        
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

@apply_spaces_decorator
def generate_image(prompt: str, style: str, negative_prompt: str = "",
                   steps: int = 20, cfg_scale: float = 6.0,
                   seed: int = -1, width: int = 896, height: int = 1152,
                   progress=gr.Progress()):
    """图像生成主函数 - 使用Kageillustrious推荐参数"""
    
    # 验证输入
    if not prompt or prompt.strip() == "":
        return None, "", "❌ Please enter a prompt"
    
    progress(0.05, desc="Initializing...")
    
    # 初始化模型
    if not initialize_model():
        return None, "", "❌ Failed to load model"
    
    # 清理之前的状态
    cleanup_pipeline()
    
    progress(0.1, desc="Processing prompt...")
    
    try:
        # 处理seed
        if seed == -1:
            seed = random.randint(0, np.iinfo(np.int32).max)
        
        # 重要：为每次生成创建新的 generator，避免状态污染
        generator = torch.Generator(device).manual_seed(seed)
        
        # 增强提示词
        enhanced_prompt = enhance_prompt(prompt, style)
        
        # 构建负面提示词
        final_negative = build_negative_prompt(style, negative_prompt)
        
        print(f"🔧 Generation params: seed={seed}, steps={steps}, cfg={cfg_scale}, size={width}x{height}")
        print(f"📝 Prompt preview: {enhanced_prompt[:100]}...")
        
        progress(0.2, desc="Generating image...")
        
        # 检查提示词长度并决定是否使用Compel
        prompt_length = len(enhanced_prompt.split())
        use_compel = prompt_length > 50 and compel_processor is not None
        
        if use_compel:
            print(f"📏 Long prompt detected ({prompt_length} words), using Compel")
            conditioning, pooled = process_with_compel(enhanced_prompt, final_negative)
            
            if conditioning is not None:
                # 使用embeddings生成
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
                # Compel失败,回退到普通模式
                print("⚠️ Falling back to standard generation")
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
            # 标准生成
            print(f"📝 Standard generation ({prompt_length} words)")
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
        
        # 确保结果是PIL Image
        if not isinstance(result, Image.Image):
            if isinstance(result, np.ndarray):
                if result.dtype != np.uint8:
                    result = (result * 255).astype(np.uint8)
                result = Image.fromarray(result)
        
        # 创建元数据
        metadata = create_metadata_content(
            prompt, enhanced_prompt, seed, steps, cfg_scale,
            width, height, style
        )
        
        generation_info = f"Style: {style} | Seed: {seed} | Size: {width}×{height} | Steps: {steps} | CFG: {cfg_scale}"
        
        # 生成后立即清理
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        progress(1.0, desc="Complete!")
        print("✅ Generation successful\n")
        
        return result, generation_info, metadata
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Generation error: {error_msg}")
        print(traceback.format_exc())
        
        # 错误后也要清理
        try:
            cleanup_pipeline()
        except:
            pass
        
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

.model-info {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
    color: #764ba2 !important;
    padding: 10px !important;
    border-radius: 8px !important;
    margin-bottom: 15px !important;
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 2px solid rgba(118, 75, 162, 0.3) !important;
}

.prompt-box textarea, .prompt-box input {
    border-radius: 10px !important;
    border: 2px solid #667eea !important;
    padding: 15px !important;
    font-size: 18px !important;
    background: linear-gradient(135deg, rgba(245, 243, 255, 0.9), rgba(237, 233, 254, 0.9)) !important;
    color: #2d2d2d !important;
}

.prompt-box textarea:focus, .prompt-box input:focus {
    border-color: #764ba2 !important;
    box-shadow: 0 0 15px rgba(118, 75, 162, 0.3) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 249, 250, 0.95)) !important;
}

.controls-section {
    background: linear-gradient(135deg, rgba(224, 218, 255, 0.8), rgba(196, 181, 253, 0.8)) !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 8px !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    backdrop-filter: blur(5px) !important;
}

.controls-section label {
    font-weight: 600 !important;
    color: #2d2d2d !important;
    margin-bottom: 8px !important;
}

.controls-section input[type="radio"] {
    accent-color: #667eea !important;
}

.controls-section input[type="number"],
.controls-section input[type="range"] {
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid #667eea !important;
    border-radius: 6px !important;
    padding: 8px !important;
    color: #2d2d2d !important;
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
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}

.generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5) !important;
}

.image-output {
    border-radius: 15px !important;
    overflow: hidden !important;
    max-width: 100% !important;
    max-height: 70vh !important;
    border: 3px solid #764ba2 !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.15) !important;
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9), rgba(248, 249, 250, 0.9)) !important;
}

.image-info {
    background: linear-gradient(135deg, rgba(248, 249, 250, 0.9), rgba(233, 236, 239, 0.9)) !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin-top: 10px !important;
    font-size: 12px !important;
    color: #495057 !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
    backdrop-filter: blur(5px) !important;
}

.metadata-box {
    background: linear-gradient(135deg, rgba(248, 249, 250, 0.9), rgba(233, 236, 239, 0.9)) !important;
    border-radius: 8px !important;
    padding: 15px !important;
    margin-top: 15px !important;
    font-family: 'Courier New', monospace !important;
    font-size: 12px !important;
    color: #495057 !important;
    border: 2px solid rgba(102, 126, 234, 0.2) !important;
    backdrop-filter: blur(5px) !important;
    white-space: pre-wrap !important;
    overflow-y: auto !important;
    max-height: 300px !important;
}

@media (max-width: 768px) {
    .main-content {
        margin: 10px !important;
        padding: 15px !important;
    }
    .title {
        font-size: 1.5rem !important;
    }
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
                            value="Standard Quality"
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        seed_input = gr.Number(
                            label="Seed (-1 for random)",
                            value=-1,
                            precision=0
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        width_input = gr.Slider(
                            label="Width",
                            minimum=512,
                            maximum=2048,
                            value=896,
                            step=64,
                            info="Recommended: 896"
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        height_input = gr.Slider(
                            label="Height",
                            minimum=512,
                            maximum=2048,
                            value=1152,
                            step=64,
                            info="Recommended: 1152"
                        )
                    
                    with gr.Group(elem_classes=["controls-section"]):
                        steps_input = gr.Slider(
                            label="Steps",
                            minimum=10,
                            maximum=50,
                            value=20,
                            step=1,
                            info="Recommended: 20"
                        )
                        
                        cfg_input = gr.Slider(
                            label="CFG Scale",
                            minimum=1.0,
                            maximum=15.0,
                            value=6.0,
                            step=0.1,
                            info="Recommended: 6.0"
                        )
                    
                    generate_button = gr.Button(
                        "GENERATE",
                        elem_classes=["generate-btn"],
                        variant="primary"
                    )
            
            image_output = gr.Image(
                label="Generated Image",
                elem_classes=["image-output"],
                show_label=False,
                container=True
            )
            
            with gr.Row():
                generation_info = gr.Textbox(
                    label="Generation Info",
                    interactive=False,
                    elem_classes=["image-info"],
                    show_label=True,
                    visible=False
                )
            
            with gr.Row():
                metadata_display = gr.Textbox(
                    label="Image Metadata",
                    interactive=True,
                    elem_classes=["metadata-box"],
                    show_label=True,
                    lines=15,
                    visible=False
                )
            
            def on_generate(prompt, style, neg_prompt, steps, cfg, seed, width, height):
                image, info, metadata = generate_image(
                    prompt, style, neg_prompt, steps, cfg, seed, width, height
                )
                
                if image is not None:
                    return (
                        image,
                        info,
                        metadata,
                        gr.update(visible=True, value=info),
                        gr.update(visible=True, value=metadata)
                    )
                else:
                    return (
                        None,
                        info,
                        "",
                        gr.update(visible=False),
                        gr.update(visible=False)
                    )
            
            generate_button.click(
                fn=on_generate,
                inputs=[
                    prompt_input, style_input, negative_prompt_input,
                    steps_input, cfg_input, seed_input, width_input, height_input
                ],
                outputs=[
                    image_output, generation_info, metadata_display,
                    generation_info, metadata_display
                ],
                show_progress=True
            )
            
            prompt_input.submit(
                fn=on_generate,
                inputs=[
                    prompt_input, style_input, negative_prompt_input,
                    steps_input, cfg_input, seed_input, width_input, height_input
                ],
                outputs=[
                    image_output, generation_info, metadata_display,
                    generation_info, metadata_display
                ],
                show_progress=True
            )
        
        return interface

# ===== 启动应用 =====
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting ADULT AI Image Generator (YAOI Friendly) ")
    print("="*50)
    print(f"📦 Model: {FIXED_MODEL_REPO}")
    print(f"📄 Model File: {FIXED_MODEL_FILE}")
    print(f"🖥️ Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"⚡ ZeroGPU: {'Enabled' if SPACES_AVAILABLE else 'Disabled'}")
    print(f"📝 Compel: {'Available' if COMPEL_AVAILABLE else 'Not Available'}")
    if LORA_CONFIGS:
        print(f"🎨 LoRA: LogoRedmond-LogoLoraForSDXL-V2 (scale: {LORA_CONFIGS[0].get('scale', 0.8)})")
    print("="*50 + "\n")
    
    # 不预加载模型,让ZeroGPU按需分配
    # 这样可以避免GPU分配冲突
    
    app = create_interface()
    app.queue(max_size=10, default_concurrency_limit=2)
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )