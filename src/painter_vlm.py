import sys
import cv2
import datetime
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import json
import base64
import os

from paint_utils3 import canvas_to_global_coordinates, get_colors, nearest_color, save_colors, show_img
from painting_optimization import log_progress
from painter import Painter
from options import Options
from my_tensorboard import TensorBoard
from brush_stroke import BrushStroke

from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VLMPainter(Painter):
    # Class-level default parameters
    DEFAULT_VLM_NAME = 'Qwen/Qwen2.5-VL-7B-Instruct'
    DEFAULT_GPU_MEMORY_UTIL = 0.85
    DEFAULT_MAX_MODEL_LEN = 8192
    DEFAULT_MAX_NUM_SEQS = 1

    def __init__(self, opt, 
                 vlm_name=None,
                 gpu_memory_utilization=None,
                 max_model_len=None,
                 max_num_seqs=None):
        super().__init__(opt)
        
        # Use provided values or defaults
        self.vlm_name = vlm_name or self.DEFAULT_VLM_NAME
        self.gpu_memory_utilization = gpu_memory_utilization or self.DEFAULT_GPU_MEMORY_UTIL
        self.max_model_len = max_model_len or self.DEFAULT_MAX_MODEL_LEN
        self.max_num_seqs = max_num_seqs or self.DEFAULT_MAX_NUM_SEQS

        # Initialize VLM
        self.llm = LLM(
            model=self.vlm_name,
            limit_mm_per_prompt={"image": 10, "video": 10},
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.vlm_name, trust_remote_code=True)

        # Load target image
        if opt.objective_data.startswith('http'):
            response = requests.get(opt.objective_data)
            target_image = Image.open(BytesIO(response.content))
            target_image = np.array(target_image)
        else:
            target_image = cv2.imread(opt.objective_data)[:,:,::-1]  # BGR to RGB
            
        opt.w_render = 256
        opt.h_render = 256
        self.target_image = cv2.resize(target_image, (opt.w_render, opt.h_render))
        self.target_image = self.target_image.astype(np.float32) / 255.0
        
        # Initialize color palette with more vibrant colors for better visibility
        base_colors = np.array([
            [1.0, 0.0, 0.0],  # Red
            [0.0, 0.0, 1.0],  # Blue
            [0.0, 0.8, 0.0],  # Green
            [0.8, 0.0, 0.8],  # Purple
            [1.0, 0.6, 0.0],  # Orange
        ])
        self.color_palette = torch.from_numpy(base_colors).float().to(device)
        
        # Create color palette visualization
        color_viz = np.ones((256, 256*len(base_colors), 3))
        for i, color in enumerate(base_colors):
            color_viz[:, i*256:(i+1)*256] = color
        self.opt.writer.add_image('paint_colors/palette', 
                                torch.from_numpy(color_viz), 
                                0)

    def _convert_to_base64(self, image):
        """Convert numpy array to base64 string"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _parse_color(self, color_value):
        """Parse color from various formats (RGB array, hex code, or color name) to RGB array"""
        if isinstance(color_value, list):
            # RGB array format [r, g, b]
            return np.array(color_value)
        elif isinstance(color_value, str):
            if color_value.startswith('#'):
                # Hex code format '#RRGGBB'
                color_value = color_value.lstrip('#')
                return np.array([int(color_value[i:i+2], 16) for i in (0, 2, 4)])
            else:
                # Color name format (e.g., 'lightgray')
                try:
                    import matplotlib.colors as mcolors
                    rgb = mcolors.to_rgb(color_value)
                    return np.array([int(c * 255) for c in rgb])
                except ValueError:
                    print(f"Warning: Unknown color name '{color_value}', using red instead")
                    return np.array([255, 0, 0])
        else:
            print("Warning: Invalid color format, using red instead")
            return np.array([255, 0, 0])

    def _map_to_palette_color(self, color_rgb):
        """Map RGB color to nearest color in the palette"""
        if not hasattr(self, 'color_palette'):
            # Initialize color palette if not exists
            base_colors = np.array([
                [1.0, 0.0, 0.0],  # Red
                [0.0, 0.0, 1.0],  # Blue
                [0.0, 0.8, 0.0],  # Green
                [0.8, 0.0, 0.8],  # Purple
                [1.0, 0.6, 0.0],  # Orange
            ])
            self.color_palette = torch.from_numpy(base_colors).float().to(device)

        # Convert RGB 0-255 to 0-1
        color_rgb = color_rgb / 255.0
        
        # Find nearest color in palette
        color_tensor = torch.from_numpy(color_rgb).float().to(device)
        distances = torch.norm(self.color_palette - color_tensor, dim=1)
        nearest_idx = torch.argmin(distances)
        return self.color_palette[nearest_idx]

    def _clean_json_string(self, text):
        """Clean the JSON string from markdown code blocks"""
        if text.startswith('```json'):
            # Remove ```json and ``` markers
            text = text.replace('```json\n', '').replace('\n```', '')
        elif text.startswith('```'):
            # Remove ``` markers
            text = text.replace('```\n', '').replace('\n```', '')
        return text.strip()

    def generate(self, images, prompt: str, temperature=0.1):
        if len(images) == 1:
            images = [images[0]]
        else:
            images = [images[0], images[1]]

        # Save base64 images to temporary files
        temp_image_paths = []
        for idx, img_base64 in enumerate(images):
            if isinstance(img_base64, str) and img_base64.startswith('/9j/'):  # base64 JPEG
                img_data = base64.b64decode(img_base64)
                temp_path = f"/tmp/temp_image_{idx}.jpg"
                with open(temp_path, 'wb') as f:
                    f.write(img_data)
                temp_image_paths.append(temp_path)
            else:
                temp_image_paths.append(img_base64)  # URL or file path

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                *[{
                    "type": "image",
                    "image": img_path,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                } for img_path in temp_image_paths]
            ]},
        ]

        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=2048,
            stop_token_ids=[],
        )

        outputs = self.llm.generate([llm_inputs], sampling_params=sampling_params)
        
        # Cleanup temporary files
        for path in temp_image_paths:
            if path.startswith('/tmp/temp_image_'):
                try:
                    os.remove(path)
                except:
                    pass
                    
        return outputs[0].outputs[0].text

    def get_clip_features(self, image):
        """Get CLIP features for an image"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray((image * 255).astype(np.uint8))
        inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
        image_features = self.clip_model.get_image_features(**inputs)
        return image_features.detach()

    def analyze_difference(self, current_canvas):
        """Analyze the difference between target and current canvas using CLIP"""
        target_features = self.get_clip_features(self.target_image)
        current_features = self.get_clip_features(current_canvas)
        
        # Compute cosine similarity
        similarity = torch.nn.functional.cosine_similarity(target_features, current_features)
        
        # Get difference regions
        diff = np.abs(self.target_image - current_canvas)
        diff_regions = cv2.dilate(diff, np.ones((5,5), np.uint8))
        
        return similarity.item(), diff_regions

    def plan_next_stroke(self, current_canvas, diff_regions):
        """Plan the next stroke based on the difference analysis"""
        # Create a random stroke with parameters biased towards high difference regions
        stroke = BrushStroke(self.opt)
        
        # Find region of maximum difference
        weighted_diff = diff_regions.mean(axis=2)
        y, x = np.unravel_index(weighted_diff.argmax(), weighted_diff.shape)
        
        # Convert to canvas coordinates
        x_prop = x / self.opt.w_render
        y_prop = 1 - (y / self.opt.h_render)
        
        return stroke, x_prop, y_prop

    def plan(self):
        """Initial planning phase to break down the painting into regions and strategies"""
        # Convert target image to base64 for VLM input
        self.target_img_base64 = self._convert_to_base64(self.target_image)
        
        # Calculate grid size
        grid_size = self.opt.squared_paper_num
        grid_h = self.opt.h_render / grid_size
        grid_w = self.opt.w_render / grid_size
        
        # Initial global planning prompt
        global_plan_prompt = f"""
        You are a professional artist planning a painting. The image is divided into a {grid_size}x{grid_size} grid for precise reference.
        Each grid cell is {grid_h:.1f}x{grid_w:.1f} pixels.
        
        Look at this image and:
        1. Identify 3-5 main regions or objects to paint
        2. For each region, specify:
           - Grid coordinates (start_x, start_y, end_x, end_y) where x and y are grid numbers from 0 to {grid_size-1}
           - What object/element it contains
           - Suggested painting order
           - Approximate number of brush strokes needed
        3. Describe the overall color palette and painting strategy
        
        Format your response as a structured JSON with:
        {{
            "regions": [
                {{
                    "id": 1,
                    "grid": [start_x, start_y, end_x, end_y],
                    "content": "description",
                    "paint_order": 1,
                    "estimated_strokes": 50
                }},
                ...
            ],
            "total_strokes": 200,
            "strategy": "overall painting strategy"
        }}
        """
        
        initial_plan = self.generate([self.target_img_base64], global_plan_prompt, temperature=0.1)
        cleaned_json = self._clean_json_string(initial_plan)
        plan_data = json.loads(cleaned_json)
        
        # Convert grid coordinates to pixel coordinates
        for region in plan_data["regions"]:
            start_x, start_y, end_x, end_y = region["grid"]
            region["bbox"] = [
                int(start_x * grid_w),
                int(start_y * grid_h),
                int(end_x * grid_w),
                int(end_y * grid_h)
            ]
        
        return plan_data

    def paint(self, initial_plan):
        """Execute painting with iterative refinement using VLM"""
        current_canvas = self.camera.get_canvas_tensor(
            h=self.opt.h_render, w=self.opt.w_render).to(device) # blank
        
        # Calculate grid size
        grid_size = self.opt.squared_paper_num
        grid_h = self.opt.h_render / grid_size
        grid_w = self.opt.w_render / grid_size
        
        # Log target image
        target_tensor = torch.from_numpy(self.target_image.copy()).float()
        if len(target_tensor.shape) == 3:
            target_tensor = target_tensor.permute(2, 0, 1)
        target_tensor = target_tensor.unsqueeze(0)
        target_np = target_tensor.detach().cpu().numpy()[0].transpose(1, 2, 0)
        self.opt.writer.add_image('target_image', target_np, 0)

        # Initialize painting variables
        total_strokes = self.opt.num_strokes
        strokes_per_iter = total_strokes // self.opt.optim_iter
        regions = initial_plan['regions']
        
        # Sort regions by painting order
        regions.sort(key=lambda x: x['paint_order'])
        
        # For each optimization iteration
        for iter_idx in range(self.opt.optim_iter):
            # Convert current canvas to base64
            current_canvas_base64 = self._convert_to_base64(
                current_canvas[0].cpu().numpy().transpose(1, 2, 0))
            
            # Get current region based on progress
            current_region_idx = iter_idx * len(regions) // self.opt.optim_iter
            current_region = regions[current_region_idx]
            
            # Local planning prompt
            local_prompt = f"""
            Compare the target image and current canvas. Focus on the region {current_region['content']} 
            in grid coordinates {current_region['grid']}.
            
            The image is divided into a {grid_size}x{grid_size} grid. Each grid cell is {grid_h:.1f}x{grid_w:.1f} pixels.
            
            1. What details are missing or need improvement in this region?
            2. Suggest {strokes_per_iter} specific brush strokes to add, describing:
               - Start position (start_x, start_y) as grid coordinates (can use decimals for precise positions)
               - End position (end_x, end_y) as grid coordinates (can use decimals for precise positions)
               - Color (color name, hex code, or RGB values)
               - Bending (0-1, how much the stroke curves)
               - Thickness (0-1, stroke thickness)
            
            Format response as JSON list of strokes:
            {{
                "strokes": [
                    {{
                        "start_x": 5.5,  # grid x coordinate (0-{grid_size-1})
                        "start_y": 3.2,  # grid y coordinate (0-{grid_size-1})
                        "end_x": 6.5,    # grid x coordinate (0-{grid_size-1})
                        "end_y": 4.2,    # grid y coordinate (0-{grid_size-1})
                        "color": "#FF0000",  # or "red" or [255,0,0]
                        "bending": 0.1,
                        "thickness": 0.1
                    }},
                    ...
                ]
            }}
            """
            
            # Get stroke plan from VLM
            stroke_plan = self.generate(
                [self.target_img_base64, current_canvas_base64], 
                local_prompt,
                temperature=0.8
            )
            stroke_plan = self._clean_json_string(stroke_plan)
            stroke_plan = json.loads(stroke_plan)
            
            # Execute strokes
            for stroke_params in stroke_plan['strokes']:
                # Convert grid coordinates to proportional coordinates
                start_x_grid = stroke_params['start_x']
                start_y_grid = stroke_params['start_y']
                end_x_grid = stroke_params['end_x']
                end_y_grid = stroke_params['end_y']
                
                # Calculate stroke length and position
                dx = end_x_grid - start_x_grid
                dy = end_y_grid - start_y_grid
                length = np.sqrt(dx**2 + dy**2) / grid_size  # Normalize by grid size
                angle = np.arctan2(dy, dx)  # Calculate angle in radians
                
                # Convert start position to proportional coordinates
                x_prop = start_x_grid / grid_size
                y_prop = start_y_grid / grid_size
                
                # Create and execute stroke
                stroke = BrushStroke(self.opt)
                
                # Parse and map color to palette
                color_rgb = self._parse_color(stroke_params['color'])
                mapped_color = self._map_to_palette_color(color_rgb)
                if mapped_color == torch.tensor([1.0, 1.0, 1.0]):
                    print(color_rgb)
                stroke.color_transform = torch.nn.Parameter(mapped_color)
                
                stroke.stroke_length = torch.nn.Parameter(torch.tensor(length).float())
                stroke.stroke_bend = torch.nn.Parameter(torch.tensor(stroke_params['bending']).float())
                stroke.stroke_z = torch.nn.Parameter(torch.tensor(stroke_params['thickness']).float())
                stroke.transformation.a = torch.nn.Parameter(torch.tensor(angle).float())
                
                x_glob, y_glob, _ = canvas_to_global_coordinates(x_prop, y_prop, None, self.opt)
                stroke.execute(self, x_glob, y_glob, angle)
            
            # Update canvas
            current_canvas = self.camera.get_canvas_tensor(
                h=self.opt.h_render, w=self.opt.w_render).to(device)
            
            # Log progress
            if iter_idx % 1 == 0:
                canvas_np = current_canvas.detach().cpu().numpy()[0].transpose(1, 2, 0)
                canvas_np = np.clip(canvas_np, 0, 1)
                self.opt.writer.add_image('progressive_painting', canvas_np, iter_idx)
        
        # Log final result
        final_canvas = current_canvas.detach().cpu().numpy()[0].transpose(1, 2, 0)
        final_canvas = np.clip(final_canvas, 0, 1)
        self.opt.writer.add_image('final_result', final_canvas, self.opt.num_strokes)

def main(
    image_url = "https://cdn.homeandmoney.com/wp-content/uploads/2022/05/31113751/Pittsburgh_FeaturedImg-1.jpg",
    cache_dir = "caches/small_brush",
    simulate = True,
    squared_paper_num = 16,
    render_height = 256,
    use_cache = True,
    num_strokes = 800,
    optim_iter = 80,
    n_colors = 30,
    tensorboard_dir = "/mnt/c/Users/Public/painting_log",
    simulate_type = "vlm",
    ):
    opt = Options()
    opt.gather_options()
    opt.simulate = simulate
    opt.simulate_type = simulate_type
    opt.render_height = render_height
    opt.use_cache = use_cache
    opt.num_strokes = num_strokes
    opt.squared_paper_num = squared_paper_num
    opt.optim_iter = optim_iter
    opt.n_colors = n_colors
    opt.tensorboard_dir = tensorboard_dir
    opt.cache_dir = cache_dir
    opt.objective_data = image_url
    opt.dont_retrain_stroke_model = True

    date_and_time = datetime.datetime.now()
    run_name = '' + date_and_time.strftime("%m_%d__%H_%M_%S")
    opt.writer = TensorBoard('{}/{}'.format(opt.tensorboard_dir, run_name))
    opt.writer.add_text('args', str(sys.argv), 0)
    
    vlm_painter = VLMPainter(opt)
    
    if not opt.simulate:
        try:
            input('Make sure blank canvas is exposed. Press enter when ready.')
        except SyntaxError:
            pass
    
    # Start painting
    initial_plan_sketches = vlm_painter.plan()
    vlm_painter.paint(initial_plan_sketches)
    
    # Clean up
    vlm_painter.robot.good_night_robot()

if __name__ == '__main__':
    main()
