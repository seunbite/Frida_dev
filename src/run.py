import requests
import PIL
from io import BytesIO
import os
import sys


def save_image_from_internet(url, fn):
    response = requests.get(url)
    img = PIL.Image.open(BytesIO(response.content))
    img = img.convert('RGB')
    img.save(fn)


def run(
    image_url = "https://cdn.homeandmoney.com/wp-content/uploads/2022/05/31113751/Pittsburgh_FeaturedImg-1.jpg",
    cache_dir = "caches/small_brush",
    simulate = True,
    render_height = 256,
    use_cache = True,
    objective = "clip_conv_loss",
    objective_weight = 1.0,
    lr_multiplier = 0.4,
    num_strokes = 800,
    optim_iter = 400,
    n_colors = 30,
    tensorboard_dir = "/mnt/c/Users/Public/painting_log",
    simulate_type = "original",
    ):
    os.system(f"wget {image_url}")
    image = image_url.split('/')[-1]

    os.system(f"python paint.py \
        --simulate \
        --render_height {render_height} \
        --use_cache \
        --cache_dir {cache_dir}  \
        --dont_retrain_stroke_model \
        --objective {objective} \
        --objective_data {image}  \
        --objective_weight {objective_weight} \
        --lr_multiplier {lr_multiplier} \
        --num_strokes {num_strokes} \
        --optim_iter {optim_iter} \
        --n_colors {n_colors} \
        --tensorboard_dir {tensorboard_dir} \
        --simulate_type {simulate_type}"
        )


if __name__ == "__main__":
    import fire
    fire.Fire(run)
