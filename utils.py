import random

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import ujson as json
from scipy.stats import gaussian_kde

from utils import find_and_replace_wildcards_from_dict, return_x64, sleep_for_cool
from utils.environment import env
from utils.generator import Generator
from utils.logger import logger
from utils.models import *  # noqa
from utils.variable import (
    NOISE_SCHEDULE,
    SAMPLER,
    UC_PRESET,
    return_quality_tags,
    return_skip_cfg_above_sigma,
    return_uc_preset_data,
    return_undesired_contentc_preset,
)

generator = Generator("https://image.novelai.net/ai/generate-image")


def generate_piecewise_beta(
    a=-3, b=3, mode=0, left_sharpness=5, right_sharpness=5, prob_neg_to_pos=0.0, prob_zero_to_one_add=0.0
):
    """
    分段 Beta 分布：左侧用 Beta(alpha_left, beta_left) 缩放到 [a, mode]
                    右侧用 Beta(alpha_right, beta_right) 缩放到 [mode, b]
    左右形状独立，通过混合比例保证众数处连续（近似）。

    新增参数：
        prob_neg_to_pos: 当生成的数在 (-∞,0) 范围内时, 转化为整数的概率
        prob_zero_to_one_add : 当生成的数在 [0,1] 范围内时，额外加 0.5 的概率
    """
    if a > b:
        a, b = b, a
    mode = max(a + 1e-6, min(b - 1e-6, mode))
    prob_neg_to_pos = max(0.0, min(1.0, prob_neg_to_pos))
    prob_zero_to_one_add = max(0.0, min(1.0, prob_zero_to_one_add))

    L_left = mode - a
    L_right = b - mode

    alpha_left = max(1.0, left_sharpness + 1)
    beta_left = 1.0

    alpha_right = 1.0
    beta_right = max(1.0, right_sharpness + 1)

    f_left_mode = (alpha_left) / L_left if beta_left == 1 else 0
    f_right_mode = (beta_right) / L_right if alpha_right == 1 else 0

    total = f_left_mode + f_right_mode
    if total == 0:
        p_left = 0.5
    else:
        p_left = f_right_mode / total

    if random.random() < p_left:
        u = random.betavariate(alpha_left, beta_left)
        raw = a + u * L_left
    else:
        u = random.betavariate(alpha_right, beta_right)
        raw = mode + u * L_right

    if raw < 0 and random.random() < prob_neg_to_pos:
        raw = min(abs(raw), b)

    if 0 <= raw <= 1 and random.random() < prob_zero_to_one_add:
        raw = min(raw + 0.5, b)

    num_2_decimals = round(raw, 2)
    num_str = f"{abs(num_2_decimals):.2f}"
    return num_2_decimals if num_str[-1] == "5" else round(num_2_decimals, 1)


def visualize_beta_distribution(a, b, mode, left_sharpness, right_sharpness, prob_neg_to_pos, prob_zero_to_one_add):
    data = [
        generate_piecewise_beta(
            a=a,
            b=b,
            mode=mode,
            left_sharpness=left_sharpness,
            right_sharpness=right_sharpness,
            prob_neg_to_pos=prob_neg_to_pos,
            prob_zero_to_one_add=prob_zero_to_one_add,
        )
        for _ in range(100000)
    ]

    plt.figure(figsize=(10, 6))
    plt.hist(
        data,
        bins=120,
        density=True,
        alpha=0.7,
        color="mediumseagreen",
        edgecolor="black",
        linewidth=0.5,
        label="Histogram",
    )

    kde = gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 500)
    kde_values = kde(x_range)

    plt.plot(x_range, kde_values, color="c", linewidth=2, label="KDE")

    plt.title("Asymmetric Beta Distribution with 0→1 Addition", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(axis="y", alpha=0.4)
    plt.axvline(x=mode, color="red", linestyle="--", label="Mode")
    plt.axvline(x=a, color="orange", linestyle="--", label="Lower Bound")
    plt.axvline(x=b, color="orange", linestyle="--", label="Upper Bound")

    plt.legend()
    plt.savefig("./outputs/temp_random_artists.png", dpi=300, bbox_inches="tight")
    return "./outputs/temp_random_artists.png"


def update_components_for_models_change(model):
    _SAMPLER = SAMPLER[:]
    _SAMPLER.remove("ddim_v3")
    _NOISE_SCHEDULE = NOISE_SCHEDULE[:]
    _NOISE_SCHEDULE.remove("native")
    _UC_PRESET = UC_PRESET[:]
    if model in ["nai-diffusion-4-5-full", "nai-diffusion-4-5-curated"]:
        if model == "nai-diffusion-4-5-curated":
            _UC_PRESET.remove("Furry Focus")
        return (
            gr.update(visible=False),  # decrisp
            gr.update(visible=False),  # sm
            gr.update(visible=False),  # legacy_uc
            gr.update(choices=_SAMPLER + ["随机"]),  # sampler
            gr.update(choices=_NOISE_SCHEDULE + ["随机"]),  # noise_schedule
            gr.update(choices=_UC_PRESET),  # uc_preset
            gr.update(visible=True),  # furry_mode
        )
    elif model in ["nai-diffusion-4-full", "nai-diffusion-4-curated-preview"]:
        _UC_PRESET.remove("Furry Focus")
        _UC_PRESET.remove("Human Focus")
        return (
            gr.update(visible=False),  # decrisp
            gr.update(visible=False),  # sm
            gr.update(visible=True),  # legacy_uc
            gr.update(choices=_SAMPLER + ["随机"]),  # sampler
            gr.update(choices=_NOISE_SCHEDULE + ["随机"]),  # noise_schedule
            gr.update(choices=_UC_PRESET),  # uc_preset
            gr.update(visible=True),  # furry_mode
        )
    elif model in ["nai-diffusion-3", "nai-diffusion-furry-3"]:
        _UC_PRESET.remove("Furry Focus")
        if model == "nai-diffusion-furry-3":
            _UC_PRESET.remove("Human Focus")
        return (
            gr.update(visible=True),  # decrisp
            gr.update(visible=True),  # sm
            gr.update(visible=False),  # legacy_uc
            gr.update(choices=SAMPLER + ["随机"]),  # sampler
            gr.update(choices=NOISE_SCHEDULE + ["随机"]),  # noise_schedule
            gr.update(choices=_UC_PRESET),  # uc_preset
            gr.update(visible=False),  # furry_mode
        )


def save_txt(txt, path="./plugins/anr_plugin_random_artists/artists.txt"):
    with open(path, "w", encoding="utf-8") as file:
        file.write(txt)
    return gr.update(value="已保存!", visible=True)


def random_line_skip_blank(text: str):
    lines = text.splitlines()
    non_blank_lines = [line for line in lines if line.strip() != ""]
    return random.choice(non_blank_lines)


def update_from_dropdown(resolution_choice):
    if resolution_choice in ["自定义", "随机"]:
        return gr.update(), gr.update()

    width, height = map(int, resolution_choice.split("x"))
    return width, height


def generate_random_artists(
    model,
    artists_positive: str,
    artists_position,
    artists_negative,
    undesired_contentc_preset,
    furry_mode,
    add_quality_tags,
    resolution: str,
    width,
    height,
    steps,
    prompt_guidance,
    prompt_guidance_rescale,
    variety,
    decrisp,
    sm,
    sm_dyn,
    seed,
    sampler,
    noise_schedule,
    legacy_uc,
    artists_area: str,
    min_artists_num,
    max_artists_num,
    years,
    enable_random_weight,
    prod_mode,
    min_weight,
    max_weight,
    mode,
    left_sharpness,
    right_sharpness,
    prob_neg_to_pos,
    prob_zero_to_one_add,
    min_num,
    max_num,
    use_parentheses,
):
    logger.info("正在生成图片...")

    if furry_mode == "🐾" and model not in ["nai-diffusion-3", "nai-diffusion-furry-3"]:
        artists_positive = "fur dataset, " + artists_positive

    artists_num = random.randint(min_artists_num, max_artists_num)

    artists_string = ""

    for i in range(artists_num):
        artist = random_line_skip_blank(artists_area)
        while artist in artists_string:
            artist = random_line_skip_blank(artists_area)
        if enable_random_weight:
            if prod_mode == "新版权重":
                artists_string += f"{generate_piecewise_beta(min_weight, max_weight, mode, left_sharpness, right_sharpness, prob_neg_to_pos, prob_zero_to_one_add)}::{artist}::,"
            else:
                parentheses_list = []
                if "使用[]" in use_parentheses:
                    parentheses_list.append(["[", "]"])
                if "使用{}" in use_parentheses:
                    parentheses_list.append(["{", "}"])
                num = random.randint(min_num, max_num)
                symbol = random.choice(parentheses_list) if parentheses_list is not None else ["", ""]
                artists_string = artists_string + symbol[0] * num + artist + symbol[1] * num + ","
        else:
            artists_string += f"{artist},"

    for year in years:
        if random.random() > 0.5:
            artists_string += f"{year},"

    if artists_position == "最前面":
        final_string = f"{artists_string},{artists_positive}"
    elif artists_position == "最后面":
        final_string = f"{artists_positive},{artists_string}"
    else:
        final_string = artists_positive.replace("__artists__", f",{artists_string},")

    model_function_map = {
        "nai-diffusion-4-5-full": nai45ft2i,  # noqa
        "nai-diffusion-4-5-curated": nai45ct2i,  # noqa
        "nai-diffusion-4-full": nai4ft2i,  # noqa
        "nai-diffusion-4-curated-preview": nai4cpt2i,  # noqa
        "nai-diffusion-3": nai3t2i,  # noqa
        "nai-diffusion-furry-3": naif3t2i,  # noqa
    }
    func = model_function_map.get(model)

    if resolution == "随机":
        w, h = random.choice(["832x1216", "1024x1024", "1216x832"]).split("x")
    elif resolution == "自定义":
        w, h = str(width), str(height)
    else:
        w, h = resolution.split("x")

    if sampler == "随机":
        sampler = random.choice(
            SAMPLER if model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else [x for x in SAMPLER if x != "ddim_v3"]
        )

    if noise_schedule == "随机":
        noise_schedule = random.choice(
            NOISE_SCHEDULE
            if model in ["nai-diffusion-3", "nai-diffusion-furry-3"]
            else [x for x in NOISE_SCHEDULE if x != "native"]
        )

    json_data = func(
        _input=final_string + return_quality_tags(model) if add_quality_tags else final_string,
        width=return_x64(int(w)),
        height=return_x64(int(h)),
        scale=prompt_guidance,
        sampler=sampler,
        steps=steps,
        ucPreset=return_uc_preset_data(model)[undesired_contentc_preset],
        qualityToggle=add_quality_tags,
        autoSmea=False,
        dynamic_thresholding=(
            random.choice([True, False])
            if (decrisp if model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False)
            else False
        ),
        legacy=False,
        add_original_image=True,
        cfg_rescale=prompt_guidance_rescale,
        noise_schedule=noise_schedule,
        legacy_v3_extend=False,
        skip_cfg_above_sigma=(random.choice([return_skip_cfg_above_sigma(model), None]) if variety else None),
        use_coords=False,
        normalize_reference_strength_multiple=True,
        use_order=True,
        legacy_uc=legacy_uc if model in ["nai-diffusion-4-full", "nai-diffusion-4-curated-preview"] else False,
        seed=random.randint(1000000000, 9999999999) if seed == "-1" else int(seed),
        negative_prompt=return_undesired_contentc_preset(model, undesired_contentc_preset)
        + (f", {artists_negative}" if undesired_contentc_preset != "None" else artists_negative),
        deliberate_euler_ancestral_bug=False,  # 仅在采样器为 k_euler_ancestral 时出现
        prefer_brownian=True,  # 仅在采样器为 k_euler_ancestral 时出现
        use_new_shared_trial=True,
        sm=random.choice([True, False]) if sm else False,
        sm_dyn=random.choice([True, False]) if sm_dyn else False,
        # reference_image_multiple=reference_image_multiple,
        # reference_information_extracted_multiple=reference_information_extracted_multiple,
        # reference_strength_multiple=reference_strength_multiple,
        v4_prompt_positive=[],
        v4_prompt_negative=[],
        characterPrompts=[],
        # director_reference_images_cached=director_reference_images_cached,
        # director_reference_descriptions=director_reference_descriptions,
        # director_reference_information_extracted=director_reference_information_extracted,
        # director_reference_strength_values=director_reference_strength_values,
        # director_reference_secondary_strength_values=director_reference_secondary_strength_values,
    )

    with open("./outputs/temp_last_origin.json", "w", encoding="utf-8") as file:
        json.dump(json_data, file, ensure_ascii=False, indent=4)

    image_data = None
    while image_data is None:
        image_data = generator.generate(find_and_replace_wildcards_from_dict(json_data))
        sleep_for_cool(env.cool_time)
        if image_data:
            path = generator.save(image_data, "text2image", json_data["parameters"]["seed"])
            break
        logger.info("正在重试...")

    return artists_string, path
