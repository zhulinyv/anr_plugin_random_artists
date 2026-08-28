"""随机画风插件核心逻辑: 分段 Beta 分布权重 + 随机抽取画师 + 生图。"""
from __future__ import annotations

import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import ujson as json  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

from utils.config import env
from utils.generator import Generator
from utils.helpers import (
    check_stop,
    find_and_replace_wildcards_from_dict,
    format_str,
    read_json,
    return_last_value,
    return_x64,
    sleep_for_cool,
    sleep_interruptible,
)
from utils.logger import logger
from utils.models import *  # noqa: F401,F403
from utils.variable import (
    NOISE_SCHEDULE,
    SAMPLER,
    return_quality_preset_id,
    return_quality_tags,
    return_skip_cfg_above_sigma,
    return_uc_preset_id,
    return_undesired_contentc_preset,
)

generator = Generator("https://image.novelai.net/ai/generate-image")

ARTISTS_FILE = "./plugins/anr_plugin_random_artists/artists.txt"
ARTISTS_BACKUP = "./plugins/anr_plugin_random_artists/artists_backup.txt"


# ---------------------------------------------------------------- Beta 分布


def generate_piecewise_beta(
    a=-3,
    b=3,
    mode=0,
    left_sharpness=5,
    right_sharpness=5,
    prob_neg_to_pos=0.0,
    prob_zero_to_one_add=0.0,
):
    """分段 Beta 分布: 左右形状独立, 支持负数转化与 [0,1] 增加概率。"""
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

    f_left_mode = alpha_left / L_left if beta_left == 1 else 0
    f_right_mode = beta_right / L_right if alpha_right == 1 else 0
    total = f_left_mode + f_right_mode
    p_left = 0.5 if total == 0 else f_right_mode / total

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


def sample_piecewise_beta_numpy(
    n,
    a=-3,
    b=3,
    mode=0,
    left_sharpness=5,
    right_sharpness=5,
    prob_neg_to_pos=0.0,
    prob_zero_to_one_add=0.0,
):
    """向量化采样分段 Beta 分布 (与 generate_piecewise_beta 逻辑一致, 但批量 numpy 计算, 快约 100 倍)。"""
    rng = np.random.default_rng()
    if a > b:
        a, b = b, a
    mode = max(a + 1e-6, min(b - 1e-6, mode))
    prob_neg_to_pos = max(0.0, min(1.0, prob_neg_to_pos))
    prob_zero_to_one_add = max(0.0, min(1.0, prob_zero_to_one_add))

    L_left = mode - a
    L_right = b - mode
    alpha_left = max(1.0, left_sharpness + 1)
    beta_right = max(1.0, right_sharpness + 1)

    f_left_mode = alpha_left / L_left
    f_right_mode = beta_right / L_right
    total = f_left_mode + f_right_mode
    p_left = 0.5 if total == 0 else f_right_mode / total

    u = rng.random(n)
    left_mask = u < p_left
    n_left = int(left_mask.sum())
    n_right = n - n_left

    raw = np.empty(n)
    raw[left_mask] = a + rng.beta(alpha_left, 1.0, size=n_left) * L_left
    raw[~left_mask] = mode + rng.beta(1.0, beta_right, size=n_right) * L_right

    neg = raw < 0
    to_pos = neg & (rng.random(n) < prob_neg_to_pos)
    raw[to_pos] = np.minimum(np.abs(raw[to_pos]), b)

    in_01 = (raw >= 0) & (raw <= 1)
    add = in_01 & (rng.random(n) < prob_zero_to_one_add)
    raw[add] = np.minimum(raw[add] + 0.5, b)

    num2 = np.round(raw, 2)
    ends5 = np.round(np.abs(num2) * 100) % 10 == 5
    return np.where(ends5, num2, np.round(num2, 1))


def visualize_beta_distribution(a, b, mode, left_sharpness, right_sharpness, prob_neg_to_pos, prob_zero_to_one_add):
    """生成分布图并返回图片路径 (向量化采样 + 降低 dpi, 速度大幅提升)。"""
    data = sample_piecewise_beta_numpy(
        30000, a=a, b=b, mode=mode, left_sharpness=left_sharpness,
        right_sharpness=right_sharpness, prob_neg_to_pos=prob_neg_to_pos,
        prob_zero_to_one_add=prob_zero_to_one_add,
    )
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=120, density=True, alpha=0.7, color="mediumseagreen", edgecolor="black", linewidth=0.5, label="Histogram")
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 500)
    plt.plot(x_range, kde(x_range), color="c", linewidth=2, label="KDE")
    plt.title("Asymmetric Beta Distribution with 0→1 Addition", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.grid(axis="y", alpha=0.4)
    plt.axvline(x=mode, color="red", linestyle="--", label="Mode")
    plt.axvline(x=a, color="orange", linestyle="--", label="Lower Bound")
    plt.axvline(x=b, color="orange", linestyle="--", label="Upper Bound")
    plt.legend()
    os.makedirs("./outputs", exist_ok=True)
    plt.savefig("./outputs/temp_random_artists.png", dpi=150, bbox_inches="tight")
    plt.close()
    return "./outputs/temp_random_artists.png"


# ---------------------------------------------------------------- 文件


def save_txt(txt, path=ARTISTS_FILE):
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    logger.success("画师列表已保存!")
    return "已保存!"


def recover_txt(path=ARTISTS_BACKUP):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    save_txt(txt)
    return txt


def random_line_skip_blank(text: str):
    lines = [line for line in text.splitlines() if line.strip()]
    return random.choice(lines) if lines else ""


# ---------------------------------------------------------------- 生图


def generate_random_artists(values: dict):
    """随机画风生图: 循环生成直到用户停止, 每次产出 (画风串, 图片路径)。"""
    model = values.get("model", "nai-diffusion-4-5-full")
    artists_positive = values.get("artists_positive", "1girl, loli, cute")
    artists_position = values.get("artists_position", "最后面")
    artists_negative = values.get("artists_negative", "")
    undesired_contentc_preset = values.get("undesired_contentc_preset", "None")
    furry_mode = values.get("furry_mode", False)
    add_quality_tags = values.get("add_quality_tags", "Standard")
    resolution = values.get("resolution", "832x1216")
    width = int(values.get("width", 832))
    height = int(values.get("height", 1216))
    steps = int(values.get("steps", 23))
    prompt_guidance = float(values.get("prompt_guidance", 5))
    prompt_guidance_rescale = float(values.get("prompt_guidance_rescale", 0))
    variety = values.get("variety", False)
    decrisp = values.get("decrisp", False)
    sm = values.get("sm", False)
    sm_dyn = values.get("sm_dyn", False)
    # 复选框类参数启用概率 (0~100, 默认 50), 供 variety/decrisp/sm/sm_dyn 随机启用
    checkbox_prob = max(0.0, min(1.0, float(values.get("checkbox_prob", 50)) / 100.0))
    seed = str(values.get("seed", "-1"))
    sampler = values.get("sampler", "k_euler_ancestral")
    noise_schedule = values.get("noise_schedule", "karras")
    legacy_uc = values.get("legacy_uc", False)
    artists_area = values.get("artists_area", "")
    min_artists_num = int(values.get("min_artists_num", 2))
    max_artists_num = int(values.get("max_artists_num", 10))
    years = values.get("years", [])
    enable_random_weight = values.get("enable_random_weight", False)
    prod_mode = values.get("prod_mode", "新版权重")
    min_weight = float(values.get("min_weight", -3))
    max_weight = float(values.get("max_weight", 3))
    mode = float(values.get("mode", 1))
    left_sharpness = float(values.get("left_sharpness", 10))
    right_sharpness = float(values.get("right_sharpness", 5))
    prob_neg_to_pos = float(values.get("prob_neg_to_pos", 0.7))
    prob_zero_to_one_add = float(values.get("prob_zero_to_one_add", 0.35))
    min_num = int(values.get("min_num", 0))
    max_num = int(values.get("max_num", 3))
    use_parentheses = values.get("use_parentheses", ["使用[]", "使用{}"])
    add_artist = values.get("add_artist", False)
    vibe_file = values.get("vibe_file", None)

    os.makedirs("./outputs", exist_ok=True)
    with open("./outputs/temp_break.json", "w") as f:
        json.dump({"break": False}, f)

    count = 0
    while count < 1000:
        if check_stop():
            logger.warning("已停止生成!")
            break
        count += 1
        logger.info("正在生成图片...")

        if furry_mode and model not in ["nai-diffusion-3", "nai-diffusion-furry-3"]:
            artists_positive = "fur dataset, " + artists_positive

        lines = artists_area.splitlines()
        non_blank_artists = [line.strip() for line in lines if line.strip()] or [""]
        target_num = random.randint(min_artists_num, max_artists_num)
        actual_num = min(target_num, len(non_blank_artists))
        selected = random.sample(non_blank_artists, actual_num)
        artists_string = ""

        for artist in selected:
            if add_artist:
                artist = f"artist:{artist}"
            if enable_random_weight:
                if prod_mode == "新版权重":
                    artists_string += f"{generate_piecewise_beta(min_weight, max_weight, mode, left_sharpness, right_sharpness, prob_neg_to_pos, prob_zero_to_one_add)}::{artist},::, "
                else:
                    parentheses_list = []
                    if "使用[]" in use_parentheses:
                        parentheses_list.append(["[", "]"])
                    if "使用{}" in use_parentheses:
                        parentheses_list.append(["{", "}"])
                    num = random.randint(min_num, max_num)
                    symbol = random.choice(parentheses_list) if parentheses_list else ["", ""]
                    artists_string += symbol[0] * num + artist + symbol[1] * num + ", "
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
            final_string = artists_positive.replace("__artists__", f",{artists_string}")

        model_function_map = {
            "nai-diffusion-5-full": nai5ft2i,
            "nai-diffusion-5-curated": nai5ct2i,
            "nai-diffusion-4-5-full": nai45ft2i,
            "nai-diffusion-4-5-curated": nai45ct2i,
            "nai-diffusion-4-full": nai4ft2i,
            "nai-diffusion-4-curated-preview": nai4cpt2i,
            "nai-diffusion-3": nai3t2i,
            "nai-diffusion-furry-3": naif3t2i,
        }
        func = model_function_map.get(model)

        if resolution == "随机":
            w, h = random.choice(["832x1216", "1024x1024", "1216x832"]).split("x")
        elif resolution == "自定义":
            w, h = str(width), str(height)
        else:
            w, h = resolution.split("x")

        current_sampler = random.choice(SAMPLER if model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else [x for x in SAMPLER if x != "ddim_v3"]) if sampler == "随机" else sampler
        current_noise = random.choice(NOISE_SCHEDULE if model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else [x for x in NOISE_SCHEDULE if x != "native"]) if noise_schedule == "随机" else noise_schedule

        reference_image_multiple = []
        reference_information_extracted_multiple = []
        reference_strength_multiple = []

        if vibe_file:
            model_function_map = {
                "nai-diffusion-4-5-full": nai45fvibe,
                "nai-diffusion-4-5-curated": nai45cvibe,
                "nai-diffusion-4-full": nai4fvibe,
                "nai-diffusion-4-curated-preview": nai4cpvibe,
            }
            func = model_function_map.get(model, func)
            model_vibe_map = {
                "nai-diffusion-4-5-full": "v4-5full",
                "nai-diffusion-4-5-curated": "v4-5curated",
                "nai-diffusion-4-full": "v4full",
                "nai-diffusion-4-curated-preview": "v4curated",
            }
            vibe_data = read_json(vibe_file)
            vibe_model_name = model_vibe_map.get(model)
            if vibe_model_name:
                try:
                    for vibe_image in vibe_data["vibes"]:
                        reference_image_multiple.append(return_last_value(vibe_image["encodings"][vibe_model_name])["encoding"])
                        reference_strength_multiple.append(vibe_image["importInfo"]["strength"])
                except KeyError:
                    reference_image_multiple.append(return_last_value(vibe_data["encodings"][vibe_model_name])["encoding"])
                    reference_strength_multiple.append(vibe_data["importInfo"]["strength"])

        json_data = func(
            _input=format_str(f"{final_string}, " + return_quality_tags(model, add_quality_tags) if add_quality_tags != "None" else final_string),
            params_version=4,
            width=return_x64(int(w)),
            height=return_x64(int(h)),
            scale=prompt_guidance,
            sampler=current_sampler,
            steps=steps,
            n_samples=1,
            ucPresetId=return_uc_preset_id(model)[undesired_contentc_preset],
            qualityPresetId=return_quality_preset_id(model)[add_quality_tags],
            autoSmea=False,
            dynamic_thresholding=(random.random() < checkbox_prob if (decrisp if model in ["nai-diffusion-3", "nai-diffusion-furry-3"] else False) else False),
            controlnet_strength=1,
            legacy=False,
            add_original_image=True,
            cfg_rescale=prompt_guidance_rescale,
            noise_schedule="karras" if model in ["nai-diffusion-5-full", "nai-diffusion-5-curated"] else current_noise,
            legacy_v3_extend=False,
            skip_cfg_above_sigma=(return_skip_cfg_above_sigma(model) if (variety and random.random() < checkbox_prob) else None),
            use_coords=False,
            normalize_reference_strength_multiple=True,
            inpaintImg2ImgStrength=1,
            use_order=True,
            legacy_uc=legacy_uc if model in ["nai-diffusion-4-full", "nai-diffusion-4-curated-preview"] else False,
            seed=random.randint(1000000000, 9999999999) if seed == "-1" else int(seed),
            negative_prompt=format_str(return_undesired_contentc_preset(model, undesired_contentc_preset) + (f", {artists_negative}" if undesired_contentc_preset != "None" else artists_negative)),
            deliberate_euler_ancestral_bug=False,
            prefer_brownian=True,
            use_new_shared_trial=True,
            sm=random.random() < checkbox_prob if sm else False,
            sm_dyn=random.random() < checkbox_prob if sm_dyn else False,
            reference_image_multiple=reference_image_multiple,
            reference_information_extracted_multiple=reference_information_extracted_multiple,
            reference_strength_multiple=reference_strength_multiple,
            v4_prompt_positive=[],
            v4_prompt_negative=[],
            characterPrompts=[],
            straight_alpha=True,
        )

        with open("./outputs/temp_last_origin.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)

        image_data = None
        path = None
        while image_data is None:
            if check_stop():
                logger.warning("已停止生成!")
                break
            try:
                image_data = generator.generate(find_and_replace_wildcards_from_dict(json_data))
            except Exception as e:
                logger.error(f"网络或请求异常: {e}")
                logger.opt(exception=True).debug("生成请求异常堆栈:")
                image_data = None
            # 生成完成(或失败)后若已请求停止: 立即退出, 不再等待冷却/重试
            if check_stop():
                logger.warning("已停止生成!")
                break
            if image_data:
                path = generator.save(image_data, "text2image", json_data["parameters"]["seed"])
                break
            # 失败: 等待后重试 (等待期间检测停止信号, 可被立即打断)
            sleep_for_cool(env.cool_time)
            logger.info("正在重试...")

        yield artists_string, path
        sleep_interruptible(1.5)

    # 正常结束时 (被停止) 也返回最后一次结果
