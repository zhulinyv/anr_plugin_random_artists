"""随机画风插件: 随机抽取画师并生成预览图。"""
from __future__ import annotations

from plugins.anr_plugin_random_artists.utils import (
    generate_random_artists,
    recover_txt,
    save_txt,
    visualize_beta_distribution,
)
from utils.helpers import read_txt
from utils.plugins import Action, Field, Panel, Plugin


def register(plugin: Plugin):
    # ---------------- 生成预览 ----------------
    preview_panel = Panel(
        id="preview",
        title="生成预览",
        icon="🖼️",
        fields=[
            Field(id="note", label="说明", type="info", default="点击开始生成后会持续随机生成画风预览, 点击停止按钮结束"),
        ],
        actions=[
            Action(
                id="generate",
                label="🚀 开始生成",
                inputs=[
                    "model", "artists_positive", "artists_position", "artists_negative",
                    "undesired_contentc_preset", "furry_mode", "add_quality_tags",
                    "resolution", "width", "height", "steps", "prompt_guidance", "prompt_guidance_rescale",
                    "variety", "decrisp", "sm", "sm_dyn", "seed", "sampler", "noise_schedule", "legacy_uc",
                    "artists_area", "min_artists_num", "max_artists_num", "years",
                    "enable_random_weight", "prod_mode", "min_weight", "max_weight", "mode",
                    "left_sharpness", "right_sharpness", "prob_neg_to_pos", "prob_zero_to_one_add",
                    "min_num", "max_num", "use_parentheses", "add_artist", "vibe_file",
                ],
                handler=generate_random_artists,
            ),
        ],
    )

    # ---------------- 提示词设置 ----------------
    prompt_panel = Panel(
        id="prompt",
        title="提示词设置",
        icon="📝",
        fields=[
            Field(id="artists_positive", label="固定正面提示词 (可使用 wildcards)", type="textarea", rows=3, default="1girl, loli, cute", autocomplete=True),
            Field(id="artists_negative", label="固定负面提示词", type="textarea", rows=3, default="nsfw, lowres, {bad}, error, worst quality, jpeg artifacts, watermark", autocomplete=True),
            Field(id="artists_position", label="画风串追加位置 (可使用 __artists__ 自定义位置)", type="radio", options=["最前面", "最后面", "自定义"], default="最后面"),
            Field(id="undesired_contentc_preset", label="负面提示词预设", type="select", options=["Heavy", "Light", "Furry Focus", "Human Focus", "None"], default="None"),
            Field(id="add_quality_tags", label="正面提示词预设", type="select", options=["Standard", "Light", "None"], default="Standard"),
            Field(id="furry_mode", label="Furry 模式", type="checkbox", default=False),
            Field(id="add_artist", label="artist 前缀", type="checkbox", default=False),
        ],
        actions=[],
    )

    # ---------------- 参数设置 ----------------
    params_panel = Panel(
        id="params",
        title="参数设置",
        icon="🎛️",
        fields=[
            Field(id="model", label="生图模型", type="select", options=[
                "nai-diffusion-5-full", "nai-diffusion-5-curated",
                "nai-diffusion-4-5-full", "nai-diffusion-4-5-curated",
                "nai-diffusion-4-full", "nai-diffusion-4-curated-preview",
                "nai-diffusion-3", "nai-diffusion-furry-3",
            ], default="nai-diffusion-4-5-full"),
            Field(id="resolution", label="分辨率预设", type="select", options=["832x1216", "1216x832", "1024x1024", "512x768", "768x768", "640x640", "自定义", "随机"], default="832x1216"),
            Field(id="width", label="宽", type="number", default=832),
            Field(id="height", label="高", type="number", default=1216),
            Field(id="steps", label="采样步数", type="slider", min=1, max=50, step=1, default=23),
            Field(id="prompt_guidance", label="提示词指导系数", type="slider", min=0, max=10, step=0.1, default=5),
            Field(id="prompt_guidance_rescale", label="提示词重采样系数", type="slider", min=0, max=10, step=0.02, default=0),
            Field(id="seed", label="种子 (-1 为随机)", type="text", default="-1"),
            Field(id="sampler", label="采样器", type="select", options=["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m", "k_dpmpp_sde", "k_dpmpp_2m_sde", "ddim_v3", "随机"], default="k_euler_ancestral"),
            Field(id="noise_schedule", label="调度器", type="select", options=["native", "karras", "exponential", "polyexponential", "随机"], default="karras"),
            Field(id="variety", label="Variety+ (50% 概率)", type="checkbox", default=False),
            Field(id="decrisp", label="Decrisp (50% 概率)", type="checkbox", default=False),
            Field(id="sm", label="SMEA (50% 概率)", type="checkbox", default=False),
            Field(id="sm_dyn", label="DYN (50% 概率)", type="checkbox", default=False),
            Field(id="legacy_uc", label="Legacy Prompt Conditioning Mode", type="checkbox", default=False),
            Field(id="vibe_file", label="*.naiv4vibebundle (nai5 不支持 vibe)", type="path"),
            Field(id="note", label="提示", type="info", default="数值类参数为固定值; 下拉列表可设为随机; 复选框 50% 概率启用"),
        ],
        actions=[],
    )

    # ---------------- 画师设置 ----------------
    artists_panel = Panel(
        id="artists",
        title="画师设置",
        icon="🎨",
        fields=[
            Field(id="artists_area", label="单画师提示词或光影质量提示词等", type="textarea", rows=12, default=read_txt("./plugins/anr_plugin_random_artists/artists.txt"), autocomplete=True),
        ],
        actions=[
            Action(id="save", label="💾 保存文件", inputs=["artists_area"], handler=lambda v: {"text": save_txt(v.get("artists_area", ""))}),
            Action(id="recover", label="🔄 还原文件", inputs=[], handler=lambda v: {"text": recover_txt()}),
        ],
    )

    # ---------------- 概率设置 ----------------
    prob_panel = Panel(
        id="prob",
        title="概率设置",
        icon="🎲",
        fields=[
            Field(id="min_artists_num", label="最少抽取画师数量", type="slider", min=1, max=10, step=1, default=2),
            Field(id="max_artists_num", label="最多抽取画师数量", type="slider", min=2, max=20, step=1, default=10),
            Field(id="years", label="年份标签 (50% 概率)", type="checkbox_group", options=["year_2022", "year_2023", "year_2024", "year_2025", "year_2026"], default=[]),
            Field(id="enable_random_weight", label="随机权重", type="checkbox", default=False),
            Field(id="prod_mode", label="权重模式", type="radio", options=["新版权重", "旧版权重"], default="新版权重", show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="min_weight", label="下界", type="slider", min=-5, max=1, step=1, default=-3, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="max_weight", label="上界", type="slider", min=-1, max=5, step=1, default=3, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="mode", label="众数", type="slider", min=-5, max=5, step=1, default=1, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="left_sharpness", label="众数左侧数据离散程度", type="slider", min=1, max=20, step=1, default=10, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="right_sharpness", label="众数右侧数据离散程度", type="slider", min=1, max=20, step=1, default=5, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="prob_neg_to_pos", label="负数转化概率", type="slider", min=0, max=1, step=0.01, default=0.7, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="prob_zero_to_one_add", label="数集 [0,1] 增加 0.5 的概率", type="slider", min=0, max=1, step=0.01, default=0.35, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="min_num", label="最少添加括号次数", type="slider", min=0, max=9, step=1, default=0, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="max_num", label="最多添加括号次数", type="slider", min=1, max=10, step=1, default=3, show_if={"field": "enable_random_weight", "equals": True}),
            Field(id="use_parentheses", label="括号类型", type="checkbox_group", options=["使用[]", "使用{}"], default=["使用[]", "使用{}"], show_if={"field": "enable_random_weight", "equals": True}),
        ],
        actions=[
            Action(
                id="refresh",
                label="📊 刷新数据分布图",
                inputs=["min_weight", "max_weight", "mode", "left_sharpness", "right_sharpness", "prob_neg_to_pos", "prob_zero_to_one_add"],
                handler=lambda v: {"image": visualize_beta_distribution(float(v.get("min_weight", -3)), float(v.get("max_weight", 3)), float(v.get("mode", 1)), float(v.get("left_sharpness", 10)), float(v.get("right_sharpness", 5)), float(v.get("prob_neg_to_pos", 0.7)), float(v.get("prob_zero_to_one_add", 0.35)))},
            ),
        ],
    )

    plugin.title = "随机画风"
    plugin.description = "随机抽取画师与权重生成风格化预览图"
    plugin.icon = "🎲"
    plugin.panels.extend([preview_panel, prompt_panel, params_panel, artists_panel, prob_panel])
