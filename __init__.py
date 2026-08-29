"""随机画风插件: 随机抽取画师并生成预览图。"""
from __future__ import annotations

from plugins.anr_plugin_random_artists.utils import (
    generate_random_artists,
    recover_txt,
    save_txt,
)
from utils.helpers import read_txt
from utils.plugins import Action, Field, Panel, Plugin


def register(plugin: Plugin):
    # ---------------- 生成预览 (内联动作, 无说明) ----------------
    preview_panel = Panel(
        id="preview",
        title="生成预览",
        icon="🖼️",
        fields=[],
        inline_actions=True,
        actions=[
            Action(
                id="generate",
                label="🚀 开始生成",
                uses_novelai=True,  # 调用 NovelAI 生图 API: 进入生图队列
                output="auto",
                show_output=False,
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

    # ---------------- 提示词设置 (角标预设) ----------------
    prompt_panel = Panel(
        id="prompt",
        title="提示词设置",
        icon="📝",
        fields=[
            Field(id="artists_positive", label="固定正面提示词 (可使用 wildcards)", type="textarea", rows=3, default="1girl, loli, cute", autocomplete=True),
            Field(id="add_quality_tags", label="⭐ 正面预设", type="corner_select", options=["Standard", "Light", "None"], default="Standard", corner_of="artists_positive"),
            Field(id="artists_negative", label="固定负面提示词", type="textarea", rows=3, default="nsfw, lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]", autocomplete=True),
            Field(id="undesired_contentc_preset", label="🚫 负面预设", type="corner_select", options=["Heavy", "Light", "Furry Focus", "Human Focus", "None"], default="None", corner_of="artists_negative"),
            Field(id="artists_position", label="画风串追加位置 (可使用 __artists__ 自定义位置)", type="radio", options=["最前面", "最后面", "自定义"], default="最后面"),
        ],
        actions=[],
    )

    # ---------------- 参数设置 (右侧说明保留) ----------------
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
            ], default="nai-diffusion-4-5-full", row_group="rg_model"),
            Field(id="furry_mode", label="", type="toggle", default=False, row_group="rg_model",
                  on_text="🐾 Furry", off_text="🌸 Anime",
                  description="点击切换 Furry 模式 (仅非 nai-3 模型可用)",
                  show_if={"field": "model", "not_in": ["nai-diffusion-3", "nai-diffusion-furry-3"]}),
            Field(id="resolution", label="分辨率预设", type="select", options=["832x1216", "1216x832", "1024x1024", "512x768", "768x768", "640x640", "自定义", "随机"], default="832x1216",
                  sync="WxH", inputs=["width", "height"]),
            Field(id="width", label="宽", type="number", default=832, row_group="rg_wh"),
            Field(id="height", label="高", type="number", default=1216, row_group="rg_wh"),
            Field(id="steps", label="采样步数", type="slider", min=1, max=50, step=1, default=23),
            Field(id="prompt_guidance", label="提示词指导系数", type="slider", min=0, max=10, step=0.1, default=5),
            Field(id="prompt_guidance_rescale", label="提示词重采样系数", type="slider", min=0, max=10, step=0.02, default=0),
            Field(id="seed", label="种子 (-1 为随机)", type="text", default="-1"),
            Field(id="sampler", label="采样器", type="select", options=["k_euler", "k_euler_ancestral", "k_dpmpp_2s_ancestral", "k_dpmpp_2m", "k_dpmpp_sde", "k_dpmpp_2m_sde", "ddim_v3", "随机"], default="k_euler_ancestral", row_group="rg_ss"),
            Field(id="noise_schedule", label="调度器", type="select", options=["native", "karras", "exponential", "polyexponential", "随机"], default="karras", row_group="rg_ss",
                  show_if=[{"field": "model", "not_in": ["nai-diffusion-5-full", "nai-diffusion-5-curated"]}, {"field": "sampler", "not_equals": "ddim_v3"}]),
            Field(id="variety", label="Variety+", type="checkbox", default=False, row_group="rg1", show_if={"field": "model", "not_in": ["nai-diffusion-5-full", "nai-diffusion-5-curated"]}),
            Field(id="decrisp", label="Decrisp", type="checkbox", default=False, row_group="rg1", show_if={"field": "model", "in": ["nai-diffusion-3", "nai-diffusion-furry-3"]}),
            Field(id="sm", label="SMEA", type="checkbox", default=False, row_group="rg2", show_if={"field": "model", "in": ["nai-diffusion-3", "nai-diffusion-furry-3"]}),
            Field(id="sm_dyn", label="DYN", type="checkbox", default=False, row_group="rg2",
                  show_if=[{"field": "model", "in": ["nai-diffusion-3", "nai-diffusion-furry-3"]}, {"field": "sm", "equals": True}]),
            Field(id="legacy_uc", label="Legacy Prompt Conditioning Mode", type="checkbox", default=False, show_if={"field": "model", "in": ["nai-diffusion-4-full", "nai-diffusion-4-curated-preview"]}),
            Field(id="checkbox_prob", label="复选框参数启用概率", type="slider", min=0, max=100, step=1, default=50, description="Variety+/Decrisp/SMEA/DYN 等复选框勾选后每次生图按此概率启用"),
            Field(id="vibe_file", label="*.naiv4vibebundle (nai5 不支持 vibe)", type="path", hidden=True),
            Field(id="params_note", label="提示", type="info", column="right", default=(
                "① 该处参数设置与左侧主页文生图出参数设置不通用\n"
                "② 复选框类参数 (如 Variety+/Decrisp/SMEA) 勾选后每次生图按下方\"复选框参数启用概率\"设置生效, 默认 50%, 可调至 100%\n"
                "③ 数值类参数目前仅可设置为固定值\n"
                "④ 下拉列表中可设置对应参数为随机\n"
                "⑤ 当选择不同模型时可设置的参数会发生改变\n"
                "nai5 目前不支持 vibe"
            )),
        ],
        actions=[],
    )

    # ---------------- 画师设置 (内联动作, 无输出框) ----------------
    artists_panel = Panel(
        id="artists",
        title="画师设置",
        icon="🎨",
        fields=[
            Field(id="artists_info", label="说明", type="info", default=(
                "该插件运行时会按照设定值抽取一定数量的行通过改变权重组成画风串, 每次启动时会加载位于 "
                "./plugins/anr_plugin_random_artists 目录下的 artists.txt, 若不想每次启动后都修改左侧文本区域中的画师提示词"
                "可以修改后点击保存文件, 若想还原该文件的修改只需将 artists_backup.txt 文件内容复制到 artists.txt"
            )),
            Field(id="artists_area", label="单画师提示词或光影质量提示词等", type="textarea", rows=8, default=read_txt("./plugins/anr_plugin_random_artists/artists.txt"), autocomplete=True, autosize=True),
            Field(id="add_artist", label="artist 前缀", type="checkbox", default=False),
        ],
        actions=[
            Action(id="save", label="💾 保存文件", inputs=["artists_area"], show_output=False, stop=False, uses_novelai=False,
                   handler=lambda v: {"text": save_txt(v.get("artists_area", ""))}),
            Action(id="recover", label="🔄 还原文件", inputs=[], show_output=False, set_field="artists_area", stop=False, uses_novelai=False,
                   handler=lambda v: {"text": "已还原!", "content": recover_txt()}),
        ],
    )

    # ---------------- 概率设置 (重置默认参数 + 右侧分布图/说明) ----------------
    prob_panel = Panel(
        id="prob",
        title="概率设置",
        icon="🎲",
        reset_defaults=True,
        fields=[
            Field(id="min_artists_num", label="最少抽取画师数量", type="slider", min=1, max=10, step=1, default=2),
            Field(id="max_artists_num", label="最多抽取画师数量", type="slider", min=2, max=20, step=1, default=10),
            Field(id="years", label="年份标签 (50% 概率)", type="checkbox_group", options=["year_2022", "year_2023", "year_2024", "year_2025", "year_2026"], default=[]),
            Field(id="enable_random_weight", label="随机权重", type="checkbox", default=False),
            Field(id="prod_mode", label="权重模式", type="radio", options=["新版权重", "旧版权重"], default="新版权重",
                  show_if={"field": "enable_random_weight", "equals": True}),
            # 新版权重
            Field(id="min_weight", label="下界", type="slider", min=-5, max=1, step=1, default=-3,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="max_weight", label="上界", type="slider", min=-1, max=5, step=1, default=3,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="mode", label="众数", type="slider", min=-5, max=5, step=1, default=1,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="left_sharpness", label="众数左侧数据离散程度", type="slider", min=1, max=20, step=1, default=10,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="right_sharpness", label="众数右侧数据离散程度", type="slider", min=1, max=20, step=1, default=5,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="prob_neg_to_pos", label="负数转化概率", type="slider", min=0, max=1, step=0.01, default=0.7,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="prob_zero_to_one_add", label="数集 [0,1] 增加 0.5 的概率", type="slider", min=0, max=1, step=0.01, default=0.35,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            # 旧版权重
            Field(id="min_num", label="最少添加括号次数", type="slider", min=0, max=9, step=1, default=0,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "旧版权重"}]),
            Field(id="max_num", label="最多添加括号次数", type="slider", min=1, max=10, step=1, default=3,
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "旧版权重"}]),
            Field(id="use_parentheses", label="括号类型", type="checkbox_group", options=["使用[]", "使用{}"], default=["使用[]", "使用{}"],
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "旧版权重"}]),
            # 右侧: 实时分布图 + 权重说明 (新版权重/旧版权重互斥显示)
            Field(id="dist_chart", label="数据分布图", type="chart", column="right",
                  inputs=["min_weight", "max_weight", "mode", "left_sharpness", "right_sharpness", "prob_neg_to_pos", "prob_zero_to_one_add"],
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="weight_note", label="权重说明", type="info", column="right",
                  default=(
                      "关于新版权重, 我引入了一个分段 Beta 分布, 它可以较为自由的控制左侧和右侧数据离散程度, "
                      "并设置众数以及上界和下界, 数据分布图反应了可能取到的权重, x 轴表示权重, y 轴表示概率, "
                      "如果你不理解它们是如何工作的, 默认数据可以应对大部分场景\n\n"
                      "① 上界: 最大权重\n"
                      "② 下界: 最小权重\n"
                      "③ 众数: 最可能的权重\n"
                      "④ 离散程度: 越高越靠近众数, 反之远离众数\n"
                      "⑤ 负数转化概率: 当随机出一个负数时, 它变成正数的概率, 原因是在正面提示词中一般较少使用负数权重, "
                      "因此取其绝对值以增加正数的概率\n"
                      "⑥ 数集增加概率: 当随机出一个范围在 [0,1] 的数时, 它增加 0.5 的概率, "
                      "原因是由于该范围权重对提示词影响较小, 因此增加 0.5 以增加其对提示词的影响\n\n"
                      "关于展示的直方图忽高忽低, 原因是因为我对生成的随机数进行了处理, 生成的随机数最多为 2 位小数, "
                      "当生成一个 2 位小数且第 2 位小数为 5 时, 不做任何处理, 否则使用四舍五入的规则进位或退位, "
                      "因此 2 位小数的数据量较少, 导致频数较低, 可以根据核密度估计曲线更平滑地查看数据分布"
                  ),
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "新版权重"}]),
            Field(id="weight_note_old", label="权重说明", type="info", column="right",
                  default=("关于旧版权重: 旧版权重更加适用于 nai-diffusion-3 或 nai-diffusion-furry-3 等旧模型"),
                  show_if=[{"field": "enable_random_weight", "equals": True}, {"field": "prod_mode", "equals": "旧版权重"}]),
        ],
        actions=[],
    )

    plugin.title = "随机画风"
    plugin.description = "随机抽取画师与权重生成风格化预览图"
    plugin.icon = "🎲"
    plugin.panels.extend([preview_panel, prompt_panel, params_panel, artists_panel, prob_panel])