import os

import gradio as gr

from plugins.anr_plugin_random_artists.utils import (
    generate_random_artists,
    save_txt,
    update_components_for_models_change,
    update_from_dropdown,
    visualize_beta_distribution,
)
from utils import read_json, read_txt
from utils.components import (
    update_components_for_sampler_change,
    update_components_for_sm_change,
    update_from_width_or_height,
)
from utils.logger import logger
from utils.variable import MODELS, NOISE_SCHEDULE, SAMPLER, UC_PRESET


def plugin():
    with gr.Tab("随机画风"):
        with gr.Tab("生成预览"):
            with gr.Row():
                generate_button = gr.Button("开始生成")
                stop_button = gr.Button("停止生成")
            gr.Markdown(
                "停止生成会等待最后一张图片生成完毕, 并不会立刻停止, 如果遇到无法停止生成的情况, 浏览器刷新 ANR 页面即可"
            )
            output_prompt = gr.Textbox(label="本次随机的画风串", interactive=False)
            output_image = gr.Image(label="随机画风串生成的图片", interactive=False)
            gr.Markdown(
                "建议长时间运行时在浏览器设置中将 ANR 地址添加到睡眠白名单, 否则可能会导致一段时间后无法继续生成图片"
            )

        with gr.Tab("提示词设置"):
            with gr.Row():
                artists_positive = gr.TextArea(
                    value="1girl, loli, cute",
                    placeholder="可以在此处使用 wildcards",
                    label="固定正面提示词",
                    interactive=True,
                    scale=3,
                )
                with gr.Column(scale=1):
                    furry_mode = gr.Button("🌸", visible=True)
                    furry_mode.click(
                        lambda x: "🐾" if x == "🌸" else "🌸",
                        inputs=furry_mode,
                        outputs=furry_mode,
                    )
                    add_quality_tags = gr.Checkbox(
                        value=True, label="添加质量词", interactive=True
                    )
                    gr.Markdown("<hr>")
                    artists_position = gr.Radio(
                        ["最前面", "最后面", "自定义"],
                        value="最后面",
                        label="画风串追加位置(可使用 __artists__ 自定义位置)",
                        interactive=True,
                    )

            with gr.Row():
                artists_negative = gr.TextArea(
                    value="nsfw, lowres, {bad}, error, fewer, extra, missing, worst quality, jpeg artifacts, bad quality, watermark, unfinished, displeasing, chromatic aberration, signature, extra digits, artistic error, username, scan, [abstract]",
                    label="固定负面提示词",
                    placeholder="可以在此处使用 wildcards",
                    interactive=True,
                    scale=3,
                )
                undesired_contentc_preset = gr.Dropdown(
                    choices=[
                        x
                        for x in UC_PRESET
                        if x
                        not in {
                            "nai-diffusion-4-5-full": [],
                            "nai-diffusion-4-5-curated": ["Furry Focus"],
                            "nai-diffusion-4-full": ["Furry Focus", "Human Focus"],
                            "nai-diffusion-4-curated-preview": [
                                "Furry Focus",
                                "Human Focus",
                            ],
                            "nai-diffusion-3": ["Furry Focus"],
                            "nai-diffusion-furry-3": ["Furry Focus", "Human Focus"],
                        }.get("nai-diffusion-4-5-full", [])
                    ],
                    value="None",
                    label="负面提示词预设",
                    interactive=True,
                    scale=1,
                )
            gr.Markdown("可以在在固定的提示词中使用 wildcards")
        with gr.Tab("参数设置"):
            with gr.Row():
                with gr.Column():
                    model = gr.Dropdown(
                        choices=MODELS,
                        value="nai-diffusion-4-5-full",
                        label="生图模型",
                        interactive=True,
                        scale=1,
                    )
                    resolution = gr.Dropdown(
                        choices=[
                            "832x1216",
                            "1216x832",
                            "512x768",
                            "768x768",
                            "640x640",
                        ]
                        + ["自定义", "随机"],
                        value="832x1216",
                        label="分辨率预设",
                        interactive=True,
                    )
                    with gr.Row():
                        width = gr.Slider(
                            minimum=0,
                            maximum=50000,
                            value=832,
                            step=64,
                            label="宽",
                            interactive=True,
                        )
                        height = gr.Slider(
                            minimum=0,
                            maximum=50000,
                            value=1216,
                            step=64,
                            label="高",
                            interactive=True,
                        )
                    resolution.change(
                        fn=update_from_dropdown,
                        inputs=[resolution],
                        outputs=[width, height],
                    )
                    width.change(
                        fn=update_from_width_or_height,
                        inputs=[width, height, resolution],
                        outputs=[resolution, gr.Radio(visible=False)],
                    )
                    height.change(
                        fn=update_from_width_or_height,
                        inputs=[width, height, resolution],
                        outputs=[resolution, gr.Radio(visible=False)],
                    )
                    steps = gr.Slider(
                        minimum=1,
                        maximum=50,
                        value=23,
                        label="采样步数",
                        step=1,
                        interactive=True,
                    )
                    prompt_guidance = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=5,
                        label="提示词指导系数",
                        step=0.1,
                        interactive=True,
                    )
                    prompt_guidance_rescale = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=0,
                        label="提示词重采样系数",
                        step=0.02,
                        interactive=True,
                    )
                    with gr.Row():
                        variety = gr.Checkbox(
                            value=False,
                            label="Variety+",
                            interactive=True,
                        )
                        decrisp = gr.Checkbox(
                            value=False,
                            label="Decrisp",
                            visible=False,
                            interactive=True,
                        )
                    with gr.Row():
                        sm = gr.Checkbox(
                            value=False,
                            label="SMEA",
                            visible=False,
                            interactive=True,
                        )
                        sm_dyn = gr.Checkbox(
                            value=False,
                            label="DYN",
                            visible=False,
                            interactive=True,
                        )
                        sm.change(update_components_for_sm_change, sm, sm_dyn)
                    with gr.Row():
                        seed = gr.Textbox(
                            value="-1", label="种子", interactive=True, scale=4
                        )
                    with gr.Row(scale=1):
                        last_seed = gr.Button(value="♻️", size="sm")
                        random_seed = gr.Button(value="🎲", size="sm")
                        last_seed.click(
                            lambda: (
                                read_json("last.json")["parameters"]["seed"]
                                if os.path.exists("last.json")
                                else "-1"
                            ),
                            outputs=seed,
                        )
                        random_seed.click(lambda: "-1", outputs=seed)
                    sampler = gr.Dropdown(
                        choices=[x for x in SAMPLER if x != "ddim_v3"] + ["随机"],
                        value="k_euler_ancestral",
                        label="采样器",
                        interactive=True,
                    )
                    noise_schedule = gr.Dropdown(
                        choices=[x for x in NOISE_SCHEDULE if x != "native"] + ["随机"],
                        value="karras",
                        label="调度器",
                        interactive=True,
                    )
                    sampler.change(
                        update_components_for_sampler_change,
                        inputs=sampler,
                        outputs=[noise_schedule, sm],
                    )
                    legacy_uc = gr.Checkbox(
                        value=False,
                        label="Legacy Prompt Conditioning Mode",
                        visible=False,
                        interactive=True,
                    )
                    model.change(
                        update_components_for_models_change,
                        inputs=model,
                        outputs=[
                            decrisp,
                            sm,
                            legacy_uc,
                            sampler,
                            noise_schedule,
                            undesired_contentc_preset,
                            furry_mode,
                        ],
                    )
                with gr.Column():
                    gr.Markdown("① 该处参数设置与左侧主页文生图出参数设置不通用")
                    gr.Markdown(
                        "② 复选框类参数勾选后每次生图默认有 50% 的概率启用, 可以去概率设置中设置修改为 100%"
                    )
                    gr.Markdown("③ 数值类参数目前仅可设置为固定值")
                    gr.Markdown("④ 下拉列表中可设置对应参数为随机")
                    gr.Markdown("③ 当选择不同模型时可设置的参数会发生改变")
        with gr.Tab("画师设置"):
            with gr.Row():
                with gr.Column():
                    artists_area = gr.TextArea(
                        read_txt("./plugins/anr_plugin_random_artists/artists.txt"),
                        label="单画师提示词或光影质量提示词等",
                        lines=30,
                        interactive=True,
                    )
                    add_artist = gr.Checkbox(False, label="artist 前缀")
                with gr.Column():
                    gr.Markdown(
                        "该插件运行时会按照设定值抽取一定数量的行通过改变权重组成画风串, 每次启动时会加载位于 ./plugins/anr_plugin_random_artists 目录下的 artists.txt, 若不想每次启动后都修改左侧文本区域中的画师提示词可以修改后点击保存文件, 若想还原该文件的修改只需将 artists_backup.txt 文件内容复制到 artists.txt"
                    )
                    save_message = gr.Markdown(show_label=False, visible=False)
                    open_file_button = gr.Button("保存文件")
                    open_file_button.click(save_txt, artists_area, save_message)
        with gr.Tab("概率设置"):
            with gr.Row():
                with gr.Column(scale=1):
                    min_artists_num = gr.Slider(
                        1, 10, value=2, label="最少抽取画师数量", interactive=True
                    )
                    max_artists_num = gr.Slider(
                        2, 20, value=10, label="最多抽取画师数量", interactive=True
                    )

                    years = gr.CheckboxGroup(
                        ["year_2022", "year_2023", "year_2024", "year_2025"],
                        show_label=False,
                    )

                    gr.Markdown("<hr>")
                    enable_random_weight = gr.Checkbox(False, label="随机权重")
                    prod_mode = gr.Radio(
                        ["新版权重", "旧版权重"],
                        value="新版权重",
                        show_label=False,
                        visible=False,
                    )

                    random_weight = gr.Column(visible=False)
                    with random_weight:
                        new_prod = gr.Column()
                        with new_prod:
                            min_weight = gr.Slider(
                                -5, 1, value=-3, step=1, label="下界"
                            )
                            max_weight = gr.Slider(-1, 5, value=3, step=1, label="上界")
                            mode = gr.Slider(-5, 5, value=1, step=1, label="众数")
                            left_sharpness = gr.Slider(
                                1, 20, value=10, step=1, label="众数左侧数据离散程度"
                            )
                            right_sharpness = gr.Slider(
                                1, 20, value=5, step=1, label="众数右侧数据离散程度"
                            )
                            prob_neg_to_pos = gr.Slider(
                                0, 1, value=0.7, step=0.01, label="负数转化概率"
                            )
                            prob_zero_to_one_add = gr.Slider(
                                0,
                                1,
                                value=0.35,
                                step=0.01,
                                label="数集 [0, 1] 增加 0.5 的概率",
                            )
                            refresh_button = gr.Button("刷新数据分布图")

                        old_prob = gr.Column(visible=False)
                        with old_prob:
                            min_num = gr.Slider(
                                0,
                                9,
                                0,
                                step=1,
                                label="最少添加括号次数",
                                interactive=True,
                            )
                            max_num = gr.Slider(
                                1,
                                10,
                                3,
                                step=1,
                                label="最多添加括号次数",
                                interactive=True,
                            )
                            use_parentheses = gr.CheckboxGroup(
                                ["使用[]", "使用{}"],
                                value=["使用[]", "使用{}"],
                                show_label=False,
                            )

                    prod_mode.change(
                        lambda x: (
                            (gr.update(visible=True), gr.update(visible=False))
                            if x == "新版权重"
                            else (gr.update(visible=False), gr.update(visible=True))
                        ),
                        inputs=prod_mode,
                        outputs=[new_prod, old_prob],
                    )
                explanation = gr.Column(scale=2, visible=False)
                with explanation:
                    prob_image = gr.Image(
                        "./plugins/anr_plugin_random_artists/beta_random_artists.png",
                        label="数据分布图",
                        interactive=False,
                    )
                    gr.Markdown(
                        "关于新版权重, 我引入了一个分段 Beta 分布, 它可以较为自由的控制左侧和右侧数据离散程度, 并设置众数以及上界和下界, 数据分布图反应了可能取到的权重, x 轴表示权重, y 轴表示概率, 如果你不理解它们是如何工作的, 默认数据可以应对大部分场景"
                    )
                    gr.Markdown("① 上界: 最大权重")
                    gr.Markdown("② 下界: 最小权重")
                    gr.Markdown("③ 众数: 最可能的权重")
                    gr.Markdown("④ 离散程度: 越高越靠近众数, 反之远离众数")
                    gr.Markdown(
                        "⑤ 负数转化概率: 当随机出一个负数时, 它变成正数的概率, 原因是在正面提示词中一般较少使用负数权重, 因此取其绝对值以增加正数的概率"
                    )
                    gr.Markdown(
                        "⑥ 数集增加概率: 当随机出一个范围在 [0,1] 的数时, 它增加 0.5 的概率, 原因是由于该范围权重对提示词影响较小, 因此增加 0.5 以增加其对提示词的影响"
                    )
                    gr.Markdown(
                        "关于展示的直方图忽高忽低, 原因是因为我对生成的随机数进行了处理, 生成的随机数最多为 2 位小数, 当生成一个 2 位小数且第 2 位小数为 5 时, 不做任何处理, 否则使用四舍五入的规则进位或退位, 因此 2 位小数的数据量较少, 导致频数较低, 可以根据核密度估计曲线更平滑地查看数据分布"
                    )
                    gr.Markdown("<hr>")
                    gr.Markdown(
                        "关于旧版权重: 旧版权重更加适用于 nai-diffusion-3 或 nai-diffusion-furry-3 等旧模型"
                    )

                enable_random_weight.change(
                    lambda x: (
                        (
                            gr.update(visible=True),
                            gr.update(visible=True),
                            gr.update(visible=True),
                        )
                        if x
                        else (
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                        )
                    ),
                    inputs=enable_random_weight,
                    outputs=[random_weight, prod_mode, explanation],
                )

                refresh_button.click(
                    visualize_beta_distribution,
                    inputs=[
                        min_weight,
                        max_weight,
                        mode,
                        left_sharpness,
                        right_sharpness,
                        prob_neg_to_pos,
                        prob_zero_to_one_add,
                    ],
                    outputs=prob_image,
                )

        cancel = output_image.change(
            generate_random_artists,
            inputs=[
                model,
                artists_positive,
                artists_position,
                artists_negative,
                undesired_contentc_preset,
                furry_mode,
                add_quality_tags,
                resolution,
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
                artists_area,
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
                add_artist,
            ],
            outputs=[output_prompt, output_image],
            show_progress=False,
        )
        generate_button.click(
            generate_random_artists,
            inputs=[
                model,
                artists_positive,
                artists_position,
                artists_negative,
                undesired_contentc_preset,
                furry_mode,
                add_quality_tags,
                resolution,
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
                artists_area,
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
                add_artist,
            ],
            outputs=[output_prompt, output_image],
        )
        stop_button.click(
            lambda: logger.warning("正在停止生成..."), None, None, cancels=[cancel]
        )
