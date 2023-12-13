import inspect
import os
from pathlib import Path
import imgaug.augmenters as iaa

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    FixedTextColorCfg,
    SimpleTextColorCfg,
    TextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.layout.same_line import SameLineLayout
from text_renderer.layout.extra_text_line import ExtraTextLineLayout
from text_renderer.effect.curve import Curve
from text_renderer.layout import SameLineLayout, ExtraTextLineLayout

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
CHAR_DIR = DATA_DIR / "char"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "text"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(20, 30),
)

perspective_transform = NormPerspectiveTransformCfg(20, 20, 1.5)


def get_char_corpus():
    return CharCorpus(
        CharCorpusCfg(
            text_paths=[TEXT_DIR / "chn_text.txt", TEXT_DIR / "eng_text.txt"],
            filter_by_chars=True,
            chars_file=CHAR_DIR / "chn.txt",
            length=(5, 10),
            char_spacing=(-0.3, 1.3),
            **font_cfg
        ),
    )


def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=False
):
    return GeneratorCfg(
        num_image=50,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )

def enum_data_0():
    print(0)
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "vn_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg
            ),
        ),
    )

def enum_data_1():
    print(1)
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "vn_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg
            ),
        ),
        corpus_effects=DropoutHorizontal(p=1, num_line=1, thickness=3),
    )

def enum_data_2():
    print(2)
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "vn_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg
            ),
        ),
        corpus_effects=[DropoutRand(p=1, dropout_p=(0.3, 0.5))],
    )
    
def enum_data_3():
    print(3)
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "vn_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                text_color_cfg=FixedTextColorCfg(),
                **font_cfg
            ),
        ),
        corpus_effects=[DropoutVertical(p=1, num_line=15)],
    )

def enum_data_4():
    print(4)
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=EnumCorpus(
            EnumCorpusCfg(
                text_paths=[TEXT_DIR / "vn_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                text_color_cfg=FixedTextColorCfg(),
                height = 48
                **font_cfg
            ),
        ),
        corpus_effects=[Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
                        ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),],
    )

configs = [
    #enum_data_0(),

    enum_data_1(),

    #enum_data_2(),

    #enum_data_3(),

    #enum_data_4(),

]
