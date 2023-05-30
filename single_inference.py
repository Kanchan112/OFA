import re
import sys
from typing import Optional

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from PIL import Image, ImageDraw
from skimage.measure import find_contours, label, regionprops
from tasks.mm_tasks.vqa_gen import VqaGenTask

# Image transform
from torchvision import transforms as T

from OFA.utils.zero_shot_utils import zero_shot_step

# Register VQA task
tasks.register_task("vqa_gen", VqaGenTask)

# turn on cuda if GPU is available
#use_cuda = torch.cuda.is_available()
#if use_cuda:
#    torch.cuda.set_device('cuda:1')
use_cuda = False

# use fp16 only when GPU is available
use_fp16 = False

# specify some options for evaluation
parser = options.get_generation_parser()
input_args = [
    "",
    "--task=vqa_gen",
    "--beam=100",
    "--unnormalized",
    "--path=/mnt/Enterprise/kanchan/VLM-SEG-2023___/OFA/checkpoints/ofa_large_384.pt",
    "--bpe-dir=/mnt/Enterprise/kanchan/VLM-SEG-2023___/OFA/utils/BPE",
]
args = options.parse_args_and_arch(parser, input_args)
cfg = convert_namespace_to_omegaconf(args)

# Load pretrained ckpt & config
task = tasks.setup_task(cfg.task)


def return_model(cfg=cfg):

    models, cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path), task=task
    )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    return models


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = T.Compose(
    (
        T.Resize(
            (cfg.task.patch_image_size, cfg.task.patch_image_size),
            interpolation=Image.BICUBIC,
        ),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std, inplace=True),
    )
)

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


# Compile regex to remove extra spaces
RE_TWO_OR_MORE_SPACES = re.compile(r"\s{2,}")


# Normalize the question
def pre_question(question: str, max_ques_words: int):
    question = (
        re.sub(
            RE_TWO_OR_MORE_SPACES,
            " ",
            question.lower().lstrip(",.!?*#:;~").replace("-", " ").replace("/", " "),
        )
        .rstrip("\n")
        .strip(" ")
    )
    # truncate question
    question_words = question.split(" ")
    if len(question_words) > max_ques_words:
        question = " ".join(question_words[:max_ques_words])
    return question


def encode_text(
    text: str,
    length: Optional[int] = None,
    append_bos: bool = False,
    append_eos: bool = False,
):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text), add_if_not_exist=False, append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s


# Construct input for open-domain VQA task
def construct_sample(image: Image.Image, question: str):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])

    question = pre_question(question, task.cfg.max_src_length)

    if not question.endswith("?"):
        question = f"{question}?"

    src_text = encode_text(f" {question}", append_bos=True, append_eos=True).unsqueeze(
        0
    )

    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    ref_dict = np.array([{"yes": 1.0}])  # just placeholder

    sample = {
        "id": np.array(["42"]),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "ref_dict": ref_dict,
    }
    return sample


# Function to turn FP32 to FP16
def apply_half(t: torch.Tensor):
    if t.is_floating_point() and t.dtype != torch.half:
        return t.to(dtype=torch.half)
    return t


def mask_to_border(mask: np.ndarray):
    """Convert a mask to border image"""
    border = np.zeros_like(mask)

    contours = find_contours(mask, 128)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border


def mask_to_props(mask: np.ndarray):
    """Mask to region props"""
    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    return props


""" Mask to bounding features """


def mask_to_overall_bbox(mask: Image.Image):
    """
    function to get auxiliary information of image
    Args:
        mask (Image.Image): A PIL Image object
    Returns:
        tuple: a tuple of overall bbox coordinates

    """
    mask = np.asarray(mask.convert("L"))

    props = mask_to_props(mask)

    bbox_area_greater = tuple(prop.bbox for prop in props if prop.area_bbox > 4)

    min_y1, min_x1 = mask.shape
    max_x2 = 0
    max_y2 = 0

    if len(bbox_area_greater) == 0:
        return (min_x1, min_y1, max_x2, max_y2)

    x1_y1_x2_y2 = np.array(bbox_area_greater)[:, [1, 0, 3, 2]]

    min_x1, min_y1 = np.minimum(x1_y1_x2_y2[:, :2], (min_x1, min_y1))[0]

    max_x2, max_y2 = np.maximum(x1_y1_x2_y2[:, 2:], (max_x2, max_y2))[0]

    return (min_x1, min_y1, max_x2, max_y2)


def get_answer(
    models,
    image: Image.Image,
    mask: Image.Image,
    question: str,
    verbose: bool = False,
):
    generator = task.build_generator(models, cfg.generation)

    image = image.convert("RGB")
    w, h = image.size

    image_draw = ImageDraw.Draw(image)
    bbox_coords = mask_to_overall_bbox(mask)
    image_draw.rectangle(
        (
            max(0, bbox_coords[0] - 10),
            max(0, bbox_coords[1] - 10),
            min(w, bbox_coords[2] + 10),
            min(h, bbox_coords[3] + 10),
        ),
        width=5,
        outline="#00ff00",
    )

    sample = construct_sample(image, question)

    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    # Run eval step for open-domain VQA
    with torch.inference_mode():
        result, scores = zero_shot_step(task, generator, models, sample)

    answer: str = result[0]["answer"]

    if verbose:
        print(f"Question: {question}")
        print(f"OFA's Answer: {answer}")

    return answer


if __name__ == "__main__":
    import os.path

    ROOT_DIR = "/mnt/Enterprise/PUBLIC_DATASETS/polyp_datasets/Kvasir-SEG"

    image_number = 1
    image_path = os.path.join(ROOT_DIR, "images_cf", f"{image_number}.jpg")
    mask_path = os.path.join(ROOT_DIR, "masks_cf", f"{image_number}.png")
    questions = (
        "What is the shape of bump enclosed in green box?",
        "What is the color of bump enclosed in green box?",
        "What is the size of bump enclosed in green box?",
        "What is the texture of bump enclosed in green box?",
        "What is the location of bump enclosed in green box?",
        "What is the color of the box?",
    )

    models = return_model()
    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        get_answer(models, image, mask, question=questions[0], verbose=True)
