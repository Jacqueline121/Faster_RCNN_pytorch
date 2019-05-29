# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
import torch
import PIL
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

STANDARD_COLORS = [
    'Red', 'Green', 'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

NUM_COLORS = len(STANDARD_COLORS)

font_path = os.path.join(os.path.dirname(__file__), 'arial.ttf')

FONT = ImageFont.truetype(font_path, 20)


def _draw_single_box2(image, xmin, ymin, xmax, ymax, color='black', thickness=2):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)

    return image


def _draw_single_box(image, xmin, ymin, xmax, ymax, display_str, font=None, color='black', thickness=2):
    draw = ImageDraw.Draw(image)
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
             (right, top), (left, top)], width=thickness, fill=color)
    text_bottom = bottom
    # Reverse list and print from bottom to top.
    text_width, text_height = font.getsize(display_str)
    margin = np.ceil(0.05 * text_height)

    # draw.rectangle(
    #     [(left, text_bottom - text_height - 2 * margin), (left + text_width,
    #                                                     text_bottom)],fill=color)

    draw.text(
        (left + margin, text_bottom - text_height - margin),
        display_str,
        fill=color,
        font=font)

    return image


def draw_detection_boxes(image, boxes, gt_classes=None, class_names=None, thresh=0, gt=False):
    num_boxes = boxes.shape[0]
    disp_image = image
    for i in range(num_boxes):
        bbox = tuple(np.round(boxes[i, :4]).astype(np.int64))
        score = boxes[i, 4]
        if score < thresh:
            continue
        gt_class_ind = gt_classes[i]
        class_name = class_names[gt_class_ind]
        disp_str = '{}: {:.2f}'.format(class_name, score)
        if gt:
            color = STANDARD_COLORS[1]
        else:
            color = STANDARD_COLORS[0]
        disp_image = _draw_single_box(disp_image,
                                      bbox[0],
                                      bbox[1],
                                      bbox[2],
                                      bbox[3],
                                      disp_str,
                                      FONT,
                                      color=color)
    return disp_image


def draw_detection_boxes2(image, boxes):
    num_boxes = boxes.shape[0]
    print(boxes.shape)
    disp_image = image
    for i in range(num_boxes):
        bbox = tuple(np.round(boxes[i, :4]).astype(np.int64))
        disp_image = _draw_single_box2(disp_image,
                                      bbox[0],
                                      bbox[1],
                                      bbox[2],
                                      bbox[3],
                                      color=STANDARD_COLORS[0])
    return disp_image