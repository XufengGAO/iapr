import copy

import matplotlib
import matplotlib.pyplot as plt
from PIL import ImageDraw


def vizCropProcedures(imgs_debug, img_crop_origin, figsize=(20, 12), main_title=None):
    fig, ax = plt.subplots(2, 3, figsize=figsize)
    ax[0, 0].imshow(imgs_debug['img_binary'], cmap='gray')
    ax[0, 0].set_title('img_binary')
    ax[0, 1].imshow(imgs_debug['img_morph'], cmap='gray')
    ax[0, 1].set_title('img_morph')
    ax[0, 2].imshow(imgs_debug['img_dist_trans_thresh'], cmap='gray')
    ax[0, 2].set_title('img_dist_trans_thresh')
    ax[1, 0].imshow(imgs_debug['img_contour'], cmap='gray')
    ax[1, 0].set_title('img_contour')
    ax[1, 1].imshow(imgs_debug['img_origin_rect'])
    ax[1, 1].set_title('img_origin_rect')
    ax[1, 2].imshow(img_crop_origin)
    ax[1, 2].set_title('img_crop_origin')
    if main_title is not None:
        fig.suptitle(main_title)


def plotMultipleImages(
    nrows, ncols, images, titles, cmap, bold_axis=None, fig_title=None, figsize=(14, 8)
):
    fig = plt.figure(figsize=figsize)
    plt.axis('off')
    fig.subplots_adjust(hspace=0.4, wspace=0.3)
    for i in range(len(titles)):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        # ax.axis('off')
        ax.set_title(titles[i])
        ax.set_xlabel(titles[i])
        if cmap[i] == 'rgb':
            ax.imshow(images[i])
        else:
            ax.imshow(images[i], cmap=cmap[i])
        # set bold boundaries
        if bold_axis is not None:
            if bold_axis[i]:
                for child in ax.get_children():
                    if isinstance(child, matplotlib.spines.Spine):
                        child.set(lw=5, color='red')
    if fig_title is not None:
        fig.suptitle(fig_title)
    plt.show()


def plotBboxInPlace(tables_crop_pil, boxes, im_names, fig_title=None):

    draw_image = copy.deepcopy(tables_crop_pil).convert("RGBA")
    draw = ImageDraw.Draw(draw_image)
    for idx, box in enumerate(boxes):
        draw.rectangle(
            (tuple(box[:2]), tuple(box[2:])), outline='black', width=50
        )  # , fill="black"
        draw.text(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), im_names[idx])
    fig, ax = plt.subplots(figsize=(6, 6))
    if fig_title is not None:
        ax.set_title(fig_title)
    ax.imshow(draw_image)
