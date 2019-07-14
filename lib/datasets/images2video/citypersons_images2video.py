
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np

"""
This scripts prepare the CityPerson dataset, so we can use this dataset to train our model.
"""

def expand_image_to_short_video(im_path, out_dir, length=2, clean_up=True):
    """
    This function random crop a patch from the input image, and resize it
    to the origin size. Then these 'length' images are compressed into a
    raw video.
    :param im_path: the source image path
    :param out_dir: the output directory to save the random processed images
    :param length: the number of the images
    :param clean_up: bool, whether to remove the transformed images
    :return:
    """

    if not os.path.exists(im_path):
        raise RuntimeError('image not exist {}'.format(im_path))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    im_split = im_path.split('.')
    extent = im_split[len(im_split)-1]

    im = cv2.imread(im_path)
    # if this is a gray image
    if len(im.shape) == 2:
        im = im[:, :, np.newaxis]
        im = np.concatenate((im, im, im), axis=2)

    # stor the ist image, the origin image
    all_out_im_path = []
    frame_id = 1
    out_im_path = os.path.join(out_dir, str(frame_id).zfill(6) + '.' + extent)
    all_out_im_path.append(out_im_path)
    cv2.imwrite(out_im_path, im)

    h, w, c = im.shape[0], im.shape[1], im.shape[2]

    # preserve the image_info of the images
    # each line is [frame_id, x1, y1, x2, y2, x_scale, y_scale]
    # the first line is the info for the origin image
    im_info = np.zeros((length, 7))  # [frame_id, x1, y1, x2, y2, x_sacle, y_scale]
    im_info[0,:] = np.array([1, 0, 0, w, h, 1, 1])

    # handle the following images
    max_scale = 0.99
    min_scale = 0.97

    for frame_id in range(2, length+1): # frame id starts from 1

        x_scale = np.random.choice(np.linspace(min_scale, max_scale, 20))
        y_scale = np.random.choice(np.linspace(min_scale, max_scale, 20))

        h_crop, w_crop = int(round(h * y_scale)), int(round(w * x_scale))

        # rewrite the sacles
        x_scale = w/w_crop
        y_scale = h/h_crop

        max_y_shift = h - h_crop
        max_x_shift = w - w_crop

        x1 = np.random.choice(list(range(0, max_x_shift)))
        y1 = np.random.choice(list(range(0, max_y_shift)))
        x2 = x1 + w_crop
        y2 = y1 + h_crop

        im_info[frame_id-1, :] = np.array([frame_id, x1, y1, x2, y2, x_scale, y_scale])

        output_im = im[y1:y2, x1:x2, :].copy()
        output_im = cv2.resize(output_im, None, None, fx=x_scale, fy=y_scale, interpolation=cv2.INTER_LINEAR)
        oh, ow, oc = output_im.shape[0], output_im.shape[1], output_im.shape[2]
        if oh != h or ow != w:
            raise RuntimeError('the output image has a different size ({},{},{}) with the input image ({},{},{})'
                               .format(oh, ow, oc, h, w, c))

        out_im_path = os.path.join(out_dir, str(frame_id).zfill(6) + '.' + extent)
        all_out_im_path.append(out_im_path)
        cv2.imwrite(out_im_path, output_im)

    # save im_info
    im_info_path = os.path.join(out_dir, 'im_info.txt')
    fmt = ['%d', '%d', '%d', '%d', '%d', '%.8f', '%.8f']
    np.savetxt(im_info_path, im_info, fmt=fmt, delimiter=',')

    # generate a sh file
    in_path = 'in_path=' + out_dir + '/%6d.' + extent
    out_path = 'out_path=' + out_dir + '/' + im_name + '.mp4'
    cmd = 'ffmpeg -i ${in_path} -c:v mpeg4 -f rawvideo -y ${out_path}'
    sh_file = in_path + '\n' + out_path + '\n' + cmd

    sh_file_path = os.path.join(out_dir, 'images2video.sh')
    f = open(sh_file_path, 'w')
    f.write(sh_file)
    f.close()

    # run the sh file to generate the video
    os.system('sh ' + sh_file_path)

    # clean up
    if clean_up:
        os.remove(sh_file_path)
        for im in all_out_im_path:
            os.remove(im)


cityscapes_path = '/data0/liuqk/Cityscapes/citysacpesdataset/'
type = 'leftImg8bit' # the type of images

split = ['train', 'val', 'test']
extent = 'png'

for one_split in split:
    one_split_path = os.path.join(cityscapes_path, type, one_split)
    cities = os.listdir(one_split_path)

    for city in cities:
        city_dir = os.path.join(one_split_path, city)

        images = os.listdir(city_dir)
        filter_images = []
        # filter the .png images
        for im in images:
            im_split = im.split('.')

            # make sure this file is an image
            if im_split[-1] == extent:
                filter_images.append(im)

        for i in range(len(filter_images)):
            im = filter_images[i]
            im_name = im.split('.')[0]
            im_path = os.path.join(city_dir, im)

            print('processing ' + im_path)

            # we make a directory for converted raw video
            out_dir = os.path.join(city_dir, im_name)
            expand_image_to_short_video(im_path=im_path, out_dir=out_dir, clean_up=False)

print('Done.')














































