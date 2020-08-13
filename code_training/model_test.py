import tensorflow as tf
import numpy as np
import cv2

import sys
import os
from time import *
import input_stream
sys.path.append(os.path.join(os.getcwd(), "code_commons"))
from global_constants import *
from auxiliary_ftns import *
import ntpath


def run_test(sess, cdnet, FLAGS, mode='webcam', frame_size=None, srcname=None,
             wait_time=1, save_video=False, out_video_file_name=None, fps=24.0,
             ckpt_basename=None, evaluation=True):
    if evaluation is True:
        dirpath = os.path.split(srcname)[0]

    if save_video is True:
        assert ckpt_basename is not None

    font = cv2.FONT_HERSHEY_SIMPLEX

    if mode == 'webcam':
        frame = input_stream.Frame('webcam')
        if frame_size is not None:
            frame.set_size(frame_size)
        else:
            frame.set_size((640, 480))
    elif mode == 'folder':
        assert srcname is not None
        frame = input_stream.Frame('folder', srcname)
        if frame_size is not None:
            frame.set_size(frame_size)
    elif mode == 'video':
        assert srcname is not None
        frame = input_stream.Frame('video', srcname)
        if frame_size is not None:
            frame.set_size(frame_size)
    else:
        raise Exception('Wrong Test Mode')

    if save_video is True:
        if out_video_file_name is None:
            filename = strftime("auto_generated_name_%d%b%Y_%H_%M", gmtime()) + ckpt_basename + ".avi"
            #filename = strftime("auto_generated_name_%d%b%Y_%H_%M", gmtime() ) + ckpt_basename + ".mp4"
        else:
            filename = out_video_file_name
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(filename, fourcc, fps, (IMAGE_WIDTH, IMAGE_HEIGHT * 3))

    total = 0
    hit = 0

    while True:
        save_result_image = True
        tic = time()
        image, filename = frame.get_frame()

        if image is None:
            break

        network_input = cv2.resize(image[:, :, ::-1], (IMAGE_WIDTH, IMAGE_HEIGHT))
        src = network_input.astype(np.float32) / 255.0

        network_input = (src.copy() - 0.5) * np.sqrt(2.0)
        gray = cv2.cvtColor(network_input, cv2.COLOR_BGR2GRAY)
        network_input = np.array(network_input).reshape([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edge_map = cv2.Canny(gray.astype(np.uint8), 20, 100)
        edge_map = edge_map.reshape([1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
        edge_map = (edge_map / 255.0 - 0.5) * np.sqrt(2.0)
        network_input = np.concatenate((network_input, edge_map), axis=-1)

        _output, _mask = sess.run([cdnet.output, cdnet.output_mask], feed_dict={cdnet.input: network_input})

        _output = _output[0]
        _mask = _mask[0]

        total = total + 1

        _mask = np.expand_dims(_mask, axis=-1)
        _mask = np.tile(_mask, [1, 1, 3])

        overlap = src.copy() * 0.5
        overlap[:, :, 0] = overlap[:, :, 0] + _mask[:, :, 0]

        rst = np.concatenate([src, overlap, _mask], axis=0)

        tmp = rst[:, :, ::-1]
        tmp = np.clip(tmp, 0, 1)
        tmp = (tmp * 255).astype(np.uint8)

        save_img = np.concatenate([src, overlap], axis=1)
        save_img = save_img[:, :, ::-1]
        save_img = np.clip(save_img, 0, 1)
        save_img = (save_img * 255).astype(np.uint8)

        if save_result_image:
            folder_name, image_name = filename.split('/')[-2:]
            if not os.path.exists(os.getcwd() + '/test_results/' + folder_name):
                os.makedirs(os.getcwd() + '/test_results/' + folder_name)
            img_save_dir = os.getcwd() + '/test_results/' + folder_name + '/' + image_name
            print(img_save_dir)
            cv2.imwrite(img_save_dir, save_img)

        if save_video is True:
            #cv2.imshow("wndx", tmp )
            out.write(tmp)

    if save_video is True:
        out.release()


if __name__ == "__main__":
    points = np.array([[10, 1], [1, 10], [10, 10]])
    print(convex_hull_area(points))
