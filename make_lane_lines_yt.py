import os
import argparse
import cv2
import torch

from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *

net = SCNN(input_size=(800, 288), pretrained=False)
mean=(0.3598, 0.3653, 0.3662) # CULane mean, std
std=(0.2573, 0.2663, 0.2756)
transform_img = Resize((800, 288))
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/demo.jpg", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    #
    # determine which computer/platform we are running on
    if (os.name == "posix"):
        os_list = os.uname()
        if (os_list[0] == "Darwin"):
            pf_detected = 'MAC'
        elif (os_list[0] == "Linux"):
            if (os_list[1] == 'en4119351l'):
                pf_detected = 'Quadro'
            elif (os_list[1] == '19fef43c2174'):
                pf_detected = 'Exxact'
            elif (os_list[1] == 'EN4113948L'):
                pf_detected = 'Kevin'
    else:
        pf_detected = 'PC'

    # set the root path based on the computer/platform
    #   rootPath is path to directory in which webots/ and imdata/ directories reside
    if (pf_detected == 'MAC'):
        rootPath = '/Users/mes/Documents/ASU-Classes/Research/Ben-Amor/code/'

    elif (pf_detected == 'Quadro'):
        rootPath = '/home/local/ASUAD/mestric1/Documents/AVCES/'

    elif (pf_detected == 'Exxact'):
        rootPath = '/home/dockeruser/Documents/AVCES/'

    elif (pf_detected == 'Kevin'):
        rootPath = '/home/local/ASUAD/mestric1/Documents/AVCES/'

    elif (pf_detected == 'PC'):
        # rootPath = 'C:\Users\cesar\Desktop\Furi\'
        print("PC platform detected.  Exiting.")
        exit()
    else:
        print("Computer/Platform not detected.  Exiting.")
        exit()
    #
    # args = parse_args()
    #  CCT007-Scene-005 has 153 frames
    #  CCT007-Scene-009 has 218 frames
    for fno in range(1, 219):
        img_path = rootPath + "imdata/video/processed/CCT007/CCT007-Scene-009/Run/rgb{0:06d}.png".format(fno)
        weight_path = "experiments/exp10/exp10_best.pth"

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transform_img({'img': img})['img']
        x = transform_to_net({'img': img})['img']
        x.unsqueeze_(0)

        save_dict = torch.load(weight_path, map_location='cpu')
        net.load_state_dict(save_dict['net'])
        net.eval()

        seg_pred, exist_pred = net(x)[:2]
        seg_pred = seg_pred.detach().cpu().numpy()
        exist_pred = exist_pred.detach().cpu().numpy()
        seg_pred = seg_pred[0]
        exist = [1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)]

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lane_img = np.zeros_like(img)
        color = np.array([[255, 125, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255]], dtype='uint8')
        coord_mask = np.argmax(seg_pred, axis=0)
        for i in range(0, 4):
            if exist_pred[0, i] > 0.5:
                lane_img[coord_mask == (i + 1)] = color[i]
        img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)
        # cv2.imwrite("demo/demo_result.jpg", img)

        # print(" ")
        # print("Lane Lines:")
        for x in getLane.prob2lines_CULane(seg_pred, exist):
            # print(x)
            img = cv2.polylines(img, np.int32([np.array(x)]), 0, (0, 0, 255), thickness=3)
        #
        img_rsz = cv2.resize(img, (1280, 720), interpolation = cv2.INTER_LANCZOS4)
        cv2.imwrite(rootPath + "imdata/video/processed/CCT007/CCT007-Scene-009/Lane/lin{0:06d}.png".format(fno), img_rsz)
        #
        print("frame {} is complete.".format(fno))
        #

    # print(" ")
    # print("exist: ", exist)
    # print("exist_pred: ", exist_pred)
    # if args.visualize:
    #     print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)])
    #     cv2.imshow("", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
