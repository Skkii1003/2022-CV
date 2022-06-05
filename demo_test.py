import argparse
import cv2
import torch
from moviepy.video.io.VideoFileClip import VideoFileClip
import math
from model import SCNN
from utils.prob2lines import getLane
from utils.transforms import *

net = SCNN(input_size=(800, 288), pretrained=False)
mean = (0.3598, 0.3653, 0.3662)  # CULane mean, std
std = (0.2573, 0.2663, 0.2756)
transform_img = Resize((800, 288))
transform_to_net = Compose(ToTensor(), Normalize(mean=mean, std=std))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_path", '-i', type=str, default="demo/sample/image/3.png", help="Path to demo img")
    parser.add_argument("--weight_path", '-w', type=str, default="experiments/vgg_SCNN_DULR_w9/vgg_SCNN_DULR_w9.pth",
                        help="Path to model weights")
    parser.add_argument("--visualize", '-v', action="store_true", default=False, help="Visualize the result")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    img_path = args.img_path
    weight_path = args.weight_path

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
    cv2.imwrite("demo/3_result.jpg", img)

    for x in getLane.prob2lines_CULane(seg_pred, exist):
        print(x)

    if args.visualize:
        print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)])
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process(img):
    args = parse_args()
    weight_path = args.weight_path

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
            # cv2.imshow('i', lane_img)
            # cv2.waitKey(0)
    img = cv2.addWeighted(src1=lane_img, alpha=0.8, src2=img, beta=1., gamma=0.)


    # 改动
    mid_lane = np.zeros_like(img)
    white = (255, 255, 255)
    a = []
    b = []
    points1 = []
    points2 = []
    for x in getLane.prob2lines_CULane(seg_pred, exist):
        length = len(x)
        a = -(x[length - 1][1] - x[0][1]) / (int(x[0][0] - x[length - 1][0])+0.0001)
        b = x[0][1] - a * x[0][0]
        points1.append((int((img.shape[0]/2 - b) / a), img.shape[0]/2))
        points2.append((int((img.shape[0] - b) / a), img.shape[0]))
    mid = img.shape[1] / 2
    k = 0
    for i in range(len(points2)):
        if points2[i][0] > mid:
            mid_point = int((points2[i][0] + points2[k][0]) / 2)
            cv2.line(img, (mid_point, points2[i][1]),
                    (int((points1[i][0] + points1[k][0]) / 2), int(img.shape[0]/1.5)), (255, 255, 0), 3)
            if mid_point > mid + 100:

                cv2.putText(img, 'should right', (int(mid-200), int(img.shape[0]-10)), cv2.FONT_HERSHEY_COMPLEX, 2.0, white, 0.5)
            else:
                cv2.putText(img, 'should left', (int(mid-200), int(img.shape[0]-10)), cv2.FONT_HERSHEY_COMPLEX, 2.0, white, 0.5)
            break
        else:
            k = i
    if args.visualize:
        print('for')
        print([1 if exist_pred[0, i] > 0.5 else 0 for i in range(4)])
        cv2.imshow("", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img


if __name__ == "__main__":
    video = VideoFileClip("demo/sample/video/6.mp4")
    video_out = video.fl_image(lambda clip:process(clip))
    video_out.write_videofile("demo/6fo.mp4",audio=False)
