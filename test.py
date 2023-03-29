
import os
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results
import glob
import numpy as np
import torch
import os
import cv2
from model.unet_model import UNet


def cal_miou(test_dir="D:/coda/AM-Net/skin/Test_Images",
             pred_dir="D:/coda/AM-Net/skin/results", gt_dir="D:/coda/AM-Net/skin/Test_Labels"):
   
    miou_mode = 0
   
    num_classes = 2
   
    name_classes = ["background", "nidus"]
   

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       
        net = UNet(n_channels=1, n_classes=1)
       
        net.to(device=device)
   
        net.load_state_dict(torch.load('best_model_skin.pth', map_location=device)) # todo

        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape
    
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (512, 512))
           
            img = img.reshape(1, 1, img.shape[0], img.shape[1])

            img_tensor = torch.from_numpy(img)

            img_tensor = img_tensor.to(device=device, dtype=torch.float32)

            pred = net(img_tensor)

            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        print(gt_dir)
        print(pred_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  
        print("Get miou done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

if __name__ == '__main__':
    cal_miou()
