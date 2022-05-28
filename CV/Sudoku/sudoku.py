import cv2
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
from mnist import MNIST
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torch.nn.functional as F


SHIFT = 36

def color(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=3,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(
                in_channels=32,              
                out_channels=64,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2), 
            nn.Flatten()
        )
        self.out = nn.Linear(64 * 3 * 3, 10)
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.out(output)
        return output  
    
def get_light_mask(img):
    HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    HUE = HLS[:, :, 0]             
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]
    mask = (LIGHT < 100) | (LIGHT > 250)
    return mask


def get_sudoku_mask(mask):
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea)
    
    max_area = cv2.contourArea(contours[-1])
    i = 2
    while True:
        if cv2.contourArea(contours[-i]) > 0.7 * max_area:
            i += 1
        else:
            break
    img_r = np.zeros_like(mask).astype(np.uint8)
    approxed = []
    for contour in contours[-i+1:]:
        perimeter = cv2.arcLength(contour, True) 
        eps = 0.05 * perimeter
        approx = cv2.approxPolyDP(contour, eps, True)
        cv2.drawContours(img_r, [approx], -1, (255,0,0), -1)
        approxed.append(approx)
    

    return approxed, np.uint8(img_r)

def get_perspective(approxed, img):
    imgs = []
    warp_matrices = []
    size = SHIFT
    width = size * 9
    height = size * 9
    for approx in approxed:
        approx = approx.squeeze(1)
        lt = min(approx, key = lambda x : sum(x))
        lb = max(approx, key = lambda x : x[1] - x[0])
        rb = max(approx, key = lambda x : sum(x))
        rt = max(approx, key = lambda x : x[0] - x[1])
        
        
        in_points = np.float32([lt, lb, rb, rt])
        out_points = np.float32([[0,0], [0,height-1], [width-1,height-1], [width-1,0]])
        # compute perspective matrix
        warp_matrix = cv2.getPerspectiveTransform(in_points,out_points)
        imgOutput = cv2.warpPerspective(img, warp_matrix, (width,height), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
        imgs.append(imgOutput)
        warp_matrices.append(warp_matrix)
    return imgs, warp_matrices


def get_sudoku_matrices(imgs, model, device):
    shift = SHIFT
    matrices = []
    cells = []
    for imgOutput in imgs:
        matrix = np.zeros((9, 9))

        for i in range(9):
            for j in range(9):
                lt = np.array([i * shift, j * shift])
                cell = (imgOutput[lt[0]+ 4 : lt[0] + shift - 4, lt[1] + 4 : lt[1] + shift - 4])


                cell = cv2.GaussianBlur(cell,(3, 3),0)
                # cell = cell[..., ::-1]

                HLS = cv2.cvtColor(cell, cv2.COLOR_BGR2HLS)
                LIGHT = HLS[:, :, 1]
                LIGHT = 255. - LIGHT
                LIGHT[LIGHT<150] = 0
                LIGHT = LIGHT / np.maximum(np.max(LIGHT), 1) * 255.
                cell = LIGHT.copy()
                
                cells.append(cell)
                
                if (LIGHT[10 : 18, 10 : 18]>0).sum() < 5:
        #             print('empty')
                    out = -1
                else:
                    cell_contours, hierarchy = cv2.findContours(LIGHT.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in cell_contours: 
                        all_inside = False
                        for w in range(8, 20):
                            for h in range(8, 20):
                                dist = cv2.pointPolygonTest(contour, (w, h), False)
                                if dist == 1:
                                    all_inside = True
                                    break
                        if not all_inside:
                            cv2.drawContours(cell, [contour], -1, (0,0,0), -1)

                    tensor_cell = torch.tensor(cell).view(1, 1, *cell.shape).float().to(device)
                    out= model(tensor_cell)
                    out = nn.Softmax()(out)
                    out = torch.argmax(out, dim=1).detach().cpu().numpy()

                matrix[i][j] = out
        matrices.append(matrix)
    return matrices, cells





def predict_image(img):
    
    device = 'cpu'
    model = CNN().to(device)
    model.load_state_dict(torch.load('/autograder/submission/model.pt'))
    model.eval()
    
    light_mask = get_light_mask(img)
    approxed, mask = get_sudoku_mask(light_mask)
    imgs, warp_matrices = get_perspective(approxed, img)
    matrices, cells = get_sudoku_matrices(imgs, model, device)
    return mask, matrices
