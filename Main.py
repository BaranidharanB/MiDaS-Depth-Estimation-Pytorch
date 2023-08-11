import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np

# Downloading the MiDaS model

MiDaS = torch.hub.load('intel-isl/MiDas','MiDaS_small')
Device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MiDaS.to(Device)
MiDaS.eval()

# Transformation of Captured video to Monocular Depth
TransForms = torch.hub.load('intel-isl/MiDas','transforms')
TransForm = TransForms.small_transform

# Integrating with capturing video through OpenCV

Image = cv2.VideoCapture(0)
while Image.isOpened():
    ret, frame = Image.read()
    rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    rgbImageBatch = TransForm(rgbImage).to(Device)

    with torch.no_grad():
        Prediction = MiDaS(rgbImageBatch)
        Prediction = torch.nn.functional.interpolate(Prediction.unsqueeze(1),
                                                     size=rgbImage.shape[:2],
                                                     mode="bicubic",
                                                     align_corners=False).squeeze()

        Output = Prediction.cpu().numpy()
        Output = cv2.normalize(Output,None,0,1,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_64F)
        #print(Output)
        Output = (Output*255).astype(np.uint8)
        Output = cv2.applyColorMap(Output,cv2.COLORMAP_MAGMA)
        plt.imshow(Output)
        cv2.imshow("Depth Map",Output)
        plt.pause(0.0001)

    
    if cv2.waitKey(10) & 0xFF ==ord('q'):
        Image.release()
        cv2.destroyAllWindows()

plt.show()