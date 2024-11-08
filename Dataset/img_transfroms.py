import cv2

class Resize:
    def __init__(self, img_h, img_w):
        self.img_h, self.img_w = img_h, img_w

    def __call__(self, img):
        img = cv2.resize(img, (self.img_w, self.img_h), interpolation = cv2.INTER_LINEAR)
        return img