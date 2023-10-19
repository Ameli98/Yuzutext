import cv2
import numpy as np
import pytesseract
from Global import *


def PixelMode(Pixels:np.array) -> np.array:
    ColorDict = {}
    Encode = lambda p : p[0] * (256 ** 2) + p[1] * 256 + p[2]
    def Decode(k:int) -> np.array:
        FirstDigit, k = k // (256 ** 2), k % (256 ** 2)
        SecondDigit, ThirdDigit = k // 256, k % 256
        return np.array([FirstDigit, SecondDigit, ThirdDigit])

    for Pixel in Pixels:
        key = Encode(Pixel)
        if not ColorDict.get(key):
            ColorDict[key] = 1
        else:
            ColorDict[key] += 1

    ModeKey = list(ColorDict.keys())[np.array(list(ColorDict.values())).argmax()]
    PixelMode = Decode(ModeKey)
    return PixelMode
        
def DiffOCR(TextImage:np.array, Background:np.array, Lang="jpn", Threshold=Threshold) -> str:
    Diff = np.clip(TextImage - Background, 0, 255)
    Mask = (Diff != 0)
    Mask = (Mask[:, :, 0] | Mask[:, :, 1] | Mask[:, :, 2])

    TextMask = np.full_like(Mask, False)
    TextMask[TextArea[0][0] : TextArea[1][0], TextArea[0][1] : TextArea[1][1]] = True
    Mask = Mask & TextMask

    TextColor = PixelMode(TextImage[Mask])
    
    SelectText = lambda m : m[:, 0] & m[:, 1] & m[:, 2]
    Mask[Mask] = SelectText(np.abs(TextImage[Mask] - TextColor) < Threshold)

    Matting = np.zeros_like(TextImage)
    Matting[Mask] = TextImage[Mask]

    text = pytesseract.image_to_string(Matting, lang=Lang)
    TextImage[~TextMask] = 0
    return text

if __name__ == "__main__":
    TextImage = cv2.imread("Images/kurumi1.png")
    Background = cv2.imread("Images/kurumi2.png")

    text = DiffOCR(TextImage, Background)
    print(text)
