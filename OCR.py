import cv2
from scipy import stats as st
import numpy as np
import pytesseract


def Erosion(Mask:np.array) -> np.array:
    # Only for old version.
    # TODO: replace by C-code for performance.
    Result = np.copy(Mask)
    # 4 corners
    Result[0, 0] = (Mask[0, 0] & Mask[0, 1] & Mask[1, 0])
    Result[0, -1] = (Mask[0, -1] & Mask[0, -2] & Mask[1, -1])
    Result[-1, 0] = (Mask[-1, 0] & Mask[-1, 1] & Mask[-2, 0])
    Result[-1, -1]= (Mask[-1, -1] & Mask[-1, -2] & Mask[-2, -1])

    # 4 edges
    for m in range(Mask.shape[0])[1:-1]:
        Result[m, 0] = (Mask[m, 0] & Mask[m, 1] & Mask[m - 1, 0] & Mask[m + 1, 0])
        Result[m, -1] = (Mask[m, -1] & Mask[m, -2] & Mask[m - 1, -1] & Mask[m + 1, -1])
    for n in range(Mask.shape[1])[1:-1]:
        Result[0, n] = (Mask[0, n] & Mask[1, n] & Mask[0, n - 1] & Mask[0, n + 1])
        Result[-1, n] = (Mask[-1, n] & Mask[-2, n] & Mask[-1, n - 1] & Mask[-1, n + 1])

    # Inside indices
    for i in range(Mask.shape[0])[1:-1]:
        for j in range(Mask.shape[1])[1:-1]:
            if Mask[i, j]:
                continue
            Result[i, j] = (Mask[i-1, j] & Mask[i+1, j] & Mask[i, j-1] & Mask[i, j+1])

    return Result

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
        
def DiffOCR(TextImage:np.array, Background:np.array, Lang="jpn", Threshold=4) -> str:
    Diff = np.clip(TextImage - Background, 0, 255)
    Mask = (Diff != 0)
    Mask = (Mask[:, :, 0] | Mask[:, :, 1] | Mask[:, :, 2])

    TextColor = PixelMode(TextImage[Mask])
    SelectText = lambda m : m[:, 0] & m[:, 1] & m[:, 2]
    Mask[Mask] = SelectText(np.abs(TextImage[Mask] - TextColor) < Threshold)

    Matting = np.zeros_like(TextImage)
    Matting[Mask] = TextImage[Mask]

    text = pytesseract.image_to_string(Matting, lang=Lang)
    return text

if __name__ == "__main__":
    TextImage = cv2.imread("Images/kanna1.png")
    Background = cv2.imread("Images/kanna2.png")

    # Diff = np.clip(TextImage - Background, 0, 255)
    # Mask = (Diff != 0)
    # Mask = (Mask[:, :, 0] | Mask[:, :, 1] | Mask[:, :, 2])


    # TextColor = PixelMode(TextImage[Mask])
    # SelectText = lambda m : m[:, 0] & m[:, 1] & m[:, 2]
    # Mask[Mask] = SelectText(np.abs(TextImage[Mask] - TextColor) < 4)


    # Matting = np.zeros_like(TextImage)
    # Matting[Mask] = TextImage[Mask]

    # cv2.imwrite("text.png", Matting)

    text = DiffOCR(TextImage, Background)
    print(text)
