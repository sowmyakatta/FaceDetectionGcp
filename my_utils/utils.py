import base64

def encodeImage(imagePath):
    with open(imagePath, "rb") as f:
        return base64.b64encode(f.read())


def decodeImage(imageString,fileName):
    imageData = base64.b64decode(imageString)
    with open(fileName, "wb") as f:
        f.write(imageData)
        f.close()
