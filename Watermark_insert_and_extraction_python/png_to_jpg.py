# Transform png to jpg
from PIL import Image


def IsValidImage(img_path):
    """
    Detect the effectiveness of picture
    :param img_path: path_of_image
    """
    bValid = True
    try:
        Image.open(img_path).verify()
    except:
        bValid = False
    return bValid


def transimg(img_path):
    '''
    Transform the format of pictures
    :param img_path: path_of_image
    '''
    if IsValidImage(img_path):
        try:
            str = img_path.rsplit(".", 1)
            output_img_path = str[0] + ".jpg"
            im = Image.open(img_path)
            im.save(output_img_path)
            return True
        except:
            return False
    else:
        return False

if __name__ == '__main__':
    # REPLACE IT!
    path = your_own_png_path
    transimg(path)

