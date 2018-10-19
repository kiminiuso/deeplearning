from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=10):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    size = random.randint(4, 20)
    image = ImageCaptcha(height=128, width=(size * 40 + 40), font_sizes=(80, 90, 100, 112))
    captcha_text = random_captcha_text(captcha_size=size)
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    w, h = captcha_image.size
    captcha_image = captcha_image.resize((w // 4, h // 4))

    captcha_image.save('train_data/' + captcha_text + '.jpg', 'jpeg')  # 写到文件

    # 调试用显示
    # captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


if __name__ == '__main__':
    # 测试
    f = plt.figure()

    for i in range(0,2501):
        text, image = gen_captcha_text_and_image()

        # ax = f.add_subplot(111)
        # ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
        # plt.axis('off')
        # plt.imshow(image)
        # plt.savefig('data/' + text + '.jpg')
        # plt.show()
