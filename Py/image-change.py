from PIL import Image


def suoxiao(im):
    w, h = im.size
    print('Original image size: %sx%s' % (w, h))
    im2 = im.resize((w //2 , h//2))
    # im2.show()
    im2.save('suoxiao.jpg', 'jpeg')

def fangda(im):
    w, h = im.size
    im2 = im.resize((w * 2, h * 2),Image.ANTIALIAS)
    # im2.show()
    im2.save('fangda.jpg', 'jpeg')


def shuipingjingxiang(im):
    w, h = im.size
    im2 = im.transform((w, h), Image.EXTENT, (0, h, w, 0))
    # im2.show()
    im2.save('shuipingjingxiang.jpg', 'jpeg')

def chuizhijingxiang(im):
    w, h = im.size
    im2 = im.transform((w, h), Image.EXTENT, (w, 0, 0, h))
    # im2.show()
    im2.save('chuizhijingxiang.jpg', 'jpeg')


def zhongxinjingxiang(im):
    w, h = im.size
    im2 = im.transform((w, h), Image.EXTENT, (w, h, 0, 0))
    # im2.show()
    im2.save('zhongxinjingxiang.jpg', 'jpeg')

def duquRGB(im):
    im2 = im.split()
    im2[0].show()
    im2[1].show()
    im2[2].show()

    w, h = im.size
    rCount = 0
    gCount = 0
    bCount = 0
    for x in range(w):
        for y in range(h):
            r, g, b = im.getpixel((x,y))
            rCount += r
            gCount += g
            bCount += b
    print('R的和为:%s\nG的和为:%s\nB的和为:%s\n' % (rCount, gCount, bCount))

imagepath = input("请输入图片地址：")
im = Image.open(imagepath)
# im = Image.open('1.jpeg')
shuipingjingxiang(im)
chuizhijingxiang(im)
zhongxinjingxiang(im)
fangda(im)
suoxiao(im)
duquRGB(im)
