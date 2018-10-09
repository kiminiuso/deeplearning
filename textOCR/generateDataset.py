import numpy as np
import string
from PIL import Image, ImageFont, ImageDraw

# s：图片质量
def MakeImg(t, f, fn, s=(100, 100), o = (16, 8)):

    img = Image.new('RGB', s, "black")
    draw = ImageDraw.Draw(img)
    draw.text(o, t, (255, 255, 255), font=f)
    img.save(fn)


CS = list(string.ascii_letters) + list(string.digits)
RTS = list(np.random.randint(10, 64, size=8192)) + [64]

S = [''.join(np.random.choice(CS, i)) for i in RTS]

font = ImageFont.truetype("LiberationMono-Regular.ttf", 16)

MS = max(font.getsize(Si) for Si in S)

OFS = ((640 - MS[0]) // 2, (32 - MS[1]) // 2)

MS = (640, 32)
Y = []
# 生成图片
for i, Si in enumerate(S):
    MakeImg(Si, font, 'dataset/' + str(i) + '.png', MS, OFS)
    Y.append('dataset/' + str(i) + '.png,' + Si)
# 写入csv
with open('Train.csv', 'w') as F:
    F.write('\n'.join(Y))
