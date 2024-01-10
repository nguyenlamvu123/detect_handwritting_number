#site: https://github.com/pbcquoc/vietocr/blob/master/vietocr_gettingstart.ipynb
import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


config = Cfg.load_config_from_name('vgg_transformer')
# config['weights'] = './weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cpu'


detector = Predictor(config)

for img in (
        'photo_6118490984477211115_y.jpg',
        'photo_6118490984477211116_y.jpg',
        'photo_6118490984477211117_y.jpg',
        'photo_6118490984477211118_y.jpg',
        'photo_6118490984477211119_y.jpg',
        'photo_6118490984477211120_y.jpg',
        'photo_6118490984477211121_y.jpg',
        "digit.jpg",
):
    print(img)
    img = Image.open(img)
    plt.imshow(img)
    s = detector.predict(img)
    print(s)
