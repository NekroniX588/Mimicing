
# ssds part
import torchvision
import torch.nn as nn
from lib.modeling.ssds import ssd
from lib.modeling.ssds import ssd_lite
from lib.modeling.ssds import ssd_lite_MobileNetV2
from lib.modeling.ssds import ssd_lite_MobileNetV3
from lib.modeling.ssds import ssd_ResNet18
from lib.modeling.ssds import rfb
from lib.modeling.ssds import rfb_lite
from lib.modeling.ssds import fssd
from lib.modeling.ssds import fssd_lite
from lib.modeling.ssds import yolo

ssds_map = {
                'ssd': ssd.build_ssd,
                'ssd_lite': ssd_lite.build_ssd_lite,
                'rfb': rfb.build_rfb,
                'rfb_lite': rfb_lite.build_rfb_lite,
                'fssd': fssd.build_fssd,
                'fssd_lite': fssd_lite.build_fssd_lite,
                'yolo_v2': yolo.build_yolo_v2,
                'yolo_v3': yolo.build_yolo_v3,
            }

# nets part
from lib.modeling.nets import vgg
from lib.modeling.nets import resnet
from lib.modeling.nets import mobilenet
from lib.modeling.nets import darknet
from lib.modeling.nets import ShuffleNetV2
from lib.modeling.nets.mobilenetv3_old import MobileNetV3_Large
networks_map = {
                    'shufflenet_v1': ShuffleNetV2.shufflenet_v1,
                    'vgg16': vgg.vgg16,
                    'resnet_18': resnet.resnet_18,
                    'resnet_34': resnet.resnet_34,
                    'resnet_50': resnet.resnet_50,
                    'resnet_101': resnet.resnet_101,
                    'mobilenet_v1': mobilenet.mobilenet_v1,
                    'mobilenet_v1_075': mobilenet.mobilenet_v1_075,
                    'mobilenet_v1_050': mobilenet.mobilenet_v1_050,
                    'mobilenet_v1_025': mobilenet.mobilenet_v1_025,
                    'mobilenet_v2': torchvision.models.mobilenet_v2(True).features[:18],
                    'mobilenet_v2_075': mobilenet.mobilenet_v2_075,
                    'mobilenet_v2_050': mobilenet.mobilenet_v2_050,
                    'mobilenet_v2_025': mobilenet.mobilenet_v2_025,
                    'darknet_19': darknet.darknet_19,
                    'darknet_53': darknet.darknet_53,

               }

from lib.layers.functions.prior_box import PriorBox
import torch

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def _forward_features_size(model, img_size):
    model.eval()
    x = torch.rand(1, 3, img_size[0], img_size[1])
    x = torch.autograd.Variable(x, volatile=True) #.cuda()
    feature_maps = model(x, phase='feature')
    return [(o.size()[2], o.size()[3], o.size()[1]) for o in feature_maps]


def create_model(cfg):
    '''
    '''
    #
    if cfg.NETS == 'mobilenet_v3':
        base = MobileNetV3_Large()
        model = torch.load("./Student/mbv3_large.old.pth.tar", map_location='cpu')
        weight = model["state_dict"]
        weights_norm = {i[7:]:weight[i] for i in weight}
        base.load_state_dict(weights_norm)

    elif cfg.NETS == 'mobilenet_v2':
        base = torchvision.models.mobilenet_v2(True).features[:18]

    elif cfg.NETS == 'resnet_18':
        base = torchvision.models.resnet18(True)
        base.fc = Identity()
    else:
        base = networks_map[cfg.NETS]

    number_box= [2*len(aspect_ratios) if isinstance(aspect_ratios[0], int) else len(aspect_ratios) for aspect_ratios in cfg.ASPECT_RATIOS]  
    
    if cfg.NETS == 'mobilenet_v3':
        model = ssd_lite_MobileNetV3.build_ssd_lite(base=base, feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES)
    elif cfg.NETS == 'mobilenet_v2':
        model = ssd_lite_MobileNetV2.build_ssd_lite(base=base, feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES)
    elif cfg.NETS == 'resnet_18':    
        model = ssd_ResNet18.build_ssd(base=base, feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES)
    else:
        model = ssds_map[cfg.SSDS](base=base, feature_layer=cfg.FEATURE_LAYER, mbox=number_box, num_classes=cfg.NUM_CLASSES)
    #   
    feature_maps = _forward_features_size(model, cfg.IMAGE_SIZE)
    print('==>Feature map size:')
    print(feature_maps)
    # 
    if cfg.PRIOR:
        priorbox = PriorBox(image_size=cfg.IMAGE_SIZE, feature_maps=feature_maps, aspect_ratios=cfg.ASPECT_RATIOS, 
                    scale=cfg.SIZES, archor_stride=cfg.STEPS, clip=cfg.CLIP)
        return model, priorbox, feature_maps
    else:
        return model