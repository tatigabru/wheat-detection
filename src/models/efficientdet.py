from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain,DetBenchEval
from effdet.efficientdet import HeadNet


def get_effnet(eval_net, config = get_efficientdet_config('tf_efficientdet_d5'), num_classes: int = 1, device: 'cuda:0'):
    
    model = EfficientDet(config, pretrained_backbone=False)    
    model.class_net = HeadNet(config, num_outputs=num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
    model = DetBenchTrain(net, config)
    model = model.train()

    return model.to_device(device)     
   


def load_effnet(checkpoint_path, config = get_efficientdet_config('tf_efficientdet_d5'), num_classes: int = 1):
    
    model = EfficientDet(config, pretrained_backbone=False)
    model.class_net = HeadNet(config, num_outputs=num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)

    net = DetBenchEval(net, config)
    net = net.train()
    return net.cuda()



def get_effnet(checkpoint_path):
    config = get_efficientdet_config('tf_efficientdet_d5')
    net = EfficientDet(config, pretrained_backbone=False)

    config.num_classes = 1
    config.image_size = 512
    net.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint)

    del checkpoint
    gc.collect()
    net = DetBenchEval(net, config)
    net.eval()

    return net.cuda()    