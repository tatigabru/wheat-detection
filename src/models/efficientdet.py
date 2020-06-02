from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain,DetBenchEval
from effdet.efficientdet import HeadNet


def get_model(eval_net, config = get_efficientdet_config('tf_efficientdet_d5'), num_classes: int = 1):
    
    model = EfficientDet(config, pretrained_backbone=False)    
    model.class_net = HeadNet(config, num_outputs=num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    # checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(eval_net.model.state_dict())

    model = DetBenchTrain(net, config)

    return model
    net = net.train()
    return net.cuda()


def get_test_net(best_weigth, config = get_efficientdet_config('tf_efficientdet_d5'), num_classes: int = 1):
    
    model = EfficientDet(config, pretrained_backbone=False)
    model.class_net = HeadNet(config, num_outputs=num_classes, norm_kwargs=dict(eps=.001, momentum=.01))

    # checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(best_weigth)

    net = DetBenchEval(net, config)
    net = net.train()
    return net.cuda()