def get_effdet_pretrain_names(alias: str = 'effdet4') -> str:
    """Returns pretrains names for differne efficient dets"""
    pretrains = { 
        'effdet0': 'efficientdet_d0-d92fd44f.pth',
        'effdet1': 'efficientdet_d1-4c7ebaf2.pth',
        'effdet2': 'efficientdet_d2-cb4ce77d.pth',
        'effdet3': 'efficientdet_d3-b0ea2cbc.pth',
        'effdet4': 'efficientdet_d4-5b370b7a.pth',
        'effdet5': 'efficientdet_d5-ef44aea8.pth',
        'effdet6': 'efficientdet_d6-51cb0132.pth',
        'effdet7': 'efficientdet_d7-f05bf714.pth',
    }
    return pretrains[alias] 