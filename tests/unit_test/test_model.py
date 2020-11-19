
import torch
import encoding

def test_model_inference():
    x = torch.rand(1, 3, 224, 224)
    for model_name in encoding.models.pretrained_model_list():
        print('Doing: ', model_name)
        if 'wideresnet' in model_name: continue # need multi-gpu
        model = encoding.models.get_model(model_name, pretrained=True)
        model.eval()
        y = model(x)

if __name__ == "__main__":
    import nose
    nose.runmodule()
