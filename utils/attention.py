from diffusers.models import UNet2DConditionModel


def prepare_unet(unet: UNet2DConditionModel):
    for name, params in unet.named_parameters():
        if 'attn1' in name: # self-attention
            pass
        elif 'attn2' in name: # cross-attention
            pass

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "Attention":
            pass

    return unet