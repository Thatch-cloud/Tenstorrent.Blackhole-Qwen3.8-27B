"""Five learned device layers from projected context; feature projection and proposal selection remain external."""

from dspark_layer import PHASES, SPECIFICATIONS, norm, pack_weight
from dspark_layer_mesh import INPUTS, execute as execute_layer


PARAMETERS = tuple(f'layers.{layer}.{name}' for layer in range(5) for name in SPECIFICATIONS)+('norm.weight',)


def pack_parameter(name, value):
    import torch

    if name not in PARAMETERS:
        raise ValueError('Declared complete five-layer backbone parameter required')
    if name!='norm.weight':
        return pack_weight(name.split('.',2)[2],value)
    if (not isinstance(value,torch.Tensor) or tuple(value.shape)!=(5120,) or value.dtype!=torch.bfloat16
            or value.device.type!='cpu' or not torch.isfinite(value).all()):
        raise ValueError('Complete finite BF16 final normalization parameter required')
    return value.reshape(1,1,1,5120).clone(),False


def flatten(result):
    if set(result)!= {'layers','final_norm'} or len(result['layers'])!=5:
        raise ValueError('All five learned layers and final normalization required')
    values = {}
    for layer,output in enumerate(result['layers']):
        if set(output)!=set(PHASES) or any(set(output[phase])!=set(stages) for phase,stages in PHASES.items()):
            raise ValueError('Every complete learned stage required in each layer')
        values.update({(layer,phase,stage):output[phase][stage] for phase,stages in PHASES.items() for stage in stages})
    values[-1,'final','final_norm'] = result['final_norm']
    return values


def execute(operations, mesh, collectives, inputs, layer_weights, final_norm, retain, *, mask_validated=False):
    if (set(inputs)!=set(INPUTS) or mask_validated is not True or not callable(retain)
            or not isinstance(layer_weights,(tuple,list)) or len(layer_weights)!=5
            or any(set(weights)!=set(SPECIFICATIONS) for weights in layer_weights)):
        raise ValueError('Five complete learned layers, complete inputs, prevalidated mask and tensor ownership required')
    if list(mesh.shape)!=[1,2]:
        raise ValueError('Explicit two-chip mesh required')
    current = dict(inputs)
    layers = []
    for weights in layer_weights:
        result = execute_layer(operations,mesh,collectives,current,weights,retain,mask_validated=True)
        layers.append(result)
        current = dict(current,noise=result['finish']['output'])
    normalized = norm(operations,current['noise'],final_norm,retain)
    return dict(layers=tuple(layers),final_norm=normalized)
