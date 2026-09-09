"""Five learned device layers from projected context; feature projection and proposal selection remain external."""

from dspark_layer import SPECIFICATIONS, norm
from dspark_layer_mesh import INPUTS, execute as execute_layer


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
