"""Device-only learned feature projection and complete DSpark backbone; target selector remains separate."""

from dspark_backbone_mesh import PARAMETERS as BACKBONE_PARAMETERS, execute as backbone, flatten as backbone_stages, pack_parameter as pack_backbone
from dspark_intake import TAPS
from dspark_layer_mesh import INPUTS as BACKBONE_INPUTS
from dspark_mesh import gather_partials
from dspark_projection import project, normalize_partials
from feature_projection import projection_shards


PARAMETERS = ('fc.weight','hidden_norm.weight',*BACKBONE_PARAMETERS)
INPUTS = (*('feature_'+str(tap) for tap in TAPS),*(name for name in BACKBONE_INPUTS if name!='context'))


def pack_parameter(name, value):
    import torch

    if name=='fc.weight':
        if tuple(value.shape)!=(5120,25600) or value.dtype!=torch.bfloat16 or not torch.isfinite(value).all():
            raise ValueError('Complete finite BF16 DSpark feature projection required')
        return torch.stack(projection_shards(value)).unsqueeze(1),True
    if name=='hidden_norm.weight':
        return pack_backbone('norm.weight',value)
    return pack_backbone(name,value)


def execute(operations, mesh, collectives, inputs, parameters, layer_weights, retain, *, mask_validated=False):
    if (set(inputs)!=set(INPUTS) or set(parameters)!=set(PARAMETERS) or mask_validated is not True
            or list(mesh.shape)!=[1,2] or not callable(retain)):
        raise ValueError('Complete TP2 learned pipeline inputs, weights, ownership and validated mask required')
    projection = project(operations,{tap:inputs['feature_'+str(tap)] for tap in TAPS},parameters['fc.weight'],retain)
    first,second = gather_partials(operations,mesh,collectives,projection['partial'],retain)
    normalized = normalize_partials(operations,first,second,parameters['hidden_norm.weight'],retain,composed_norm=True)
    current = {name:inputs[name] for name in BACKBONE_INPUTS if name!='context'}
    current['context'] = normalized['context']
    result = backbone(operations,mesh,collectives,current,layer_weights,parameters['norm.weight'],retain,mask_validated=True)
    return dict(projection={**projection,**normalized},backbone=result)


def flatten(result):
    if set(result)!= {'projection','backbone'} or set(result['projection'])!= {
            'joined','partial','sum','narrowed','unweighted_norm','context'}:
        raise ValueError('Complete projection and five-layer backbone stages required')
    stages = {'projection.'+name:value for name,value in result['projection'].items()}
    stages.update({f'{layer}.{phase}.{stage}':value for (layer,phase,stage),value in backbone_stages(result['backbone']).items()})
    return stages
