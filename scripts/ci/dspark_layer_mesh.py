"""Single-call learned DSpark layer with device-only TP2 handoffs and caller-owned tensors."""

from dspark_layer import attention_partial, finish, mlp_partial
from dspark_mesh import gather_partials


INPUTS = ('context','noise','q_cos','q_sin','k_cos','k_sin','mask','live')


def execute(operations, mesh, collectives, inputs, weights, retain, *, mask_validated=False):
    if set(inputs)!=set(INPUTS) or mask_validated is not True or not callable(retain):
        raise ValueError('Complete learned inputs, prevalidated mask and caller-owned tensors required')
    if list(mesh.shape)!=[1,2]:
        raise ValueError('Explicit two-chip mesh required')
    attention = attention_partial(operations,inputs['context'],inputs['noise'],weights,
        {name:(inputs[name+'_cos'],inputs[name+'_sin']) for name in ('q','k')},
        inputs['mask'],inputs['live'],retain,mask_validated=True,composed_attention=True,mesh=mesh)
    parts = gather_partials(operations,mesh,collectives,attention['attention_partial'],retain)
    mlp = mlp_partial(operations,*parts,inputs['noise'],weights,retain)
    parts = gather_partials(operations,mesh,collectives,mlp['down_partial'],retain)
    output = finish(operations,*parts,mlp['attention_residual'],retain)
    return dict(attention=attention,mlp=mlp,finish=output)
