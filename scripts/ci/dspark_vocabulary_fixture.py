"""Full-size synthetic logits for transport only; distinct ranks, rows and vocabulary-boundary sentinels."""

from dspark_inputs import PROPOSALS, VOCABULARY


def inputs(pattern):
    import torch

    if type(pattern) is not int or pattern not in (0,1,2):
        raise ValueError('One of three declared full-vocabulary transport patterns required')
    columns = torch.arange(VOCABULARY,dtype=torch.int64)
    rows = torch.arange(32,dtype=torch.int64)[:,None]
    values = (((columns%113)+rows*3+pattern*7).float()/64+(columns//(VOCABULARY//2)).float()*4)
    full = values.reshape(1,1,32,VOCABULARY).bfloat16()
    for row in (0,1,6,7,31):
        for ordinal,column in enumerate((0,VOCABULARY//2-1,VOCABULARY//2,VOCABULARY-1)):
            full[0,0,row,column] = -64+ordinal*16+row/4+pattern/2
    packed = torch.cat(full.chunk(2,dim=-1),dim=0).contiguous()
    return dict(packed=packed,full_logits=full,base_logits=full[:,:,:PROPOSALS].float().clone())
