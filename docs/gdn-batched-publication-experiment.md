# Batched GDN publication experiment

Status: implementation and host checks only. No simulator or hardware pass yet.
The qualified publication path and serving defaults remain unchanged.

## Why test it

The matched score-layout request spends about 11.77 ms/block in selection and
commit, including more than this kernel. Source inspection shows the current
publication kernel waits after each recurrent tile and each convolution tile.
That is not evidence that all 11.77 ms is removable.

The isolated `gdn_commit_batched_dma` candidate keeps two workers per layer
(96 workers for 48 layers). Each worker batches eight independent transfers
before a read barrier and eight publications before a write barrier. Scratch
increases from 4 KiB to 32 KiB per worker. Convolution reads fetch only the two
32-byte face rows actually used, instead of each complete 2048-byte tile.
The arithmetic, selected prefix, checkpoint padding and inactive-slot writes
are unchanged by design; simulator evidence must verify that claim.

## Admission sequence

1. Compare native and candidate publication in TTsim on both chips, T16 prefixes
   0 through 16, with distinct recurrent/convolution histories and inactive slots.
2. Require bit-exact native and checkpoint outputs, zeroed checkpoint padding,
   unchanged entry/history inputs and all seven inactive native slots preserved.
3. Capture and replay changed histories, poison outputs, and check stable binding
   and clean release. Include a 48-layer case to exercise all 96 workers.
4. Only after simulator qualification, compare learned-state hardware publication
   and full accepted-prefix continuation. Keep native publication as control.
5. Run matched full requests with the repeat-confirmed score layout in both arms;
   compare committed TG and the complete selection/publication boundary, not
   kernel timing alone. Reject any token, state, acceptance or quality regression.

The current verifier profiler is independent and still needed: even removing
publication completely would not make the existing request reach 200 TG.
