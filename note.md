# TODO:

- [ ] Big run
- [ ] Report / PowerPoint presentation
- [ ] Plot also the serial TIME

# NOTE:

- **No good way of doing good memory allingment** based on how I access them. At least I think. On my computer I see this:

```asm
smvmovupd (%r12,%rax,1),%ymm7    # Loading 4 doubles at once
vaddpd 0x0(%r13,%rax,1),%ymm7,%ymm0  # Adding 4 doubles at once
```

There is an improvement of around 15 %
From 3.8 to 3.2 seconds 8 threads 1 mpi task

- Tiling improve and avoids the problem being memory bound so it is much more scalable
- Though it doesn't actually help when I access the same numb region
- Putting things in different ccd ccx regions actually helps out a bit

## Scaling MPI

_Interestingly the communication time goes down this is probably because of the memory bounded nature of the task and exchanging smaller borders with more memory only helps_

### Additional notes:

Message size effect, Each rank’s subdomain is smaller when you scale out. Halo sizes shrink too:

- With 1 big domain per rank -> each message is large.

- With more ranks -> each message is smaller.

- MPI transfers smaller halos faster (lower bytes to move), so the MPI_Waitall finishes earlier.

This effect can dominate: smaller halo -> shorter wait time -> measured comm time decreases. Max-reduction hides imbalance

### Strong scaling is done with 50k x 50k for 1000 iteration

### Weak scaling is done with 25k \* 25k for 1000 iteration as a starting point -> 35350 \* 35350 -> 50k \* 50k -> 70711 \* 70711 ->
