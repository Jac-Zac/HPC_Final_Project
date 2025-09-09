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
