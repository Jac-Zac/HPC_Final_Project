# TODO:

- [ ] Make Irecv and Isend communication
- [ ] Try mpi datatypes for east and west and north and south continues
- [ ] Big run
- [ ] Report / PowerPoint presentation
- [ ] Plot also the serial TIME

# NOTE:

There is an improvement of around 15 %
From 3.8 to 3.2 seconds 8 threads 1 mpi task

- Tiling improve and avoids the problem being memory bound so it is much more scalable
- Though it doesn't actually help when I access the same numb region
- Putting things in different ccd ccx regions actually helps out a bit
