# TODO:

- [x] Understand why the code doesn't scale with openmp
- [x] Think of mapping correctly in mpi
- [x] Used the tools and print where things are placed
- [x] Review mpi communication
- [x] OpenMP scaling + touch first works correctly compare it to not touching and not spread through NUMA regions
- [ ] Test old version of the update loop vs my custom version to see the difference
- [ ] Tiling reduces cache misses drastically but makes code too slow because of overhead perhaps you can just tile the threads
- [ ] Make Irecv and Isend communication
- [ ] Check correctness of parallel code with python code
- [ ] Try mpi datatypes for east and west and north and south continues
- [ ] Alternative domain decomposition with OPENMP (need to touch the memory correctly) -> and then I can mpi scale by node
      but for each tile I touch a part that is filled by the thread
- [ ] Big run
- [ ] Report
