# TODO:

- [x] Understand why the code doesn't scale with openmp
- [x] Think of mapping correctly in mpi
- [x] Used the tools and print where things are placed
- [x] Review mpi communication
- [ ] Make Irecv and Isend communication
- [ ] Do a rough estimate of how much of the peak performance you are using
- [ ] Check correctness of parallel code with python code
- [ ] Try mpi datatypes for east and west and north and south continues
- [ ] Alternative domain decomposition with OPENMP (need to touch the memory correctly) -> and then I can mpi scale by node
      but for each tile I touch a part that is filled by the thread
- [ ] Big run
- [ ] Report
