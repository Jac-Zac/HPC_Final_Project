# TODO:

- Understand why the code doesn't scale with openmp
- Think of mapping correctly
- Used the tools and print where things are placed
- Do a raugh estimate of how much of the peak performance you are using
- Alternative domain decomposition with OPENMP (need to touch the memory correctly) -> and then I can mpi scale by node
  but for each tile I touch a part that is filled by the tread
