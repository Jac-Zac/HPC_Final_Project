#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *

= Code optimizations 

1. Rewrite intensive computation part to be more efficient
1. Implemented gcc extension SIMD implementation but realized that the compiler already does SIMD at O3 by himselfe
2. Added openmp parallelism tested different things guided non collapsed is the fastest
3. Added mpi communication with blocking send receive (As the num of processes increases communication becomes the bottleneck)


=== Node Status Script

Finally we can _create/use_ the following script to check what nodes are running.

#figure(
  sourcecode(lang: "bash")[
    ```bash
    #!/bin/bash
    n=${1:-9}
    nodes=(master)
    for i in $(seq 1 "$n"); do
      num=$(printf "%02d" "$i")
      nodes+=("node-$num")
    done

    for node in "${nodes[@]}"; do
      echo -n "$node: "
      ping -c 1 -W 1 "$node" >/dev/null && echo "UP" || echo "DOWN"
    done
    ```
  ],
  caption: "Script check_node.sh: to get the machines that are up",
)

#ideabox()[Save scripts in `/shared/scripts/` for shared access]
