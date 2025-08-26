#import "@preview/hei-synd-report:0.1.1": *
#import "/metadata.typ": *

= Conclusions

To conclude, the benchmark analysis confirms what was expected.
Containerised environments consistently outperform VMs across most areas. 
However, it is important to note that this project was done on a macOS system, which runs containers inside a virtualised Linux environment, introducing an additional layer of abstraction. 

Thus, containers remain superior in performance and scalability compared to VMs. _This is probably because of the implementation of a very efficient lightweight (VM) running inside Docker Engine_. Though the overall efficiency is not as high as it would be on native Linux host. 
Indeed, CPU and memory benchmarks using sysbench show worse performance with virtual machines than containers, though both appear to be behind native host performance. 
In Disk I/O benchmarks the disparity between the two modalities is even bigger especially when dealing with shared file-systems. Indeed, containers, in particular those with docker-managed volumes, show much better write and consistency performance than VMs. 
The highest difference can be seen in the network benchmarks, where containers have achieved near-native bandwith and very low latency (below millisecond) because they are essentially just two process communicating one to the other. 
On the other hand, virtual machines show lower throughput and higher latency in communication between each other, because network traffic must pass through a virtual switch, which adds significant overhead.

Overall, containers demonstrated strong performance, though it would be interesting to test a true containerized macOS environment to evaluate it and compare it to running a virtualized Linux instance inside Docker Engine. 
Such a comparison could reveal the performance benefits of native containerization versus the virtualization overhead.
VMs, due to their heavier virtualization stack, tend to exhibit performance penalties; however, they remain highly valuable as they enable emulation of different architectures, which can be essential in many scenarios.
