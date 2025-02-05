
# Hardware

Here we provide additional information about the hardware used to execute the experiments in the paper.

These details can be useful in terms of reproducibility, to tune your runtime expectations and asses the results when running Ultrack on different platforms.

Three different hardware configurations were used in the experiments,

### Desktop

- Intel i9 9900K CPU
- 4x DDR4 3200Hz 32GB RAM (128GB in total)
- Asus Motherboard Z390 PRO
- Corsair power supply unit 1200
- 2x NVIDIA RTX 3090 Ti
- 1TB Samsung NVMe SSD
- 4x 4TB hard drives
- 10Gbit Ethernet card
- Operational System: Ubuntu 20.04 LTS

### Cluster

The cluster constains elastic amount of resources, we constrained our jobs to use at most 100-cpu cores and 20 GPUS when measuring our runtime for different steps.

The resources used for each step are as follows:

0. Network prediction: 20x 4-cpu core 100GB RAM 1 A40 GPU nodes
1. Database: 1x 10-cpu core 1000GB RAM node (extremely excessive, today we are using 128GB of RAM)
2. Segmentation (hypotheses generation): 90x 1-cpu core 100GB RAM nodes
3. Linking (hypotheses linking): 90x 1-cpu core 5GB RAM nodes
4. Tracking: 8x 12-cpu cores 400GB RAM nodes
5. Exporting: 1x 90-cpu core 400GB RAM node

Steps are run sequentially, thus each line corresponds to the total resources used for that step, with the exception of the database which is on and available for all steps after item 1.

The cluster runs Rocky Linux 8, Slurm is used for job scheduling, and GPFS file system.

### Laptop

- Intel i9 11980HK CPU
- DDR5 3200Hz 32GB RAM
- NVIDIA RTX 3080 Ti
- 2TB NVMe SSD
- 1Gbit Ethernet
- Operational System: Ubuntu 22.04 LTS
