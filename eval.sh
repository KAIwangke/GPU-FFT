# 1. Memory Transaction Analysis
ncu --metrics "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct" \
    ./a.out /home/kw3484/GPU_course/FFT-cuda/data/50Hz

# 2. Memory Throughput Analysis
ncu --metrics "dram__bytes_read.sum,\
dram__bytes_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed" \
    ./a.out /home/kw3484/GPU_course/FFT-cuda/data/50Hz

# 3. Cache Efficiency Analysis
ncu --metrics "l1tex__t_hit_rate.pct,\
lts__t_hit_rate.pct,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed" \
    ./a.out /home/kw3484/GPU_course/FFT-cuda/data/50Hz

# 4. Memory Access Pattern Analysis
ncu --metrics "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio" \
    ./a.out /home/kw3484/GPU_course/FFT-cuda/data/50Hz

# 5. Detailed Memory Workload Analysis
ncu --section MemoryWorkloadAnalysis \
    --section-folder "memory_analysis" \
    ./a.out /home/kw3484/GPU_course/FFT-cuda/data/50Hz




# Metric Name                                    Metric Unit Metric Value
# ---------------------------------------------- ----------- ------------
# l1tex__t_sector_hit_rate.pct                             %        75.76
# l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum      sector           32
# l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum      sector           34
# lts__t_sector_hit_rate.pct                               %       245.26    