# swarm.py
## FOR RUNNING SWARM ON NIH BIOWULF HPC CLUSTER ##
import subprocess


def run_feat_red_swarm(
    cohort_list,
    yr_rng_list,
    log_base_dir,
    cmd_dir,
    n_threads,
    swarm_time,
    gb,
    partition,
):
    for cohort in cohort_list:
        for yr_rng in yr_rng_list:
            swarm_path = cmd_dir / yr_rng / f"{cohort}.swarm"
            log_dir = log_base_dir / yr_rng / cohort
            log_dir.mkdir(parents=True, exist_ok=True)
            swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {(2*n_threads)} --logdir={log_dir}  --job-name=feat_red_{cohort.upper()}_{yr_rng} --partition={partition} -b 1 {swarm_path}"
            subprocess.run(swarm_cmd, shell=True)


def run_feat_red_confirm_swarm(
    group_size_list, log_base_dir, cmd_dir, n_threads, swarm_time, gb, partition
):
    for group_size in group_size_list:
        swarm_path = cmd_dir / f"{group_size}.swarm"
        log_dir = log_base_dir / group_size
        log_dir.mkdir(parents=True, exist_ok=True)
        swarm_cmd = f"swarm --time={swarm_time} -g {gb} -t {(3*n_threads)} --logdir={log_dir}  --job-name=reduc_confirm_{group_size.upper()} --partition={partition} -b 1 {swarm_path}"
        subprocess.run(swarm_cmd, shell=True)
