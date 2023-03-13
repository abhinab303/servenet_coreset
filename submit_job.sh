#!/bin/bash
base_job_name="sn_cs_350"
job_file="the_job.sh"
identifier_name="sn_every_eph"
dir="op_"$identifier_name
mkdir -p $dir


methods="TD"
ss_sizes={0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}

job_name=$base_job_name-"EE"
out_file=$dir/$job_name.out
error_file=$dir/$job_name.err

# for ss_size in {1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9};
for ss_size in {1..1};
    do
        export ss_size 
        job_name=$base_job_name-$ss_size-ss_size
        out_file=$dir/$job_name.out
        error_file=$dir/$job_name.err

        echo $ss_size ------------------------------------------------------------------
        sbatch -J $job_name -o $out_file -e $error_file $job_file     
done

# sbatch -J $job_name -o $out_file -e $error_file $job_file