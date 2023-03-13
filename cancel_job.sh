#!/bin/bash


for job_id in {15835561..15835568};
    do
        scancel $job_id
done