#!/bin/bash
#./start-training --algo ppo --env-type prune-still  training_runs/tasks_aup/tasks_l1e8-1e4/prune-still   
#./start-training --algo ppo --env-type navigate     training_runs/tasks_aup/tasks_l86-5/navigate      
./start-training --algo ppo-rr --env-type prune-spawn  training_runs/tasks_aup/long_l1e8-1e3_5rr_z100/prune-spawn   
./start-training --algo ppo-rr --env-type append-still training_runs/tasks_aup/long_l1e8-1e3_5rr_z100/append-still 
./start-training --algo ppo-rr --env-type append-spawn training_runs/tasks_aup/long_l1e8-1e3_5rr_z100/append-spawn  
