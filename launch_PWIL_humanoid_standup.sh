DEMO_DIR=/home/chenu/git/pwil/demonstrations/humanoid_standup_env/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenu/anaconda3/envs/pwil/lib/
cd /home/chenu/git/
#nohup python pwil/trainer.py --workdir='/home/chenu/git/pwil/results/' --env_name='HumanoidStandupEnv-v0' --demo_dir=$DEMO_DIR &> /home/chenu/git/pwil/results/pwil_humanoidstandupenv_1.out &
python pwil/trainer.py --workdir='/home/chenu/git/pwil/results/' --env_name='HumanoidStandupEnv-v0' --demo_dir=$DEMO_DIR 
