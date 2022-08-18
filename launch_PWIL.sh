DEMO_DIR=/home/chenu/git/pwil/demonstrations/humanoidenv/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chenu/anaconda3/envs/pwil/lib/
cd /home/chenu/git/
#nohup python pwil/trainer.py --workdir='/home/chenu/git/pwil/results/' --env_name='DubinsMazeEnv-v0' --demo_dir=$DEMO_DIR &> /home/chenu/git/pwil/results/pwil_nohup14.out &
#nohup python pwil/trainer.py --workdir='/home/chenu/git/pwil/results/' --env_name='HumanoidEnv-v0' --demo_dir=$DEMO_DIR &> /home/chenu/git/pwil/results/pwil_humanoidenv_1.out &
python pwil/trainer.py --workdir='/home/chenu/git/pwil/results/' --env_name='HumanoidEnv-v0' --demo_dir=$DEMO_DIR
