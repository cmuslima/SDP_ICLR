## SDP Code base


##  Installation

```
Follow instructions at these links to install mujoco 200:
http://www.cs.cmu.edu/~cga/controls-intro-22/kantor/How_to_Install_MuJoCo_on_Ubuntu_V1.pdf

License: https://www.roboti.us/file/mjkey.txt
Add this into .mujoco dir
We recommend creating separate virtual envs for dmc and metaworld experiments. 

conda create --name sdp_env_dmc python=3.8
conda activate sdp_env_dmc
pip install -r dmc_requirements.txt
cd custom_dmcontrol
pip install -e .
cd ../custom_dmc2gym
pip install -e .

conda create --name sdp_env_metaworld python=3.8
conda activate sdp_env_metaworld
pip install -r metaworld_requirements.txt
```




## To train an preference learning agent with a simulated teacher:

## SDP
* Step 1:
    * Run reward model pretraining via:
        * python SDP_reward_pretraining.py num_seed_steps=50000 num_train_steps=50000 activation=tanh 

    * Note that this assumes you want to get suboptimal transitions via a random policy, where num_seed_steps is the number of sub-optimal transtitions you want to use. Num_train_steps must equal the num_seed_steps


* Step 2: Human in the loop RL algorithm that follows in SDP
    * To train SDP-PEBBLE: 
        * python train_SDP_PEBBLE.py prior_data_path="" base_dir="" env_domain=DMCONTROL env=walker_walk experiment=test device=cpu 
        * where base_dir/prior_data_path is the location of the actor/critic, reward model and replay buffer that was saved in step 1

    * Note that the hyperparameters for the SAC neural network arch. must be the same for both the reward model pretraining and the HiTL alg that follows, if you want to use different neural network arch for SAC, set the load_agent arg to be 0.

    * I like to keep the SDP phases done in Step 1 seperate from the HiTL alg that follows because this allows you to keep the same pretrained reward model/agent but change up the hyperparameters used in the HiTL alg that follows if you wanted to. This way you do not need to repeat step 1. 

    * To train SDP-RUNE, SDP-SURF, SDP-MRN: change the .py file to train_SDP_{alg} where alg=RUNE or SURF or MRN

## Other baseline algs
* To train PEBBLE, SURF, RUNE, SURF:
    * python train_PEBBLE.py env_domain=DMCONTROL env=walker_walk experiment=test device=cpu 

* To train RUNE, SURF, MRN: 
    * change the .py file to train_{alg} where alg=RUNE or SURF or MRN



## To train an preference learning agent with a  human teacher:

* Step 1:
    * Run reward model pretraining via:
        * python SDP_reward_pretraining.py num_seed_steps=50000 activation=tanh  prior_data_path="" base_dir="" 
        * where base_dir/prior_data_path is the location of the actor/critic, reward model and replay buffer that was saved in step 1
* Step 2: Human in the loop RL algorithm that follows in SDP
    * To train SDP-PEBBLE: 
        * python train_SDP_human_preferences.py 

* To train PEBBLE:
    * python train_PEBBLE_human_preferences.py 


## To train an scalar feedback agent with a simulated teacher:


* To train SDP + R-PEBBLE:
    * python SDP_regression.py num_seed_steps=50000 
    * where num_train_steps > 50000
    * Note that this assumes you want to get suboptimal transitions via a random policy, where num_seed_steps is the number of sub-optimal transtitions you want to use. 

* To train PEBBLE:
    * python train_RPEBBLE.py  

