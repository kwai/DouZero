# [ICML 2021] DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning
<img width="500" src="https://raw.githubusercontent.com/kwai/DouZero/main/imgs/douzero_logo.jpg" alt="Logo" />

[![Building](https://github.com/kwai/DouZero/actions/workflows/python-package.yml/badge.svg)](https://github.com/kwai/DouZero/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/douzero.svg)](https://badge.fury.io/py/douzero)
[![Downloads](https://pepy.tech/badge/douzero)](https://pepy.tech/project/douzero)
[![Downloads](https://pepy.tech/badge/douzero/month)](https://pepy.tech/project/douzero)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)

[中文文档](README.zh-CN.md)

DouZero is a reinforcement learning framework for [DouDizhu](https://en.wikipedia.org/wiki/Dou_dizhu) ([斗地主](https://baike.baidu.com/item/%E6%96%97%E5%9C%B0%E4%B8%BB/177997)), the most popular card game in China. It is a shedding-type game where the player’s objective is to empty one’s hand of all cards before other players. DouDizhu is a very challenging domain with competition, collaboration, imperfect information, large state space, and particularly a massive set of possible actions where the legal actions vary significantly from turn to turn. DouZero is developed by AI Platform, Kwai Inc. (快手).

*   Online Demo: [https://www.douzero.org/](https://www.douzero.org/)
       * :loudspeaker: New Version with Bid（叫牌版本）: [https://www.douzero.org/bid](https://www.douzero.org/bid)
*   Run the Demo Locally: [https://github.com/datamllab/rlcard-showdown](https://github.com/datamllab/rlcard-showdown)
*   Paper: [https://arxiv.org/abs/2106.06135](https://arxiv.org/abs/2106.06135) 
*   Related Project: [RLCard Project](https://github.com/datamllab/rlcard)
*   Related Resources: [Awesome-Game-AI](https://github.com/datamllab/awesome-game-ai)
*   Google Colab: [jupyter notebook](https://github.com/daochenzha/douzero-colab/blob/main/douzero-colab.ipynb)
*   Unofficial improved versions of DouZero by the community: [[DouZero ResNet]](https://github.com/Vincentzyx/Douzero_Resnet) [[DouZero FullAuto]](https://github.com/Vincentzyx/DouZero_For_HLDDZ_FullAuto)
*   Zhihu: [https://zhuanlan.zhihu.com/p/526723604](https://zhuanlan.zhihu.com/p/526723604)

**Community:**
*  **Slack**: Discuss in [DouZero](https://join.slack.com/t/douzero/shared_invite/zt-rg3rygcw-ouxxDk5o4O0bPZ23vpdwxA) channel.
*  **QQ Group**: Join our QQ group to discuss. Password: douzeroqqgroup

	*  Group 1: 819204202
	*  Group 2: 954183174
	*  Group 3: 834954839
	*  Group 4: 211434658

**News:**
*   Thanks for the contribution of [@Vincentzyx](https://github.com/Vincentzyx) for enabling CPU training. Now Windows users can train with CPUs.

<img width="500" src="https://douzero.org/public/demo.gif" alt="Demo" />

## Cite this Work
If you find this project helpful in your research, please cite our paper:

Zha, Daochen et al. “DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning.” ICML (2021).

```bibtex
@InProceedings{pmlr-v139-zha21a,
  title = 	 {DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning},
  author =       {Zha, Daochen and Xie, Jingru and Ma, Wenye and Zhang, Sheng and Lian, Xiangru and Hu, Xia and Liu, Ji},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {12333--12344},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/zha21a/zha21a.pdf},
  url = 	 {http://proceedings.mlr.press/v139/zha21a.html},
  abstract = 	 {Games are abstractions of the real world, where artificial agents learn to compete and cooperate with other agents. While significant achievements have been made in various perfect- and imperfect-information games, DouDizhu (a.k.a. Fighting the Landlord), a three-player card game, is still unsolved. DouDizhu is a very challenging domain with competition, collaboration, imperfect information, large state space, and particularly a massive set of possible actions where the legal actions vary significantly from turn to turn. Unfortunately, modern reinforcement learning algorithms mainly focus on simple and small action spaces, and not surprisingly, are shown not to make satisfactory progress in DouDizhu. In this work, we propose a conceptually simple yet effective DouDizhu AI system, namely DouZero, which enhances traditional Monte-Carlo methods with deep neural networks, action encoding, and parallel actors. Starting from scratch in a single server with four GPUs, DouZero outperformed all the existing DouDizhu AI programs in days of training and was ranked the first in the Botzone leaderboard among 344 AI agents. Through building DouZero, we show that classic Monte-Carlo methods can be made to deliver strong results in a hard domain with a complex action space. The code and an online demo are released at https://github.com/kwai/DouZero with the hope that this insight could motivate future work.}
}
```

## What Makes DouDizhu Challenging?
In addition to the challenge of imperfect information, DouDizhu has huge state and action spaces. In particular, the action space of DouDizhu is 10^4 (see [this table](https://github.com/datamllab/rlcard#available-environments)). Unfortunately, most reinforcement learning algorithms can only handle very small action spaces. Moreover, the players in DouDizhu need to both compete and cooperate with others in a partially-observable environment with limited communication, i.e., two Peasants players will play as a team to fight against the Landlord player. Modeling both competing and cooperation is an open research challenge.

In this work, we propose Deep Monte Carlo (DMC) algorithm with action encoding and parallel actors. This leads to a very simple yet surprisingly effective solution for DouDizhu. Please read [our paper](https://arxiv.org/abs/2106.06135) for more details.

## Installation
The training code is designed for GPUs. Thus, you need to first install CUDA if you want to train models. You may refer to [this guide](https://docs.nvidia.com/cuda/index.html#installation-guides). For evaluation, CUDA is optional and you can use CPU for evaluation.

First, clone the repo with (if you are in China and Github is slow, you can use the mirror in [Gitee](https://gitee.com/daochenzha/DouZero)):
```
git clone https://github.com/kwai/DouZero.git
```
Make sure you have python 3.6+ installed. Install dependencies:
```
cd douzero
pip3 install -r requirements.txt
```
We recommend installing the stable version of DouZero with
```
pip3 install douzero
```
If you are in China and the above command is too slow, you can use the mirror provided by Tsinghua University:
```
pip3 install douzero -i https://pypi.tuna.tsinghua.edu.cn/simple
```
or install the up-to-date version (it could be not stable) with
```
pip3 install -e .
```
Note that Windows users can only use CPU as actors. See [Issues in Windows](README.md#issues-in-windows) about why GPUs are not supported. Nonetheless, Windows users can still [run the demo locally](https://github.com/datamllab/rlcard-showdown).  

## Training
To use GPU for training, run
```
python3 train.py
```
This will train DouZero on one GPU. To train DouZero on multiple GPUs. Use the following arguments.
*   `--gpu_devices`: what gpu devices are visible
*   `--num_actor_devices`: how many of the GPU deveices will be used for simulation, i.e., self-play
*   `--num_actors`: how many actor processes will be used for each device
*   `--training_device`: which device will be used for training DouZero

For example, if we have 4 GPUs, where we want to use the first 3 GPUs to have 15 actors each for simulating and the 4th GPU for training, we can run the following command:
```
python3 train.py --gpu_devices 0,1,2,3 --num_actor_devices 3 --num_actors 15 --training_device 3
```
To use CPU training or simulation (Windows can only use CPU for actors), use the following arguments:
*   `--training_device cpu`: Use CPU to train the model
*   `--actor_device_cpu`: Use CPU as actors

For example, use the following command to run everything on CPU:
```
python3 train.py --actor_device_cpu --training_device cpu
```
The following command only runs actors on CPU:
```
python3 train.py --actor_device_cpu
```
For more customized configuration of training, see the following optional arguments:
```
--xpid XPID           Experiment id (default: douzero)
--save_interval SAVE_INTERVAL
                      Time interval (in minutes) at which to save the model
--objective {adp,wp}  Use ADP or WP as reward (default: ADP)
--actor_device_cpu    Use CPU as actor device
--gpu_devices GPU_DEVICES
                      Which GPUs to be used for training
--num_actor_devices NUM_ACTOR_DEVICES
                      The number of devices used for simulation
--num_actors NUM_ACTORS
                      The number of actors for each simulation device
--training_device TRAINING_DEVICE
                      The index of the GPU used for training models. `cpu`
                	  means using cpu
--load_model          Load an existing model
--disable_checkpoint  Disable saving checkpoint
--savedir SAVEDIR     Root dir where experiment data will be saved
--total_frames TOTAL_FRAMES
                      Total environment frames to train for
--exp_epsilon EXP_EPSILON
                      The probability for exploration
--batch_size BATCH_SIZE
                      Learner batch size
--unroll_length UNROLL_LENGTH
                      The unroll length (time dimension)
--num_buffers NUM_BUFFERS
                      Number of shared-memory buffers
--num_threads NUM_THREADS
                      Number learner threads
--max_grad_norm MAX_GRAD_NORM
                      Max norm of gradients
--learning_rate LEARNING_RATE
                      Learning rate
--alpha ALPHA         RMSProp smoothing constant
--momentum MOMENTUM   RMSProp momentum
--epsilon EPSILON     RMSProp epsilon
```

## Evaluation
The evaluation can be performed with GPU or CPU (GPU will be much faster). Pretrained model is available at [Google Drive](https://drive.google.com/drive/folders/1NmM2cXnI5CIWHaLJeoDZMiwt6lOTV_UB?usp=sharing) or [百度网盘](https://pan.baidu.com/s/18g-JUKad6D8rmBONXUDuOQ), 提取码: 4624. Put pre-trained weights in `baselines/`. The performance is evaluated through self-play. We have provided pre-trained models and some heuristics as baselines:
*   [random](douzero/evaluation/random_agent.py): agents that play randomly (uniformly)
*   [rlcard](douzero/evaluation/rlcard_agent.py): the rule-based agent in [RLCard](https://github.com/datamllab/rlcard)
*   SL (`baselines/sl/`): the pre-trained deep agents on human data
*   DouZero-ADP (`baselines/douzero_ADP/`): the pretrained DouZero agents with Average Difference Points (ADP) as objective
*   DouZero-WP (`baselines/douzero_WP/`): the pretrained DouZero agents with Winning Percentage (WP) as objective

### Step 1: Generate evaluation data
```
python3 generate_eval_data.py
```
Some important hyperparameters are as follows.
*   `--output`: where the pickled data will be saved
*   `--num_games`: how many random games will be generated, default 10000

### Step 2: Self-Play
```
python3 evaluate.py
```
Some important hyperparameters are as follows.
*   `--landlord`: which agent will play as Landlord, which can be random, rlcard, or the path of the pre-trained model
*   `--landlord_up`: which agent will play as LandlordUp (the one plays before the Landlord), which can be random, rlcard, or the path of the pre-trained model
*   `--landlord_down`: which agent will play as LandlordDown (the one plays after the Landlord), which can be random, rlcard, or the path of the pre-trained model
*   `--eval_data`: the pickle file that contains evaluation data
*   `--num_workers`: how many subprocesses will be used
*   `--gpu_device`: which GPU to use. It will use CPU by default

For example, the following command evaluates DouZero-ADP in Landlord position against random agents
```
python3 evaluate.py --landlord baselines/douzero_ADP/landlord.ckpt --landlord_up random --landlord_down random
```
The following command evaluates DouZero-ADP in Peasants position against RLCard agents
```
python3 evaluate.py --landlord rlcard --landlord_up baselines/douzero_ADP/landlord_up.ckpt --landlord_down baselines/douzero_ADP/landlord_down.ckpt
```
By default, our model will be saved in `douzero_checkpoints/douzero` every half an hour. We provide a script to help you identify the most recent checkpoint. Run
```
sh get_most_recent.sh douzero_checkpoints/douzero/
```
The most recent model will be in `most_recent_model`.

## Issues in Windows
You may encounter `operation not supported` error if you use a Windows system to train with GPU as actors. This is because doing multiprocessing on CUDA tensors is not supported in Windows. However, our code extensively operates on the CUDA tensors since the code is optimized for GPUs. Please contact us if you find any solutions!

## Core Team
*   Algorithm: [Daochen Zha](https://github.com/daochenzha), [Jingru Xie](https://github.com/karoka), Wenye Ma, Sheng Zhang, [Xiangru Lian](https://xrlian.com/), Xia Hu, [Ji Liu](http://jiliu-ml.org/)
*   GUI Demo: [Songyi Huang](https://github.com/hsywhu)
*   Community contributors: [@Vincentzyx](https://github.com/Vincentzyx)

## Acknowlegements
*   The demo is largely based on [RLCard-Showdown](https://github.com/datamllab/rlcard-showdown)
*   Code implementation is inspired by [TorchBeast](https://github.com/facebookresearch/torchbeast)














