# Introduction
这是一个复旦大学《具身智能引论》课程的作业项目。

![](pictures\2025-06-22-13-56-14.png)

由于这不是专业课, 所以本项目有很多漏洞, 配置管理很简单粗暴, 仅供参考。

# Overview
本项目实现了两种功能:
1. 固定环境下的寻路;
2. 可变化环境下的寻路。

前者直接对应课程要求, 只需要修改问题的配置就可以完成作业的两个问题; 后者属于拓展的内容, 简单展示深度强化学习的使用。

# Installation
## (a) With Conda
如果你会用 conda, 可以按如下方法配置环境。
```bash
conda create -n EI_intro python=3.9
conda activate EI_intro
```
保证你移动到了 EI_intro 目录后运行:
```bash
pip install -r requirements.txt
```

# (b) Without Conda
如果你还没有 conda, 那么在移动到项目根目录后, 直接你的 python 环境运行如下命令即可:
```bash
pip install -r requirements.txt
```

# Usage
## (a) Customization
我没有使用各种配置文件, 你可以在直接在 RL/data_define.py 的 data_used 简单类中定义超参和任务配置, 比如每一步的惩罚, 是否允许回退, 终点奖励等。

## Run & Test
### Stable Version
默认情况下, data_used 内的设置是我们作业中的第一种情形, 即不允许回到上一步的起点, 每步惩罚为 -6。**如果要运行第二种情形, 只要将 allow_backtrack 设为 1, cost_each_step 设为 -4 即可**。 
设置完成后, 使用下面的命令可以直接运行项目。其首先会面向环境进行训练, 然后询问是否演示训练效果。训练的结果会保存在 checkpoint 中。
```bash
python run.py
```

你可以直接进入到 train 或者 demo 来单独运行 train.py 和 demo.py, 但是由于各种参数和文件路径都定义在 data_define.py 中, 所以请注意过程中不要修改 data_define.py, 这可能导致结果错误或者模型参数无法对齐的报错。

### Bonus: Flexible Version
借助深度强化学习训练一个可以在变化的环境中自动寻路的智能体。

你也可以运行下面的程序来跑一个类似上面的 baseline。。
```bash
python run_flexible_pro.py
```

你也可以仅进行训练。根据测试, 就 4*4 的网格而言, 学习率为 4e-5 时, pro 版本训练大约四万个时间步模型就收敛了。在台式机 5080 上耗时大约一分钟, 使用 CPU 耗时大约 15 分钟。 
```bash
python train/train_flexible_pro.py
```
然后运行下面的命令进行测试, **注意过程中尽量不要修改 data_define.py, 否则可能找不到文件或者参数无法对齐导致错误**:
```bash
python demo/demo_flexible_pro.py
```