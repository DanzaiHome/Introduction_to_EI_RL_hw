# Introduction
这是一个用于《具身智能引论》课程的很小的, 但是能一定程度上自定义化的项目。其在给定的奖励网格环境下训练一个智能体, 自起点出发, 寻找到达终点时能够获得最大得分的路径。

# Installation
```bash
conda create -n EI_intro python=3.9
conda activate EI_intro
```
保证你移动到了 EI_intro 目录后运行:
*tips: 如果想要尝试 flexible version, 建议上 Pytorch 官网下载带 cuda 的 torch*
```bash
pip install -r requirements.txt
```

# Usage
## Customization
我没有使用各种配置文件, 所以可以在直接在 EI_intro/data_define.py 的 data_used 简单类中定义超参, 奖励网格, 奖励规则。

## Run & Test
### Stable Version
稳定模式即我们项目的基本要求, 即训练一个在给定环境下的智能体。
默认情况下, data_used 内的设置是我们作业中的第一种情形, 即不允许回到上一步的起点, 每步惩罚为 -6。**如果要运行第二种情形, 需要将 allow_backtrack 设为 1, cost_each_step 设为 -4**。 
设置完成后, 使用下面的命令可以直接运行项目。其首先会面向环境进行训练, 然后询问是否演示训练效果。训练的结果会保存在 checkpoint 中。
```bash
python run.py
```
经过测试, 对于项目提供的 4*4 网格,  40 个 episodes 就能非常稳定地得到正确答案。使用 CPU 完成 40 个 episodes 的训练耗时大约 10 秒。
根据项目的定义, 最终答案很可能是正无穷。如果训练中发现这一情况(基于有无正回路)会直接退出。

你可以直接进入到 train 或者 demo 来单独运行 train.py 和 demo.py, 但是请注意过程中不要修改 data_define.py, 这可能导致结果错误或者模型参数无法对齐的报错。

### Bonus: Flexible Version
上面训练的智能体无法迁移到其他环境, 也不能体现深度学习的优势。虽然我使用的是 DQL, 但实际效果和普通 QL 一样, 速度还更慢。深度强化学习在更复杂的问题能够有好的表现, 如训练一个能够**在给定大小的, 各格奖励数会随机变化的地图中给出较好结果的智能体**。

项目默认自带了一个我训练好的参数 zip 文件, 如果想检查效果, 不修改任何配置, 直接运行如下命令即可:
```bash
python demo/train_flexible.py
```
虽然由于 requirements.txt 中已经指定了包的版本, 理论上不会出错, 但如果你遇到这个错误: ModuleNotFoundError: No module named 'numpy._core.numeric', 这是 stable-baseline3 的问题, 你必须保证保存这个模型所用的 Python 环境的 Numpy 版本和测试它的环境的 Numpy 版本保持一致。你可以通过 demo/zip_reader.py 查看一个模型保存时记录的包版本并依此修改 Python 环境来解决这个问题

为了提供一种验证逻辑, 你也可以运行下面的程序来跑一个类似上面的 baseline:
```bash
python run_flexible.py
```
但我不建议这样做, 因为这里的训练过程是很漫长的(虽然随着训练的进行, 每个 episode 的耗时会逐渐变短, 因为 agent 越来越不会走弯路了), 具体来说在 4090 上运行 250 个 episode 耗时大约一小时。我建议直接运行下面的代码来训练, **注意在 data_define.py 中修改 save_dir_name 到你想要的位置**:
```bash
python train/train_flexible.py
```
然后运行下面的命令进行测试, **注意尽量不要修改 data_define.py, 否则可能找不到模型或者参数无法对齐导致错误**:
```bash
python demo/demo_flexible.py
```