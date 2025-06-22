# RL 实验报告
23307130104 何欣喆

## 1. 实验目的
1. 学习强化学习的基础知识;
2. 强化 pytorch, gym 的编程技能。

## 2. 实验环境
在 Ubuntu20.04 系统下, 使用 python 包 pytorch 和 gym 的功能完成项目。

## 3. 基础知识, 算法思路和技术实现
### 基础知识
强化学习由两个可以交互的对象组成: 智能体 Agent 和环境 Environment(下记为 env)。
描述强化学习有以下基本要素: 
状态 $s$: 对 env 的描述;
动作 $\alpha$: 对 agent 行为的描述, 其动作空间为 $A$;
策略 $\pi(\alpha|s)$: agent 根据 $s$ 来决定下一步动作 $a$ 的函数;
状态转移概率 $p(s'|s,a)$: agent 根据当前状态 $s$ 做出一个动作 $a$ 之后, env 在下一个时刻转为状态 $s'$ 的概率;
即时奖励 $r(s,a,s')$: 一个标量函数, 即 agent 做出动作后环境给 agent 反馈的奖励, 其往往和下一个时刻的状态 $s'$ 有关;
策略: agent 如何根据 $s$ 来决定下一步的动作 $a$。

### 算法思路
#### 训练算法
Q 学习(Q-Learning) 算法是一种异策略的时序差分方法。这里的 Q 指的是 Q 函数是状态-动作值函数:
$$
Q^{\pi}(s, a) = \mathbb{E}_{s' \sim p(s'|s,a)} \left[ r(s, a, s') + \gamma V^{\pi}(s') \right]
$$
表示 agent 在初始状态 $s$ 执行动作 $a$ 后执行策略 $\pi$ 得到的期望总汇报。
所以, 最容易想到的训练 agent 的方法就是让其策略 $\pi$ 贴近于选择最好的 Q 函数:
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$
总结下来, 理论上智能体在一个有解的环境下进行足够多轮, 学习率设置合理的 Q 学习总是能找到最优解的。由于我们的项目环境规模很小, 状态空间离散, 轻量化的 Q 学习非常合适。

#### 探索思路
我们必须想办法避免模型一条路走到底, 也要帮助模型脱离局部最优解。
最基础, 最简单, 但是在离散环境下非常有效的探索方法就是 $\epsilon$-greedy。其引入变量 $\epsilon$, 每次选取动作 $a$ 时, 有 $\epsilon$ 的概率随机选择一个动作进行探索, $1-\epsilon$ 的概率选择策略所决定的动作。

### 技术实现
#### 自定义 env
直接用 Gymnasium 提供的环境没有意思, 也不完全贴合项目的需求, 自定义 env 是必须的。
OpenAI 的 Gym library 提供了一种自定义 RL 环境的方法。它提供了一个类 gym.Env, 我们可以继承它来定义自己的 RL 环境。

总的来说, 我们需要完成/修改以下 method 的定义:
1. 空间定义: __init__ 中需要定义 obs, action 的空间, 这对于自定义特征传输网络以及增强代码健壮性很重要; 
2. step(self, action): 描述模型执行一个 action 进入到下一个时刻的行为。这个函数返回 obs, reward, done, extra_information, 它是 agent 与 env 交互的窗口;
3. reset(self): 很显然我们需要多次从头运行 env。reset 提供重置 env 和 agent 的方法;
4. 其他自定义接口, 如 get_position(self), _deploy_action(self, action) 等等。

我们的项目实现起来还是很简单的, 实现细节参见 envs.py。下面讲一讲我在自定义时选择一些方法的原因:

**observation_space 的设置**
直观来看, obs 应该由两部分组成: agent_pos 和 score_map:
```python
self.observation_space = sapces.Dict({
    'agent_pos': spaces.Discrete(size_of_grid),
    'score_map': spaces.Box(low=-np.inf, high=np.inf, shape=grid_size, dtype=np.uint32)

})
```
然后通过一个 CNN 和一个全连接层把二者连接起来进行训练和推理。这很有道理, 似乎也能让模型更聪明, 但是处理 score_map 的 CNN 会带来很多额外的参数(相对仅处理 agent_pos 而言), 大大拖慢训练速度。而且, 这些参数是无意义的, 因为 score_map 根本就不会变化。此外, score_map 的不变性容易导致 obs 高度相似, 非常容易出现 agent 一个动作执行到底的情况(本质上就是 agent_pos 的影响被 score_map 的影响覆盖了)。
所以综上, observation_space 仅包含 agent_pos 了。但前面的定义方法在训练能灵活迁移到不同环境的 agent 时很有效, 这我会在报告的最后介绍。

**Network Definition**
有了上面的分析, 这个问题本质上就是一个四分类, 体量小, 状态离散, 重要性对等, 输入类型单一, 输出离散且规模小, 那么网络的选择就很明确了: 全连接神经网络/多层感知器能很好地处理它们:
```python
class QNetwork(nn.Module):
    def __init__(self, n_states: int, n_actions: int, embed_dim=64):
        super(QNetwork, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=n_states, embedding_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, n_actions)
        
    def forward(self, obs: torch.LongTensor) -> torch.Tensor:
        x = self.embedding(obs)         # (batch_size, embed_dim)
        x = torch.relu(self.fc1(x))     # (batch_size, 128)
        q_values = self.fc2(x)          # (batch_size, n_actions)
        return q_values
```
由于结构简单, 我直接把它放在了 train.py 中而非再开一个 python 文件单独维护它。

**优化方法**
虽然这种小规模的, 离散的问题看起来完全用不上 DQN, 但用不上≠不好用, 实测深度 Q-learning 效果是最好的, 实现也很容易, 差不多 40K 的运行起来也很快。
自定义的简单的优化模型见 train.py line.76 的 optimize_model。

## 4. 实验效果
RL 本质上就是最大化 reward, 所以模型的行为理想下仅由 reward 的定义方式决定。
### 
