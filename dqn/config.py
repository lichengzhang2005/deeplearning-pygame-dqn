# -*- encoding=utf-8 -*-
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

SCREEN_SIZE = [160, 200]
BAR_SIZE = [30, 3]
BALL_SIZE = [9, 9]

# 神经网络的输出
MOVE_STAY = [1, 0, 0]
MOVE_LEFT = [0, 1, 0]
MOVE_RIGHT = [0, 0, 1]

# learning_rate
LEARNING_RATE = 0.99
# 更新梯度
INITIAL_EPSILON = 1.0 # 0.5
FINAL_EPSILON = 0.1 # 0.05
# 测试观测次数
EXPLORE = 500000 # 500000
OBSERVE = 10000 # 50000
# 存储过往经验大小
REPLAY_MEMORY = 500000 # 500000

BATCH = 32

GAMMA = 0.99 # decay rate of past observations

'''
版本1：
1、初始化replay memory D 容量为N

2、用一个深度神经网络作为Q值网络，初始化权重参数

3、设定游戏片段总数M

4、初始化网络输入，大小为84*84*4，并且计算网络输出

5、以概率ϵ随机选择动作at或者通过网络输出的Q（max）值选择动作at

6、得到执行at后的奖励rt和下一个网络的输入

7、根据当前的值计算下一时刻网络的输出

8、将四个参数作为此刻的状态一起存入到D中（D中存放着N个时刻的状态）

9、随机从D中取出minibatch个状态

10、计算每一个状态的目标值（通过执行at后的reward来更新Q值作为目标值）

11、通过SGD更新weight

下在的是2015年的版本2：
1、初始化replay memory D，容量是N 用于存储训练的样本

2、初始化action-value function的Q卷积神经网络 ，随机初始化权重参数θ

3、初始化 target action-value function的Q^卷积神经网络，结构以及初始化权重θ和Q相同

4、设定游戏片段总数M

5、初始化网络输入，大小为84*84*4，并且计算网络输出

6、根据概率ϵ（很小）选择一个随机的动作或者根据当前的状态输入到当前的网络中 （用了一次CNN）计算出每个动作的Q值，选择Q值最大的一个动作（最优动作）

7、得到执行at后的奖励rt和下一个网络的输入

8、将四个参数作为此刻的状态一起存入到D中（D中存放着N个时刻的状态）

9、随机从D中取出minibatch个状态

10、计算每一个状态的目标值（通过执行at后的reward来更新Q值作为目标值）

11、通过SGD更新weight

12、每C次迭代后更新target action-value function网络的参数为当前action-value function的参数

参考文献：

    一个 Q-learning 算法的简明教程
    如何用简单例子讲解 Q - learning 的具体过程
    《Code for a painless q-learning tutorial》以及百度网盘地址
    DQN 从入门到放弃4 动态规划与Q-Learning
    DQN从入门到放弃5 深度解读DQN算法
    Deep Reinforcement Learning 基础知识（DQN方面）
    Paper Reading 1 - Playing Atari with Deep Reinforcement Learning
    Playing Atari with Deep Reinforcement Learning 论文及翻译百度网盘地址
    Paper Reading 2:Human-level control through deep reinforcement learning
    Human-level control through deep reinforcement learning 论文及翻译百度网盘地址
    重磅 | 详解深度强化学习，搭建DQN详细指南（附论文）
    Playing Atari with Deep Reinforcement Learning算法解读

'''