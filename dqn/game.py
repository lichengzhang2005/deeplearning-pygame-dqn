# -*- encoding=utf-8 -*-
import pygame
import random
from pygame.locals import *
import numpy as np
import sys

BLACK     = (0  ,0  ,0  )
WHITE     = (255,255,255)
SCREEN_SIZE = [60,110]
BAR_SIZE = [10, 8]
BALL_SIZE = [8, 8]
#  神经网络的输出
MOVE_STAY = 0 #[1, 0, 0]
MOVE_LEFT = 1 #[0, 1, 0]
MOVE_RIGHT = 2 #[0, 0, 1]
class Game(object):
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.render_ed = False
        self.ball_pos_x = SCREEN_SIZE[0]//2 - BALL_SIZE[0]/2
        self.ball_pos_y = SCREEN_SIZE[1]//2 - BALL_SIZE[1]/2
        self.ball_dir_x = -1 # -1 = left 1 = right  
        self.ball_dir_y = -1 # -1 = up   1 = down
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
        self.bar_pos_x = SCREEN_SIZE[0]//2-BAR_SIZE[0]//2
        self.bar_pos = pygame.Rect(self.bar_pos_x, SCREEN_SIZE[1]-BAR_SIZE[1], BAR_SIZE[0], BAR_SIZE[1])
        topl_x = SCREEN_SIZE[0] // 3
        topl_y = 8
        topl_sx = SCREEN_SIZE[0] // 3
        topl_sy = 3

        leftl_x = 8
        leftl_y = SCREEN_SIZE[1] // 3
        leftl_sx = 2
        leftl_sy = SCREEN_SIZE[1] // 3
        self.topl_pos = pygame.Rect(topl_x, topl_y, topl_sx, topl_sy)
        self.leftl_pos = pygame.Rect(leftl_x, leftl_y, leftl_sx, leftl_sy)
        #self.leftl_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, BALL_SIZE[0], BALL_SIZE[1])
    # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # ai控制棒子左右移动；返回游戏界面像素数和对应的奖励。(像素->奖励->强化棒子往奖励高的方向移动)
    def render(self):
        if not self.render_ed:
            self.screen = pygame.display.set_mode(SCREEN_SIZE)
            pygame.display.set_caption('Simple Game')
            self.render_ed = True
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, WHITE, self.bar_pos)
        pygame.draw.rect(self.screen, WHITE, self.ball_pos)
        '''
        if self.ball_dir_y == -1:
            pygame.draw.rect(self.screen, WHITE, self.topl_pos)
        if self.ball_dir_x == -1:
            pygame.draw.rect(self.screen, WHITE, self.leftl_pos)
        '''
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        return screen_image
    def cur_state(self):
        state = np.asarray([self.ball_pos.left, self.ball_pos.bottom, self.bar_pos.top, self.bar_pos_x, self.ball_dir_x, self.ball_dir_y])
        return state
    def step(self, action):
        if action == MOVE_LEFT:
            self.bar_pos_x = self.bar_pos_x - 2
        elif action == MOVE_RIGHT:
            self.bar_pos_x = self.bar_pos_x + 2
        else:
            pass
        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
        if self.bar_pos_x > SCREEN_SIZE[0] - BAR_SIZE[0]:
            self.bar_pos_x = SCREEN_SIZE[0] - BAR_SIZE[0]
        self.bar_pos.left = self.bar_pos_x
        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 3
        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (SCREEN_SIZE[1] - BAR_SIZE[1]+1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= (SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1
        reward = 0
        if self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
            reward = 100    # 击中奖励
        elif self.bar_pos.top <= self.ball_pos.bottom and (self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
            reward = -100   # 没击中惩罚
        # 返回游戏界面像素和对应的奖励
        return self.cur_state(), reward, reward != 0, None

    def dealEvents(self):
        for event in pygame.event.get():  # macOS需要事件循环，否则白屏
            if event.type == QUIT:
                pygame.quit()
                print("keyboard interrupt exit")
                sys.exit()
#  定义CNN-卷积神经网络 参考:http://blog.topspeedsnail.com/archives/10451
