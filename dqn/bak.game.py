# -*- encoding=utf-8 -*-
import pygame
from pygame.locals import *
import config
import sys
import numpy as np

class Game(object):
    render_ed = False
    def __init__(self):
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
        pygame.display.set_caption('Simple Game')

        self.ball_pos_x = config.SCREEN_SIZE[0] // 2 - config.BALL_SIZE[0] / 2
        self.ball_pos_y = config.SCREEN_SIZE[1] // 2 - config.BALL_SIZE[1] / 2

        self.ball_dir_x = -1  # -1 = left 1 = right
        self.ball_dir_y = -1  # -1 = up   1 = down
        self.ball_pos = pygame.Rect(self.ball_pos_x, self.ball_pos_y, config.BALL_SIZE[0], config.BALL_SIZE[1])

        self.bar_pos_x = config.SCREEN_SIZE[0] // 2 - config.BAR_SIZE[0] // 2
        self.bar_pos = pygame.Rect(self.bar_pos_x, config.SCREEN_SIZE[1] - config.BAR_SIZE[1], config.BAR_SIZE[0], config.BAR_SIZE[1])

    # action是MOVE_STAY、MOVE_LEFT、MOVE_RIGHT
    # ai控制棒子左右移动；返回游戏界面像素数和对应的奖励。(像素->奖励->强化棒子往奖励高的方向移动)
    def render(self):
        if not self.render_ed:
            self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
            pygame.display.set_caption('Simple Game')
            self.render_ed = True
        self.screen.fill(config.BLACK)
        pygame.draw.rect(self.screen, config.WHITE, self.bar_pos)
        pygame.draw.rect(self.screen, config.WHITE, self.ball_pos)
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
        state = np.asarray([self.ball_pos.left, self.ball_pos.bottom, self.bar_pos.top, self.bar_pos_x, self.ball_dir_x,
                            self.ball_dir_y])
        return state
    def step(self, action):
        if action == config.MOVE_LEFT:
            self.bar_pos_x = self.bar_pos_x - 2
        elif action == config.MOVE_RIGHT:
            self.bar_pos_x = self.bar_pos_x + 2
        else:
            pass
        if self.bar_pos_x < 0:
            self.bar_pos_x = 0
        if self.bar_pos_x > config.SCREEN_SIZE[0] - config.BAR_SIZE[0]:
            self.bar_pos_x = config.SCREEN_SIZE[0] - config.BAR_SIZE[0]

        self.screen.fill(config.BLACK)
        self.bar_pos.left = self.bar_pos_x
        pygame.draw.rect(self.screen, config.WHITE, self.bar_pos)

        self.ball_pos.left += self.ball_dir_x * 2
        self.ball_pos.bottom += self.ball_dir_y * 3
        pygame.draw.rect(self.screen, config.WHITE, self.ball_pos)

        if self.ball_pos.top <= 0 or self.ball_pos.bottom >= (config.SCREEN_SIZE[1] - config.BAR_SIZE[1] + 1):
            self.ball_dir_y = self.ball_dir_y * -1
        if self.ball_pos.left <= 0 or self.ball_pos.right >= (config.SCREEN_SIZE[0]):
            self.ball_dir_x = self.ball_dir_x * -1

        reward = 0
        if self.bar_pos.top <= self.ball_pos.bottom and (
                self.bar_pos.left < self.ball_pos.right and self.bar_pos.right > self.ball_pos.left):
            reward = 1  # 击中奖励
        elif self.bar_pos.top <= self.ball_pos.bottom and (
                self.bar_pos.left > self.ball_pos.right or self.bar_pos.right < self.ball_pos.left):
            reward = -1  # 没击中惩罚

        # 获得游戏界面像素
        screen_image = pygame.surfarray.array3d(pygame.display.get_surface())
        pygame.display.update()
        # 返回游戏界面像素和对应的奖励
        # return screen_image, reward, reward != 0
        return self.cur_state(), reward, reward != 0

    def toAction(self, act):
        if 0 == act:
            return [0, 0, 0]
        elif 1 == act:
            return [0, 1, 0]
        elif 2 == act:
            return [0, 0, 1]
        else:
            return [0, 0, 0]

    def dealEvent(self):
        for event in pygame.event.get():  # macOS需要事件循环，否则白屏
            if event.type == QUIT:
                pygame.quit()
                print("keyboard interrupt exit")
                sys.exit()