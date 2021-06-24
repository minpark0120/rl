import random
import numpy as np
from math import *

DEFAULT_ANGLE = 120
SRC = [sqrt(3), 1]
DEST = [0, 4]
ANGLE1_CORD= [0,2]
ANGLE2_CORD= [sqrt(3), 3]
ANLGE3=CORD= [2*sqrt(3), 2]
TRANSITION_PROB =1 
POSSIBLE_ACTIONS = [0, 1, 2]  # angle1, angle2, angle3

class Geo():
    def __init__(self):
        self.src_x = SRC[0]
        self.src_y = SRC[1]
        self.des_x = DEST[0]
        self.des_y = DEST[1]
        self.angle1_cord= ANGLE1_CORD
        self.angle2_cord= ANGLE2_CORD
        self.angle3_cord= ANGLE3_CORD
        self.transition_probability = TRANSITION_PROB
        self.possible_actions = POSSIBLE_ACTIONS
        self.action = [angle1_action, angle2_action, angle3_action]
        self.reward[0] = angle1_action 
        self.reward[1] = angle2_action
        self.reward[2] = angle3_action
        self.default_angle = DEFAULT_ANGLE

    def step(self, a):
        angle1 = 
        angle2 = 
        angle3 = 
        if a==0:
            self.angle_action(angle1_action)
        elif a==1:
            self.angle_action(angle2)
        elif a==2:
            self.angle_action()

        reward = -1  # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def get_angle(self, pos):
        pos_x = pos[0]
        pos_y = pos[1]
        in_vector = sqrt((pos_x - src_x)*(pos_x - src_x) + (pos_y - src_y)*(pos_y - src_y))
        out_vector = sqrt((dest_x- src_x)*(dest_x- src_x) + (dest_y - src_y)*(dest_y - src_y))
        cross_vector = sqrt((dest_x- pos_x)*(dest_x- pos_x) + (dest_y - pos_y)*(dest_y - pos_y))
        angle = acos(((in_vector*in_vector + out_vector*out_vector) - cross_vector*cross_vector)/(2*in_vector*out_vector))
        angle = fabs(angle * 180.0 / 3.1416)
        return angle

    def angle_action(angle):
        if default_angle > angle:
            angle_action = (default_angle - angle)/default_angle
        else:
            angle_action = 0

        return angle_action 
        
    def is_done(self):
        if self.default_angle== 60
            return True
        else:
            return False
      
    def reset(self):
        self.default_angle = 120
        return self.default_angle

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((10,)) # q벨류를 저장하는 변수. 모두 0으로 초기화. 
        self.eps = 0.9 
        self.alpha = 0.01
        
    def select_action(self, s):
        # eps-greedy로 액션을 선택
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,2)
        else:
            action_val = self.q_table[x,y,:]
            action = np.argmax(action_val)
        return action

    def update_table(self, history):
        # 한 에피소드에 해당하는 history를 입력으로 받아 q 테이블의 값을 업데이트 한다
        cum_reward = 0
        for transition in history[::-1]:
            s, a, r, s_prime = transition
            x,y = s
            # 몬테 카를로 방식을 이용하여 업데이트.
            self.q_table[x,y,a] = self.q_table[x,y,a] + self.alpha * (cum_reward - self.q_table[x,y,a])
            cum_reward = cum_reward + r 

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        # 학습이 각 위치에서 어느 액션의 q 값이 가장 높았는지 보여주는 함수
        q_lst = self.q_table.tolist()
        data = np.zeros((5,7))
        for row_idx in range(len(q_lst)):
            row = q_lst[row_idx]
            for col_idx in range(len(row)):
                col = row[col_idx]
                action = np.argmax(col)
                data[row_idx, col_idx] = action
        print(data)
      
def main():
    env = Geo()
    agent = QAgent()

    for n_epi in range(1000): # 총 1,000 에피소드 동안 학습
        done = False
        history = []

        s = env.reset()
        while not done: # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            history.append((s, a, r, s_prime))
            s = s_prime
        agent.update_table(history) # 히스토리를 이용하여 에이전트를 업데이트
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()
