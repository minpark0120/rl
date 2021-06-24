import random
import numpy as np
from math import *

DEFAULT_ANGLE = 60
SRC = [sqrt(3), 1]
DEST = [0, 4]
ANGLE1_CORD= [0,2]
ANGLE2_CORD= [sqrt(3), 3]
ANGLE3_CORD= [2*sqrt(3), 2]
DONE_ANGLE = 30
#TRANSITION_PROB =1 
#POSSIBLE_ACTIONS = [0, 1, 2]  # angle1, angle2, angle3

class Geo():
    def __init__(self):
        self.default_angle = DEFAULT_ANGLE
        self.src_x = SRC[0]
        self.src_y = SRC[1]
        self.dest_x = DEST[0]
        self.dest_y = DEST[1]
        self.angle1_cord= ANGLE1_CORD
        self.angle2_cord= ANGLE2_CORD
        self.angle3_cord= ANGLE3_CORD
        self.angle1 = self.get_angle(ANGLE1_CORD)
        self.angle2 = self.get_angle(ANGLE2_CORD)
        self.angle3 = self.get_angle(ANGLE3_CORD)

        #self.transition_probability = TRANSITION_PROB
        #self.possible_actions = POSSIBLE_ACTIONS
        #self.reward[0] = angle1_action 
        #self.reward[1] = angle2_action
        #self.reward[2] = angle3_action

    def get_angle(self, pos):
        pos_x = pos[0]
        pos_y = pos[1]
        in_vector = sqrt((pos_x - self.src_x)*(pos_x - self.src_x) + (pos_y - self.src_y)*(pos_y - self.src_y))
        out_vector = sqrt((self.dest_x- self.src_x)*(self.dest_x- self.src_x) + (self.dest_y - self.src_y)*(self.dest_y - self.src_y))
        cross_vector = sqrt((self.dest_x- pos_x)*(self.dest_x- pos_x) + (self.dest_y - pos_y)*(self.dest_y - pos_y))
        angle = acos(((in_vector*in_vector + out_vector*out_vector) - cross_vector*cross_vector)/(2*in_vector*out_vector))
        angle = fabs(angle * 180.0 / 3.1416)
        return angle

    def get_angle_action(self, angle):
        if self.default_angle > angle:
            self.angle_action = (self.default_angle - angle)/self.default_angle
        else:
            self.angle_action = 0

        return self.angle_action 

    def step(self, a):
        print('step')
        print(self.default_angle)
        if a==0:
            print('04')
            self.default_angle = self.default_angle - self.get_angle_action(self.angle1)

        elif a==1:
            print('root3')
            self.default_angle = self.default_angle - self.get_angle_action(self.angle2)

        elif a==2:
            print('root6')
            self.default_angle = self.default_angle - self.get_angle_action(self.angle3)


        reward = -1  # 보상은 항상 -1로 고정
        done = self.is_done()
        return self.default_angle, reward, done
        
    def is_done(self):
        if self.default_angle <= DONE_ANGLE:
            return True
        else:
            return False
      
    def reset(self):
        self.default_angle = 60
        return self.default_angle

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((121,3)) # 마찬가지로 Q 테이블을 0으로 초기화
        self.eps = 0.9
        self.action = 3

    def select_action(self, s):
        # eps-greedy로 액션을 선택해준다

        x = int(s)
        coin = random.random()
        if coin < self.eps:
            self.action = random.randint(0,2)
            print(self.action)
        else:
            action_val = self.q_table[x,self.action]
            self.action = np.argmax(action_val)
        
        return self.action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x = int(s)
        next_x = int(s_prime)
        print(s_prime)
        a_prime = self.select_action(s_prime) # S'에서 선택할 액션 (실제로 취한 액션이 아님)
        # SARSA 업데이트 식을 이용
        self.q_table[x,a] = self.q_table[x,a] + 0.1 * (r + self.q_table[next_x, a_prime] - self.q_table[x,a])

    def anneal_eps(self):
        self.eps -= 0.03
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        q_list = self.q_table.tolist()
        data = np.zeros((121))
        for row_idx in range(len(q_list)):
            row = q_list[row_idx]
            action = np.argmax(row)
            data[row_idx] = action
        print(data)


def main():
    env = Geo()
    agent = QAgent()

    for n_epi in range(20):
        done = False

        s = env.reset()
        count = 0
        while not done:
            count = count +1 
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime))
            s = s_prime
            print(count)
        agent.anneal_eps()
    agent.show_table()


if __name__ == '__main__':
    main()
