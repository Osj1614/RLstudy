import gym
import random

class RobotEnv(gym.Env):
    def __init__(self):
        self.dir = 0 #0 Left, 1 Up, 2 Right, 3 Down
        self.iscarry = 0 #0 No, 1 Yes
        self.grid_size = 4
        self.x = 0
        self.y = 0
        self.ox = 0
        self.oy = 0
        self.tx = 0
        self.ty = 0
        
        self.step_count = 0
        self.prev_action = 0

        self.observation_space = gym.spaces.Discrete(4 * 2 * self.grid_size**6) #4*2*4*4*4*4*4*4
        self.action_space = gym.spaces.Discrete(5) #Move forward, turn left, turn right, pick up, put down
        self.action_names = ['Move', 'Turn left', 'Turn right', 'Pick up', 'Put down']

    def get_obs(self):
        val = self.dir * 2 + self.iscarry
        for ob in [self.x, self.y, self.ox, self.oy, self.tx, self.ty]:
            val = val * self.grid_size + ob
        return val

    def reset(self):
        self.dir = random.randint(0, 3)
        self.iscarry = 0
        self.x = random.randint(0, self.grid_size-1) 
        self.y = random.randint(0, self.grid_size-1)
        self.ox = random.randint(0, self.grid_size-1)
        self.oy = random.randint(0, self.grid_size-1)
        self.tx = random.randint(0, self.grid_size-1)
        self.ty = random.randint(0, self.grid_size-1)

        self.step_count = 0

        return self.get_obs()

    def move_robot(self):
        if self.dir == 0 and self.x < self.grid_size-1:
            self.x += 1
        elif self.dir == 1 and self.y < self.grid_size-1:
            self.y += 1
        elif self.dir == 2 and self.x > 0:
            self.x -= 1
        elif self.dir == 3 and self.y > 0:
            self.y -= 1

    def step(self, action):
        self.step_count += 1
        self.prev_action = action

        reward = -1
        done = False
        
        if action == 0:
            self.move_robot()
        elif action == 1:
            self.dir = (self.dir+1)%4
        elif action == 2:
            self.dir = (self.dir+3)%4
        elif action == 3:
            if self.iscarry == 0 and self.x == self.ox and self.y == self.oy:
                self.iscarry = 1
        elif action == 4:
            if self.iscarry == 1:
                self.iscarry = 0
                self.ox = self.x
                self.oy = self.y
        
        if self.ox == self.tx and self.oy == self.ty:
            reward = 100
            done = True
        
        if self.step_count > 1000:
            done = True

        return self.get_obs(), reward, done, {}

    def render(self, mode='human'):
        if self.iscarry == 0:
            robot_shape = ['▷', '△', '◁', '▽']
        else:
            robot_shape = ['▶', '▲', '◀', '▼']

        for y in reversed(range(self.grid_size)):
            line = ''
            for x in range(self.grid_size):
                if x == self.x and y == self.y:
                    line += robot_shape[self.dir]
                elif self.iscarry == 0 and x == self.ox and y == self.oy:
                    line += '●'
                elif x == self.tx and y == self.ty:
                    line += '◎'
                else:
                    line += '□'
            print(line)
        print(self.action_names[self.prev_action])