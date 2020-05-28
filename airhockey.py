import numpy as np
import pygame as pg
from gym import spaces

width = 30.4/2
height = 64.7/2
goal_width = 9.5/2
corner_rad = 5.9
eps = 0.001

def clip(vec):
    lens = vec.dot(vec)
    if lens < 1:
        return vec
    return vec / lens**0.5

class Puck:
    def __init__(self, side):
        self.pos = np.array([0., height/4 * -(side*2-1)])
        self.radius = 1.55
        self.vel = np.random.rand(2) * 30 - 15
    
    def render(self, screen):
        pg.draw.circle(screen, (0, 255, 0), (int((self.pos[0]+width)*10), int((self.pos[1]+height)*10)), int(self.radius*10))

    def step(self, dt, walls, handles):
        while dt > 0:
            update = (dt, self.pos + self.vel * dt, self.vel.copy())
            for ob in walls:
                val = ob.collision(self, dt)
                if isinstance(val, tuple):
                    if val[0] < update[0]:
                        update = val
            dt -= update[0]
            self.pos = update[1]
            self.vel = update[2]
            for h in handles:
                h.step(update[0])

class Handle:
    def __init__(self, pos, radius=0, loss=0, max_vel=100.):
        self.pos = np.array(pos, dtype=np.float)
        self.radius = radius
        self.vel = np.array([0., 0.], dtype=np.float)
        self.loss = loss
        self.reward = 0
        self.max_vel = max_vel

    def set_vel(self, pos, dt):
        velocity = (pos - self.pos) / dt
        speed = np.sqrt(velocity.dot(velocity))
        if speed > self.max_vel:
            velocity = velocity * (self.max_vel / speed)
        dif = velocity-self.vel
        #self.reward -= dif.dot(dif)**0.5 / (self.max_vel*500) + self.vel.dot(self.vel)**0.5*dt / (self.max_vel*100)
        self.vel = velocity

    def step(self, dt):
        self.pos += self.vel * dt

    def get_reward(self):
        r = self.reward
        self.reward = 0
        return r

    def render(self, screen):
        if self.radius > 0:
            pg.draw.circle(screen, (255, 0, 0), (int((self.pos[0]+width)*10), int((self.pos[1]+height)*10)), int(self.radius*10))

    def collision(self, c, dt):
        posp = c.pos - self.pos
        velp = c.vel - self.vel
        r = c.radius + self.radius
        if velp[0] == 0 and velp[1] == 0:
            return 0
        a = velp.dot(velp)
        b = 2*(posp.dot(velp))
        cc = posp.dot(posp)-r**2

        det = b**2 - 4*a*cc
        if det < 0:
            return 0
        t = (-b - det**0.5)/a/2
        if t < 0 or t > dt:
            return 0 
        
        posp = posp + velp*(t-eps)
        nposp = posp / r
        vel_pdir = velp.dot(nposp) * nposp
        velp = velp + vel_pdir * (-2+self.loss)
        return (t, c.pos + c.vel*(t-eps), velp+self.vel)

class Wall:
    def __init__(self, st, ed, loss=0.2):
        self.xs = 1 if abs(st[0] - ed[0]) < 0.001 else 0
        self.loss = loss
        self.d = st[-(self.xs-1)]
        self.st = st[self.xs]
        self.ed = ed[self.xs]

        self.sta = [int((st[0] + width) * 10), int((st[1] + height) * 10)]
        self.eda = [int((ed[0] + width) * 10), int((ed[1] + height) * 10)]

    def render(self, screen):
        pg.draw.line(screen, (255, 255, 255), self.sta, self.eda, 3)

    def collision(self, c, dt):
        flip = -(self.xs - 1)
        pos = np.copy(c.pos)
        vel = np.copy(c.vel)
        if pos[flip] > self.d:
            r = -c.radius
        else:
            r = c.radius
        
        if vel[flip] == 0:
            return 0

        t = (self.d - (pos[flip]+r)) / vel[flip]
        if t < 0 or t > dt:
            return 0
        pos += vel * (t-eps)
        if pos[self.xs] < self.st or pos[self.xs] > self.ed:
            return 0
        vel[flip] = -vel[flip] * (1-self.loss)
        return (t, pos, vel)

class Corner:
    def __init__(self, pos, st, ed, radius=corner_rad):
        self.pos = np.array(pos)
        self.radius = radius
        self.st = st
        self.ed = ed
    
    def render(self, screen):
        pg.draw.arc(screen, (255, 255, 255), (int((self.pos[0]-self.radius+width)*10), int((self.pos[1]-self.radius+height)*10), int(self.radius*20), int(self.radius*20)), -self.ed, -self.st, 3)
    
    def clip(self, pos, radius):
        posp = pos - self.pos
        r = self.radius - radius
        theta = np.arctan2(posp[1], posp[0])
        if theta >= self.st and theta <= self.ed:
            l = posp.dot(posp)**0.5
            if l > r:
                posp = posp * (r/l)
                return posp + self.pos
        return pos


    def collision(self, c, dt):
        posp = c.pos - self.pos
        vel = np.copy(c.vel)
        r = self.radius - c.radius
        if vel[0] == 0 and vel[1] == 0:
            return 0
        a = vel.dot(vel)
        b = 2*(posp.dot(vel))
        cc = posp.dot(posp)-r**2

        det = b**2 - 4*a*cc
        if det < 0:
            return 0
        t = (-b + det**0.5)/a/2
        if t < 0 or t > dt:
            return 0 
        posp = posp + vel*t
        theta = np.arctan2(posp[1], posp[0])
        if theta < self.st or theta > self.ed:
            return 0
        nposp = posp / r
        vel_pdir = vel.dot(nposp) * nposp
        vel = vel + vel_pdir * (-2)
        return (t, c.pos + c.vel*(t-eps), vel)

class AirHockey:
    def __init__(self, render=False, human=False):
        if render:
            pg.init()
            self.clock = pg.time.Clock()
            self.mouse = pg.mouse
            size = [1000,800]
            self.screen = pg.display.set_mode(size)
            pg.display.set_caption("AirHockey")

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,))
        self.h1 = Handle(np.array([0., 30.]), radius=1.96, loss=0.75, max_vel=100. if human else 100.)
        self.h2 = Handle(np.array([0., -30.]), radius=1.96, loss=0.75)
        self.handles = [self.h1, self.h2]
        self.puck = Puck(np.random.randint(2))
        self.walls = self.handles.copy()
        self.walls.append(Wall((-width, -height), (-goal_width, -height)))
        self.walls.append(Wall((goal_width, -height), (width, -height)))
        self.walls.append(Wall((-width, height), (-goal_width, height)))
        self.walls.append(Wall((goal_width, height), (width, height)))
        self.walls.append(Wall((-width, -height), (-width, height)))
        self.walls.append(Wall((width, -height), (width, height)))
        self.walls.append(Handle((-goal_width, -height)))
        self.walls.append(Handle((-goal_width, height)))
        self.walls.append(Handle((goal_width, -height)))
        self.walls.append(Handle((goal_width, height)))
        hpi = np.pi / 2
        self.corners = list()
        self.corners.append(Corner((width-corner_rad, height-corner_rad), 0, hpi))
        self.corners.append(Corner((-width+corner_rad, height-corner_rad), hpi, hpi*2))
        self.corners.append(Corner((-width+corner_rad, -height+corner_rad), -hpi*2, -hpi))
        self.corners.append(Corner((width-corner_rad, -height+corner_rad), -hpi, 0))
        self.walls.extend(self.corners)

        self.r = 0


    def getstate(self):
        wh = np.array((width, height))
        state = list()
        for ob in (self.h1, self.h2, self.puck):
            state.extend(ob.pos / wh)
            state.extend(ob.vel / self.h1.max_vel) 
        s1 = np.array(state, dtype=np.float32)

        wh = -wh
        state = list()
        for ob in (self.h2, self.h1, self.puck):
            state.extend(ob.pos / wh)
            state.extend(ob.vel / (-self.h2.max_vel)) 
        s2 = np.array(state, dtype=np.float32)
        return s1, s2

    def reset(self):
        self.puck = Puck(np.random.randint(2))
        self.h1.pos = np.array([0., 30.])
        self.h2.pos = np.array([0., -30.])
        s1, s2 = self.getstate()
        return s1, s2
    def render(self):
        pg.event.get()
        self.screen.fill((0,0,0))
        self.puck.render(self.screen)
        for w in self.walls:
            w.render(self.screen)
        pg.display.flip()
        self.clock.tick(60)

    def step(self, p1, p2, dt):
        p1 = p1*(width-self.h1.radius, height/2-self.h2.radius)+(0, height/2)
        p2 = p2*(-width+self.h2.radius, -height/2+self.h2.radius)-(0, height/2)
        #p1 = clip(p1)
        #p2 = -clip(p2)
        #p1 = np.clip(p1*self.h1.max_vel*dt + self.h1.pos, (-width+self.h1.radius, self.h1.radius*2), (width-self.h1.radius, height-self.h1.radius))
        #p2 = np.clip(p2*self.h2.max_vel*dt + self.h2.pos, (-width+self.h2.radius, -height+self.h2.radius), (width-self.h2.radius, -self.h2.radius*2))
        for corner in self.corners:
            p1 = corner.clip(p1, self.h1.radius)
            p2 = corner.clip(p2, self.h2.radius)
        self.h1.set_vel(p1, dt)
        self.h2.set_vel(p2, dt)
        prev_neg = self.puck.pos[1] < 0
        self.puck.step(dt, self.walls, self.handles)
        self.r = 0
        done = False
        if self.puck.pos[1]+self.puck.radius < -height:
            self.r = 1
            done = True
        elif self.puck.pos[1]-self.puck.radius > height:
            self.r = -1
            done = True
        elif not prev_neg and self.puck.pos[1] < 0:
            self.r = 0.05
        elif prev_neg and self.puck.pos[1] > 0:
            self.r = -0.05
        s1, s2 = self.getstate()
        return s1, s2, self.r, -self.r, done

    def close(self):
        pg.quit()

def main():
    hockey = AirHockey(render=True, human=True)
    done= False
    
    while True:
        pos = np.array(hockey.mouse.get_pos())
        wh = np.array((width, height/2)) * 10
        s1, s2, r1, r2, d = hockey.step(np.clip((pos-wh-(0, height*10))/wh, -1, 1), np.clip((pos-wh-(0, height*10))/wh, -1, 1), 1/60)
        if r1 != 0 or r2 != 0:
            print(f"{r1}\t{r2}")
        hockey.render()
        if d:
            hockey.reset()
if __name__ == "__main__":
    main()