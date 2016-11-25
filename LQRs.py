#!/usr/env/bin python3

class LQR(object):
    def __init__(self,x0,x_range=(0,4)):
        self.x = x0
        self.x_range = x_range
    def update(self,a,dt):
        self.x += a*dt
        self.x = max(self.x_range[0],min(self.x_range[1],self.x))
        return self.x
    def state(self):
        return self.x
    def reward(self):
        return -(self.x**2)

class LQRs(list):
    def __init__(self,x0list,x_range=(0,4)):
        for x0 in x0list:
            self.append(LQR(x0,x_range))

    def update(self,action_list,dt):
        """
        action_list : [act0,act1,...]
        dt          : float
        """
        for i,act in enumerate(action_list):
            self[i].update(act,dt)
    def state(self):
        return [s.state() for s in self]
    def reward(self):
        return [s.reward() for s in self]
    def __str__(self):
        return str(self.state())

if __name__ == '__main__':
    lqr = LQR(1)
    lqr.update(1,1)
    print(lqr.x)

    lqrs = LQRs([1,1,1])
    lqrs.update([1,1,1],1)
    print(lqrs)
