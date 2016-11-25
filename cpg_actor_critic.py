#!/usr/env/bin python3
from actor_critic import actor_critic
from cpg import cpg

class cpg_actor_critic(actor_critic):
    def __init__(self,inout,A,alpha=0.05,gamma=0.95,**cpg_args):
        inout_ = (inout[0]+inout[1],inout[1])
        super().__init__(inout_,alpha=alpha,gamma=gamma)
        self.CPG = cpg(inout[1],A,**cpg_args)

        self.lastOutput = np.zeros((1,inout[1]))
    def __call__(self,state,reward,dt):
        #print(state,self.lastOutput)
        state_ = np.c_[state,self.lastOutput]
        self.act,self._ = self.action(state_,reward,dt)
        self.lastOutput = np.array([self.CPG(dt,self.act)])
        return self.lastOutput

if __name__ == '__main__':
    import numpy as np
    inout = (2,2)
    a = 2.5
    A = a - a*np.tri(inout[1])
    A += A.T
    cpg_ac = cpg_actor_critic(inout,A,x0=[1,2,0,0],alpha=0.15)
    state = np.ones((1,inout[0]))
    reward = 1
    dt = 0.01
    tmax = 2000

    def state(timelist):
        return np.maximum(np.array(np.sin(timelist))+1,0)
    def reward(timelist,value):
        return 2-sum(np.power(value - np.sin(timelist),2)[0])
    lastact = np.array([[0,0]])

    from save_plot import logger
    import os

    for i in range(10):
        print(i)
        z = np.arange(i*tmax,(i+1)*tmax,dt)
        t = [z,z+3.14]
        lgr = logger(['act','output','state','reward','TDerr'])

        for t0,t1 in zip(t[0],t[1]):
            tl = [[t0,t1]]
            s = state(tl)
            r = reward(tl,lastact)
            lastact = cpg_ac(s,r,dt)
            lgr.append([cpg_ac.act[0],lastact[0],s[0],r,cpg_ac._[0]])

        lgr.output(os.path.expanduser('~')+'/Pictures/log/cpg_actor_critic_test',t[0],
        title='serial #'+str(i)+'\nstate=$[sin(t),sin(t+\pi)]$ reward=$n-\sum state$\n$ \\alpha,\gamma,b,T = default$')
