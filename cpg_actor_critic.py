#!/usr/env/bin python3
from actor_critic import actor_critic
from cpg import cpg

class cpg_actor_critic(actor_critic):
    def __init__(self,inout,A,alpha=0.05,gamma=0.95,**cpg_args):
        i,o = inout
        super().__init__((i+o,i),alpha=alpha,gamma=gamma)
        self.CPG = cpg(o,A,**cpg_args)

        self.lastOutput = np.zeros(o)
    def __call__(self,state,reward,dt):
        #print('cpg_ac',state,self.lastOutput)
        state_ = np.r_[state,self.lastOutput]
        self.act,self._ = self.action(state_,reward,dt)
        #print('cpg_ac.act',self.act)
        self.lastOutput = np.array(self.CPG(dt,self.act[0]))
        return self.lastOutput

class sin_env(object):
    frequence = 0.1
    def state(self,timelist):
        t = np.array(timelist)
        return np.maximum(np.sin(self.frequence*t),0)+1
    def reward(self,timelist,value):
        t = np.array(timelist)
        return 2-sum(np.power(value - np.sin(self.frequence*t),2))

if __name__ == '__main__':
    import numpy as np
    inout = (2,2)
    a = 2.5
    A = a - a*np.tri(inout[1])
    A += A.T
    cpg_ac = cpg_actor_critic(inout,A,x0=[1,2,0,0],alpha=0.15)
    reward = 1
    dt = 0.01
    tmax = 20

    lastact = np.array([[0,0]])
    se = sin_env()

    from save_plot import logger
    import os

    for i in range(1):
        print(i)
        z = np.arange(i*tmax,(i+1)*tmax,dt)
        t = [z,z+3.14]
        lgr = logger(['act','output','state','reward','TDerr'])

        for t0,t1 in zip(t[0],t[1]):
            print(t0)
            tl = [t0,t1]
            s = se.state((t0,t1))
            #print('state',s,t0)
            r = se.reward(tl,lastact[0])
            #print('reward',r)
            lastact = cpg_ac(s,r,dt)
            lgr.append([cpg_ac.act[0],lastact[0],s[0],r,cpg_ac._[0]])

        lgr.output(os.path.expanduser('~')+'/Pictures/log/cpg_actor_critic_test',t[0],
        title='serial #'+str(i)+'\nstate=$[sin(t),sin(t+\pi)]$ reward=$n-\sum state$\n$ \\alpha,\gamma,b,T = default$')
