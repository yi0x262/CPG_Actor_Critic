#!/usr/env/bin python2
from actor_critic import actor_critic
from cpg import cpg

import numpy as np
class cpg_actor_critic(actor_critic,object):
    def __init__(self,(i,o),A,alpha=0.05,gamma=0.95,**cpg_args):
        super(cpg_actor_critic,self).__init__((i+o,2*o),alpha=alpha,gamma=gamma)
        self.CPG = cpg(o,A,**cpg_args)

        self.lastCPGout = np.zeros((1,o))
    def __call__(self,state,reward,dt):
        """
        state : flatten(ndim=1) list or ndarray
        """
        #print state
        #print self.lastCPGout
        s = np.array(np.concatenate((state,self.lastCPGout),axis=1))
        print 'cpg_ac s:\n',s
        self.act,self._ = self.action(s,reward,dt)
        a = np.hsplit(self.act[0],2)
        print 'cpg_ac a[0]:\n',a[0]
        self.lasCPGout = self.CPG(dt,a[0])
        return self.lastCPGout+a[1]

class sin_env(object):
    frequence = 5
    def state(self,timelist):
        t = np.array(timelist)
        return np.maximum(np.sin(self.frequence*t),0)
    def reward(self,timelist,value):
        t = np.array(timelist)
        return 2-sum(np.power(value - np.sin(self.frequence*t),2))

if __name__ == '__main__':
    import numpy as np
    inout = (2,2)
    a = 2.5
    A = a - a*np.tri(inout[1])
    A += A.T
    cpg_ac = cpg_actor_critic(inout,A,x0=[1,2,0,0],alpha=0.5)
    dt = 0.01
    tmax = 100

    lastact = np.array([[0,0]])
    se = sin_env()

    from save_plot import logger
    import os

    trec = 0

    i = 0
    z = np.arange(i*tmax,(i+1)*tmax,dt)
    t = [z,z+1.57*0.4]
    lgr = logger(['act','output','state','reward','TDerr'])
    for t0,t1 in zip(t[0],t[1]):
        try:
            if trec != int(t0):
                trec = int(t0)
                print int(t0)
            tl = [t0,t1]
            s = se.state((t0,t1))
            #print('state',s,t0)
            r = se.reward(tl,lastact[0])
            #print('reward',r)
            lastact = cpg_ac(s,r,dt)
            lgr.append([cpg_ac.act[0],lastact,s,r,cpg_ac._[0]])
        except:
            break

    lgr.output(os.path.expanduser('~')+'/Pictures/log/cpg_actor_critic_test',t[0],
    title='serial #'+str(i)+'\nstate=$[sin(t),sin(t+\pi)]$ reward=$n-\sum state$\n$ \\alpha = 1,\gamma,b,T = default$')
