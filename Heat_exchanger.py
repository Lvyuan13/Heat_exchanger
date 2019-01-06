# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:59:50 2018

@author: Lyu yuan
"""
import numpy as np
import matplotlib.pyplot as plt
class State():
    
    def __init__(self,length=1,timestep=1000):
        cp_air=4210.9
        cp_water=4210.9
        H1_init=cp_air*7   #initial termperature of air is 17 celsius
        H2_init=cp_water*7  #initial temperature of water is 40 celsius
        H1_border=cp_air*7   # input air termperature of air is 7 celsius
        H2_border=cp_water*40 # input water termperature of water is 40 celsius
        alpha=300000   #heat exchange coefficent
        rho_air=1000
        rho_water=1000
        A1=0.2
        A2=0.2
        S1=0.6   #circumference of section above
        S2=0.6   #circumference of section above
# =============================================================================
#         L1=2     #length of heat exchanger
#         L2=2     #length of heat exchanger
# =============================================================================
        u1=1    #vector of cold fluid
        u2=0.5  #vector of hot fluid
        K=timestep #time steps
        N=10  #nude NO
        L1=length
        Deltax=L1/N   #length step set depending on nodes NO
        Deltat=0.01 #time steps set as 0.01s
        H1=np.zeros((K,N+1))
        T1=np.zeros((K,N+1))
        H2=np.zeros((K,N+1))
        T2=np.zeros((K,N+1))
        q=np.zeros((K,N+1))
        self.cp_fluid1=cp_air
        self.cp_fluid2=cp_water
        self.H1_init=H1_init
        self.H2_init=H2_init
        self.H1_border=H1_border
        self.H2_border=H2_border
        self.alpha=alpha
        self.rho_fluid1=rho_air
        self.rho_fluid2=rho_water
        self.A1=A1
        self.A2=A2
        self.S1=S1
        self.S2=S2
        self.L1=length
        self.L2=length
        self.u1=u1
        self.u2=u2
        self.Deltax=Deltax
        self.Deltat=Deltat
        self.H1=H1
        self.T1=T1
        self.H2=H2
        self.T2=T2
        self.q=q
        self.N=N
        self.K=K
        self.time=[]
        self.x=[]
        self.addtimestep=0   #before turb run addtimestep is 0
        for j in range(self.K):
            self.time.append(j*self.Deltat)
        for i in range(self.N+1):
            self.x.append(i*self.Deltax)
            
    def set_initial(self):
        j=0
        for i in range(self.N+1):
            self.H1[j,i] = self.H1_init
            self.T1[j,i] = self.h2T1(self.H1[j,i])
            self.H2[j,i] = self.H2_init
            self.T2[j,i] = self.h2T2(self.H2[j,i])
            
    def set_border(self):
        i=0
        for j in range(self.K):
            self.H1[j,i] = self.H1_border
            self.T1[j,i] = self.h2T1(self.H1[j,i])
            self.H2[j,i]=self.H2_border
            self.T2[j,i]=self.h2T2(self.H2[j,i])
            
    def turb(self,H_input1,H_input2,addtimestep=2000):
        # note:must be run after method:calculate
        self.addtimestep=addtimestep
        self.H1_PLUS=np.zeros((self.addtimestep,self.N+1))
        self.T1_PLUS=np.zeros((self.addtimestep,self.N+1))
        self.H2_PLUS=np.zeros((self.addtimestep,self.N+1))
        self.T2_PLUS=np.zeros((self.addtimestep,self.N+1))
        self.q_PLUS=np.zeros((self.addtimestep,self.N+1))
        #-------------set disturb initial state-----------
        j=0
        for i in range(self.N+1):
            self.H1_PLUS[j,i]=self.H1[j+self.K-1,i]
            self.T1_PLUS[j,i]=self.T1[j+self.K-1,i]
            self.H2_PLUS[j,i]=self.H2[j+self.K-1,i]
            self.T2_PLUS[j,i]=self.T2[j+self.K-1,i]
        #--------------set border state of turb-----------
        i=0
        for j in range(self.addtimestep):
            self.H1_PLUS[j,i]=H_input1
            self.T1_PLUS[j,i]=self.h2T1(self.H1_PLUS[j,i])
            self.H2_PLUS[j,i]=H_input2
            self.T2_PLUS[j,i]=self.h2T2(self.H2_PLUS[j,i])
#-------------------main loop-------------------------------
        for j in range(self.addtimestep-1):
            for i in range(self.N):
                self.q_PLUS[j,i]=self.alpha*(self.T2_PLUS[j,i]-self.T1_PLUS[j,i]+self.T2_PLUS[j,i+1]-self.T1_PLUS[j,i+1])/2
                inter=-self.u1*(self.H1_PLUS[j,i+1]-self.H1_PLUS[j,i])/self.Deltax+self.q_PLUS[j,i]*self.S1/(self.rho_fluid1*self.A1)
                self.H1_PLUS[j+1,i+1]=inter*self.Deltat+self.H1_PLUS[j,i+1]
                self.T1_PLUS[j+1,i+1]=self.h2T1(self.H1_PLUS[j+1,i+1])
                inter=-self.u2*(self.H2_PLUS[j,i+1]-self.H2_PLUS[j,i])/self.Deltax-self.q_PLUS[j,i]*self.S2/(self.rho_fluid2*self.A2)
                self.H2_PLUS[j+1,i+1]=inter*self.Deltat+self.H2_PLUS[j,i+1]
                self.T2_PLUS[j+1,i+1]=self.h2T2(self.H2_PLUS[j+1,i+1])    
                #---------------link all martix
        self.T1=np.vstack((self.T1,self.T1_PLUS))
        self.T2=np.vstack((self.T2,self.T2_PLUS))
        self.time=[]
        for j in range(self.K+self.addtimestep):
            self.time.append(j*self.Deltat)
            
    def calculate(self):
        for j in range(self.K-1):
            for i in range(self.N):
                self.q[j,i]=self.alpha*(self.T2[j,i]-self.T1[j,i]+self.T2[j,i+1]-self.T1[j,i+1])/2
                inter=-self.u1*(self.H1[j,i+1]-self.H1[j,i])/self.Deltax+self.q[j,i]*self.S1/(self.rho_fluid1*self.A1)
                self.H1[j+1,i+1]=inter*self.Deltat+self.H1[j,i+1]
                self.T1[j+1,i+1]=self.h2T1(self.H1[j+1,i+1])     
                inter=-self.u2*(self.H2[j,i+1]-self.H2[j,i])/self.Deltax-self.q[j,i]*self.S2/(self.rho_fluid2*self.A2)
                self.H2[j+1,i+1]=inter*self.Deltat+self.H2[j,i+1]
                self.T2[j+1,i+1]=self.h2T2(self.H2[j+1,i+1]) 
    def output1(self):

        #return output temperature of channel 1
        return self.T1[:,self.N]
    def output2(self):
        #return output temperature of channel 2
        return self.T2[:,self.N]
    def print_info(self):

        print('this function print out nessasary information:\n')
# =============================================================================
        print('length of the heat exchanger is %6.3f meters'%(self.L1))
        print('there are %d nodes and space steps is %5.3f meters'%(self.N,self.Deltax))
        print('there are %d time steps and time steps is %5.3f seconds'%(self.K+self.addtimestep,self.Deltat))
# =============================================================================
        
    def h2T2(self,h):

        cp_water=4210.9
        T2=h/cp_water
        return T2

    def h2T1(self,h):

        cp_air=4210.9
        T1=h/cp_air
        return T1

    def T1_x(self,timestepNO):

        #return x direction temperature disturbtion of channel 1
        return self.T1[timestepNO,:]

    def T2_x(self,timestepNO):

        #return x direction temperature disturbtion of channel 2
        return self.T2[timestepNO,:]

a=State()
a.set_initial()
a.set_border()
a.calculate()
T1=a.output1()
T2=a.output2()
#-------output temperature change----------------------------------------------
plt.figure(1)
plt.plot(a.time,T1,label='output1')
plt.plot(a.time,T2,label='output2')
plt.xlabel('time(s)')
plt.ylabel('temperature(celsius)')
plt.title('tempeature of outputs')
plt.legend()
plt.show()

#--------x direction disturbtion of temperature--------------------------------
plt.figure(2)
T1_X=a.T1_x(a.K-100)
T2_X=a.T2_x(a.K-100)
plt.plot(a.x,T1_X,label='disturbtion of channel 1')
plt.plot(a.x,T2_X,label='disturbtion of channel 2')
plt.title('disturbiton of temperature in x direction')
plt.legend()
plt.show()

#--------------------disturb---------------------------------------------------
a.turb(4210.9*7,4210.9*50)
T11=a.output1()
T22=a.output2()
plt.figure(3)
plt.plot(a.time,T11,label='output11')
plt.plot(a.time,T22,label='output22')
plt.xlabel('time(s)')
plt.ylabel('temperature(celsius)')
plt.title('tempeature of outputs_2')
plt.legend()
plt.show()

plt.figure(4)
#a.turb(4210.9*7,4210.9*50)
T11_X=a.T1_x(a.K+a.addtimestep-1)
T22_X=a.T2_x(a.K+a.addtimestep-1)
plt.plot(a.x,T11_X,label='disturbtion of channel 1')
plt.plot(a.x,T22_X,label='disturbtion of channel 2')
plt.title('disturbiton of temperature in x direction_2')
plt.legend()
plt.show()

a.print_info()
