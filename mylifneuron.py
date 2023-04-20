import numpy as np
import matplotlib.pyplot as plt
#modules file
def simulation(I, dt, V0=-75, E=-75, g=10, tau=10, V_threshold=-55, V_reset=-75, tau_refractory=2):
    # Default parameters are for a typical neuron:
    # I: injected current (pA)
    # dt: sample interval (ms)
    # V0: initial membrane voltage (mV)
    # E: reversal potential (mV)
    # g: conductance (nS)
    # tau: membrane time constant (ms)
    # V_threshold: spike threshold (mV)
    # V_reset: refractory potential (mV)
    # tau_refractory: refractory time (ms)
    
    # list of spike times
    spike_times = []

    # we will use this to keep track of whether the neuron is in a refractory period
    refractory_time = 0

    # init voltage array for all time steps
    V = np.zeros(I.shape)

    # set the initial voltage
    V[0] = V0

    for i in range(1, len(V)):
        # compute voltage at time step i based on the voltage at time step i-1

        # in refractory period?
        if refractory_time > 0:
            V[i] = V_reset
            refractory_time -= dt
            continue
        
        # change in membrane voltage for ith time step
        dV = ( -(V[i-1] - E) + I[i-1] / g ) * (dt / tau)

        V[i] = V[i-1] + dV

        # spike?
        if V[i] >= V_threshold:
            # record spike time step index (we'll convert to time later)
            spike_times.append(i)
            V[i] = 0  # just so spike is obvious
            # start refractory period
            refractory_time = tau_refractory
    
    # spike time step indices -> times
    spike_times = np.array(spike_times) * dt

    # return LIF neuron membrane voltage time series and array of spike times
    return V, spike_times

def splot(time, I, V):
    plt.subplot(2, 1, 1)
    plt.plot(time, I)
    plt.ylabel('Current (pA)')
    plt.subplot(2, 1, 2)
    plt.plot(time, V)
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)');

def synaptic(dt, I, 
                spikesE, spikesI, sE=1.2, sI=1.6, tauE=5, tauI=10, EE=0, EI=-80, 
                V0=-75, EL=-75, gL=10, tau=10, V_threshold=-55, V_reset=-75, tau_refractory=2):
    # dt: sample interval (ms)
    # I: injected current (pA)
    # spikes_E: # of excitatory synaptic inputs per sample interval (each row is a neuron)
    # spikes_I: # of inhibitory synaptic inputs per sample interval (each row is a neuron)
    # s_E: excitatory synaptic strength (nS)
    # s_I: inhibitory synaptic strength (nS)
    # tau_E: excitatory synaptic time constant (ms)
    # tau_I: inhibitory synaptic time constant (ms)
    # E_E: excitatory reversal potential (mV)
    # E_I: inhibitory reversal potential (mV)
    # V0: initial membrane voltage (mV)
    # E_L: leak reversal potential (mV)
    # g_L: leak conductance (nS)
    # tau: membrane time constant (ms)
    # V_threshold: spike threshold (mV)
    # V_reset: refractory potential (mV)
    # tau_refractory: refractory time (ms)
    
    # total number of spikes from all input neurons per time step
    NE = spikesE.sum(axis=0)
    NI = spikesI.sum(axis=0)

    # time dependent excitatory and inhibitory conductances
    # these will be computed from the input spike trains
    gE = np.zeros(NE.shape)
    gI = np.zeros(NI.shape)

    # LIF neuron
    spike_times = []
    refractory_time = 0
    V = np.zeros(NE.shape)
    V[0] = V0
    for i in range(1, len(V)):
        # update the synaptic conductances
        gE[i] = gE[i-1] - gE[i-1] * (dt / tauE) + NE[i] * sE
        gI[i] = gI[i-1] - gI[i-1] * (dt / tauI) + NI[i] * sI

        # in refractory period?
        if refractory_time > 0:
            V[i] = V_reset
            refractory_time -= dt
            continue
        
        # change in membrane voltage for ith time step
        dV = (
            -(V[i-1] - EL) 
            - gE[i-1] / gL * (V[i-1] - EE) 
            - gI[i-1] / gL * (V[i-1] - EI) 
            + I[i-1] / gL
            ) * (dt / tau)

        V[i] = V[i-1] + dV

        # spike?
        if V[i] >= V_threshold:
            spike_times.append(i)
            V[i] = 0  # just so spike is obvious
            refractory_time = tau_refractory
    
    spike_times = np.array(spike_times) * dt

    return V, spike_times, gE, gI

dt = 0.1
# impulse response to each excitatory or inhibitory spike
sE=1.2
sI=1.6
tauE=5
tauI=10

t = np.arange(500) * dt

excitatory_spike_response = sE * np.exp(-t / tauE)
inhibitory_spike_response = sI * np.exp(-t / tauI)

def synaptic_conv(dt, I, 
                spikesE, spikesI, sE=1.2, sI=1.6, tauE=5, tauI=10, EE=0, EI=-80, 
                V0=-75, EL=-75, gL=10, tau=10, V_threshold=-55, V_reset=-75, tau_refractory=2):
    # dt: sample interval (ms)
    # I: injected current (pA)
    # spikes_E: # of excitatory synaptic inputs per sample interval (each row is a neuron)
    # spikes_I: # of inhibitory synaptic inputs per sample interval (each row is a neuron)
    # s_E: excitatory synaptic strength (nS)
    # s_I: inhibitory synaptic strength (nS)
    # tau_E: excitatory synaptic time constant (ms)
    # tau_I: inhibitory synaptic time constant (ms)
    # E_E: excitatory reversal potential (mV)
    # E_I: inhibitory reversal potential (mV)
    # V0: initial membrane voltage (mV)
    # E_L: leak reversal potential (mV)
    # g_L: leak conductance (nS)
    # tau: membrane time constant (ms)
    # V_threshold: spike threshold (mV)
    # V_reset: refractory potential (mV)
    # tau_refractory: refractory time (ms)
    
    # total number of spikes from all input neurons per time step
    NE = spikesE.sum(axis=0)
    NI = spikesI.sum(axis=0)

    # time dependent excitatory and inhibitory conductances
    gE = np.convolve(NE, excitatory_spike_response)[:len(NE)]
    gI = np.convolve(NI, inhibitory_spike_response)[:len(NI)]

    # LIF neuron
    spike_times = []
    refractory_time = 0
    V = np.zeros(NE.shape)
    V[0] = V0
    for i in range(1, len(V)):
        # in refractory period?
        if refractory_time > 0:
            V[i] = V_reset
            refractory_time -= dt
            continue
        
        # change in membrane voltage for ith time step
        dV = (
            -(V[i-1] - EL) 
            - gE[i-1] / gL * (V[i-1] - EE) 
            - gI[i-1] / gL * (V[i-1] - EI) 
            + I[i-1] / gL
            ) * (dt / tau)

        V[i] = V[i-1] + dV

        # spike?
        if V[i] >= V_threshold:
            spike_times.append(i)
            V[i] = 0  # just so spike is obvious
            refractory_time = tau_refractory
    
    spike_times = np.array(spike_times) * dt

    return V, spike_times, gE, gI