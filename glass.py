import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk



def show_message():
    label["text"] = ent1.get(), ent2.get(), ent3.get(), ent4.get(), ent5.get(), ent6.get()  
     
    # === Physical Constants ===
    R = 8.31  # Gas constant (J/mol·K)

    # === Material Parameters ===
    Delta_h_star = 315000  # Activation enthalpy (J/mol)
    x = 0.5                # Mixing parameter
    beta = 0.7             # Stretching exponent (KWW exponent)

    # Calculate proper pre-exponential factor A
    # Target: τ(Tg) = 100s at Tg = 500K
    Tg = 500  # Glass transition temperature (K)
    tau_Tg = 100  # Relaxation time at Tg (seconds)
    A = tau_Tg / np.exp(Delta_h_star/(R*Tg))  # Properly scaled pre-exponential factor

    print(f"Calculated A = {A:.3e}")  # Should be around 4.2e-31

    # === Simulation Parameters ===
    dt = 1.0  # Time step (seconds)

    # === Create Temperature Profile ===
    # 1. Cooling: 500K → 300K @ 0.1 K/s
    cooling_rate = float(ent1.get())  # K/s
    T_start_cool = int(ent2.get())
    T_end_cool = int(ent3.get())
    cooling_time = (T_start_cool - T_end_cool) / cooling_rate
    steps_cool = int(cooling_time / dt)
    T_cool = np.linspace(T_start_cool, T_end_cool, steps_cool)

    # 2. Heating: 300K → 450K @ 0.5 K/s
    heating_rate = float(ent4.get())  # K/s
    T_start_heat = int(ent3.get())
    T_end_heat = int(ent5.get())  # Below Tg for slow relaxation
    heating_time = (T_end_heat - T_start_heat) / heating_rate
    steps_heat = int(heating_time / dt)
    T_heat = np.linspace(T_start_heat, T_end_heat, steps_heat)

    # 3. Isothermal hold at 450K for 3 hours
    hold_time = float(ent6.get()) * 3600  # 3 hours in seconds
    steps_iso = int(hold_time / dt)
    T_iso = np.full(steps_iso, T_end_heat)  # Hold at final heat temperature


    #########################################################################
    # Combine all segments
    T_profile = np.concatenate([T_cool, T_heat, T_iso])
    total_steps = len(T_profile)

    # === Relaxation Time Function ===
    def tau(T, Tf):
        """Calculate relaxation time using Tool-Narayanaswamy model"""
        return A * np.exp(
            (x * Delta_h_star) / (R * T) + 
            ((1 - x) * Delta_h_star) / (R * Tf)
        )

    # === Initialize Fictive Temperature ===
    Tf = np.zeros(total_steps)
    Tf[0] = T_profile[0]  # Start in equilibrium

    # === Time-stepping loop ===
    for i in range(1, total_steps):
        # Previous state variables
        T_prev = T_profile[i-1]
        Tf_prev = Tf[i-1]
        
        # Current temperature
        T_now = T_profile[i]
        
        # 1. Compute relaxation time using PREVIOUS state
        tau_prev = tau(T_prev, Tf_prev)
        
        # 2. Calculate reduced time increment
        reduced_time = dt / tau_prev
        
        # 3. Compute relaxation factor (KWW equation)
        if tau_prev > 0:
            relaxation_factor = 1 - np.exp(-(reduced_time ** beta))
        else:
            relaxation_factor = 1  # Handle degenerate case
        
        # 4. Update fictive temperature
        Tf[i] = Tf_prev + (T_now - Tf_prev) * relaxation_factor

    # === Plot Results ===
    plt.figure(figsize=(8, 8))

    # Temperature profile
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(total_steps)*dt/3600, T_profile, 'b-', label='Actual Temperature')
    plt.ylabel('Temperature (K)')
    plt.axvline(len(T_cool)*dt/3600, color='k', linestyle='--', alpha=0.5)
    plt.axvline((len(T_cool)+len(T_heat))*dt/3600, color='k', linestyle='--', alpha=0.5)
    plt.grid(True)
    plt.title('Temperature Profile')

    # Fictive temperature vs actual temperature
    plt.subplot(2, 1, 2)
    plt.plot(T_profile, Tf, 'r-', label='Fictive Temperature')
    plt.plot(T_profile, T_profile, 'k--', label='Equilibrium')
    plt.xlabel('Actual Temperature (K)')
    plt.ylabel('Fictive Temperature (K)')
    plt.title('Fictive vs Actual Temperature')
    plt.grid(True)
    plt.legend()

    # Mark profile segments
    cool_end = len(T_cool)
    heat_end = len(T_cool) + len(T_heat)
    plt.annotate('Cooling', (T_profile[cool_end//2], Tf[cool_end//2]+20), ha='center')
    plt.annotate('Heating', (T_profile[cool_end + len(T_heat)//2], Tf[cool_end + len(T_heat)//2]-20), ha='center')
    plt.annotate('Isothermal Hold', (T_profile[heat_end + len(T_iso)//2], Tf[heat_end + len(T_iso)//2]), ha='center')

    plt.tight_layout()
    #plt.show()

    # === Additional Diagnostic: Relaxation Time ===
    tau_profile = np.array([tau(T_profile[i], Tf[i]) for i in range(total_steps)])

    plt.figure(figsize=(10, 4))
    plt.semilogy(np.arange(total_steps)*dt/3600, tau_profile/3600, 'g-')
    plt.axvline(len(T_cool)*dt/3600, color='k', linestyle='--', alpha=0.5)
    plt.axvline((len(T_cool)+len(T_heat))*dt/3600, color='k', linestyle='--', alpha=0.5)
    plt.axhline(1, color='r', linestyle='--', label='1 hour')
    plt.ylabel('Relaxation Time (hours)')
    plt.xlabel('Time (hours)')
    plt.title('Relaxation Time Evolution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() # получаем введенный текст

 
root = Tk()
root.title("Glass")
root.geometry("250x200") 
 

#cooling rate
label1 = ttk.Label(text="Enter cooling rate K/s")
label1.pack(anchor=NW, padx=6, pady=6)
ent1 = ttk.Entry()
ent1.pack(anchor=NW, padx=6, pady=6)

#start temp C
label2 = ttk.Label(text="Enter start temp for cooling")
label2.pack(anchor=NW, padx=6, pady=6)
ent2 = ttk.Entry()
ent2.pack(anchor=NW, padx=6, pady=6)

#start temp H = end temp C 
label3 = ttk.Label(text="Enter start temp for heating (same as end temp for cooling)")
label3.pack(anchor=NW, padx=6, pady=6)
ent3 = ttk.Entry()
ent3.pack(anchor=NW, padx=6, pady=6)

#heating rate
label4 = ttk.Label(text="Enter heating rate")
label4.pack(anchor=NW, padx=6, pady=6)
ent4 = ttk.Entry()
ent4.pack(anchor=NW, padx=6, pady=6)

#hold temp 
label5 = ttk.Label(text="Enter end heating temp(same as isotermic hold temp)")
label5.pack(anchor=NW, padx=6, pady=6)
ent5 = ttk.Entry()
ent5.pack(anchor=NW, padx=6, pady=6)

label6 = ttk.Label(text="Enter isotermic hold time")
label6.pack(anchor=NW, padx=6, pady=6)
ent6 = ttk.Entry()
ent6.pack(anchor=NW, padx=6, pady=6)
  
#end temp H 
btn = ttk.Button(text="Make a graph", command=show_message)
btn.pack(anchor=NW, padx=6, pady=6)
 
label = ttk.Label()
label.pack(anchor=NW, padx=6, pady=6)
  
root.mainloop()
