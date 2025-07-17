import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk 
from tkinter import ttk



def show_message():
    label["text"] = ent1.get(), ent2.get(), ent3.get(), ent4.get(), ent5.get(),ent6.get(), ent7.get(), ent8.get(), ent9.get(), ent10.get(), delta_h_star_ent.get(), x_ent.get(), beta_ent.get()

    #variable_list = [ent1.get(), ent2.get(), ent3.get(), ent4.get(), ent5.get(), ent6.get(), ent7.get(), ent8.get(), ent9.get(), ent10.get(), delta_h_star_ent.get(), x_ent.get(), beta_ent.get()]

    #empty_list=[]
    #for i in variable_list:
    #   if i == None:
    #        print(i)
    #label["text"]="Please complete these enterys:", empty_list Надо убрать наверно
            
     
    # === Physical Constants ===
    R = 8.31  # Gas constant (J/mol·K)
    # === Material Parameters ===
    Delta_h_star = int(delta_h_star_ent.get())  # Activation enthalpy (J/mol)
    x = float(x_ent.get())              # Mixing parameter
    beta = float(beta_ent.get())            # Stretching exponent (KWW exponent)

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

    # 4. Cooling: 500K → 300K @ 0.1 K/s
    cooling_rate2 = float(ent7.get())  # K/s
    T_start_cool2 = int(ent5.get())
    T_end_cool2 = int(ent8.get())
    cooling_time2 = (T_start_cool2 - T_end_cool2) / cooling_rate2
    steps_cool2 = int(cooling_time2 / dt)
    T_cool2 = np.linspace(T_start_cool2, T_end_cool2, steps_cool2)

    # 5. Heating: 300K → 450K @ 0.5 K/s
    heating_rate2 = float(ent9.get())  # K/s
    T_start_heat2 = int(ent8.get())
    T_end_heat2 = int(ent10.get())  # Below Tg for slow relaxation
    heating_time2 = (T_end_heat2 - T_start_heat2) / heating_rate2
    steps_heat2 = int(heating_time2 / dt)
    T_heat2 = np.linspace(T_start_heat2, T_end_heat2, steps_heat2)


    #########################################################################
    # Combine all segments
    T_profile = np.concatenate([T_cool, T_heat, T_iso, T_cool2, T_heat2])
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

 
root = tk.Tk()
root.title("Glass")
root.geometry("250x250") 
#root.attributes("-fullscreen", True)
 
main_frame = tk.Frame(root, padx=20, pady=20)
main_frame.pack(fill=tk.BOTH, expand=True)
###############SIM FRAME####################
data1_frame = tk.LabelFrame(main_frame, text="Simulation Data",  padx=10, pady=10)
data1_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

#cooling rate
label1 = ttk.Label(data1_frame, text="Enter cooling rate K/s")
label1.grid(row=0,column=0,sticky="w", padx=6, pady=6)
ent1 = ttk.Entry(data1_frame)
ent1.grid(row=0,column=1, padx=6, pady=6)

#start temp C
label2 = ttk.Label(data1_frame, text="Enter start temp for cooling")
label2.grid(row=1,column=0,sticky="w", padx=6, pady=6)
ent2 = ttk.Entry(data1_frame)
ent2.grid(row=1,column=1, padx=6, pady=6)

#start temp H = end temp C 
label3 = ttk.Label(data1_frame, text="Enter start temp for heating (same as end temp for cooling)")
label3.grid(row=2,column=0,sticky="w", padx=6, pady=6)
ent3 = ttk.Entry(data1_frame)
ent3.grid(row=2,column=1, padx=6, pady=6)

#heating rate
label4 = ttk.Label(data1_frame,text="Enter heating rate")
label4.grid(row=3,column=0,sticky="w", padx=6, pady=6)
ent4 = ttk.Entry(data1_frame)
ent4.grid(row=3,column=1, padx=6, pady=6)

#hold temp 
label5 = ttk.Label(data1_frame, text="Enter end heating temp(same as isotermic hold temp,"+ "\n" + " and start temp of second cooling period)")
label5.grid(row=4,column=0,sticky="w", padx=6, pady=6)
ent5 = ttk.Entry(data1_frame)
ent5.grid(row=4,column=1, padx=6, pady=6)

#hold time 
label6 = ttk.Label(data1_frame, text="Enter isotermic hold time")
label6.grid(row=5,column=0,sticky="w", padx=6, pady=6)
ent6 = ttk.Entry(data1_frame)
ent6.grid(row=5,column=1, padx=6, pady=6)

#second cooling rate
cool2_rate = ttk.Label(data1_frame, text="Enter second cooling rate")
cool2_rate.grid(row=6,column=0,sticky="w", padx=6, pady=6)
ent7 = ttk.Entry(data1_frame)
ent7.grid(row=6,column=1, padx=6, pady=6)

#cooling 2 end temp (same as start heating 2 temp)
cooling2_end_temp = ttk.Label(data1_frame, text="Enter second cooling end temp "+ "\n" + "(same as start temp for second heating)")
cooling2_end_temp.grid(row=7,column=0,sticky="w", padx=6, pady=6)
ent8 = ttk.Entry(data1_frame)
ent8.grid(row=7,column=1, padx=6, pady=6)

#heating2 rate
heating2_rate = ttk.Label(data1_frame, text="Enter second heating rate")
heating2_rate.grid(row=8,column=0,sticky="w", padx=6, pady=6)
ent9 = ttk.Entry(data1_frame)
ent9.grid(row=8,column=1, padx=6, pady=6)

#heating2 end temp
heating2_end_temp = ttk.Label(data1_frame, text="Enter second heating end temp")
heating2_end_temp.grid(row=9,column=0,sticky="w", padx=6, pady=6)
ent10 = ttk.Entry(data1_frame)
ent10.grid(row=9,column=1, padx=6, pady=6)

############MATERIAL FRAME#################################
material_frame = tk.LabelFrame(main_frame, text="Material Data",  padx=10, pady=10)
material_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

delta_h_star_label = ttk.Label(material_frame, text="Enter delta h star:")
delta_h_star_label.grid(row=0,column=0,sticky="w", padx=6, pady=6)
delta_h_star_ent = ttk.Entry(material_frame)
delta_h_star_ent.grid(row=0,column=1, padx=6, pady=6)

x_label = ttk.Label(material_frame, text="Enter x:")
x_label.grid(row=1,column=0,sticky="w", padx=6, pady=6)
x_ent = ttk.Entry(material_frame)
x_ent.grid(row=1,column=1, padx=6, pady=6)

beta_label = ttk.Label(material_frame, text="Enter beta:")
beta_label.grid(row=2,column=0,sticky="w", padx=6, pady=6)
beta_ent = ttk.Entry(material_frame)
beta_ent.grid(row=2,column=1, padx=6, pady=6)

A_label = ttk.Label(material_frame, text="Enter A(does't do anything right now ):")
A_label.grid(row=3,column=0,sticky="w", padx=6, pady=6)
A_ent = ttk.Entry(material_frame)
A_ent.grid(row=3,column=1, padx=6, pady=6)

##################TRASH#####################################
button_frame = tk.Frame(root)
button_frame.pack(pady=20) 

btn = ttk.Button(button_frame, text="Run Simulation", command=show_message)
btn.grid(row=10,column=0, padx=6, pady=6)
 

dop_frame = tk.Frame(root)
dop_frame.pack(pady=20)  

label = ttk.Label(dop_frame, text="hello there")
label.grid(row=11,column=0, padx=6, pady=6)

ent2.insert(0, 500)
delta_h_star_ent.insert(0,315000)
x_ent.insert(0, 0.5)
beta_ent.insert(0, 0.7)
  
root.mainloop()
