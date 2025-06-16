# Create the Reference in controllable plane
import warnings
from MPC_utils import SamplingBasedMPC, OptimizationBasedMPC, SimpleSamplingBasedMPC
from MPC_config import *
from utils import vehicle_model_without_delay, cost_function1, get_unique_filepath, new_vehicle_model, new_vehicle_model_for_logging
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

# ================================== FOR ADDING WARNING ============================
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"

warnings.formatwarning = custom_warning_format
# ==================================================================================

# function to convert from (d, vp, vf) to (z1, z2)
def real_to_z(state, h = 1):
    (d, vp, vf) = state
    z1 = d
    z2 = vp
    if (z1/h != vf):
        warnings.warn(f'State {state} is not on controllable plane. Use d as z1!')
    return (z1,z2)

# function to convert from (z1, z2) to (d, vp, vf)
def z_to_real(z, h = 1):
    d = z[0]
    vp = z[1]
    vf = z[0]/h
    return (d, vp, vf)

def state_distance(state1, state2):
    state1_np = np.array(state1)
    state2_np = np.array(state2)

    difference = state1_np-state2_np
    return np.sqrt(np.sum(np.square(difference)))
    

if __name__ == '__main__':
    
    # set parameter
    h = 1

    # want initial point to be (15, 15, 15)
    intitial_state_z = (15,15)
    
    # want target point to be (10,7,10) -> 1
    # target_z = (10,7)
    # want target point to be (20, 23, 20) -> 2
    # target_z = (20,23)  
    # want target point to be (12,7,12) -> 3 
    target_z = (12,7)  

    # save_dir = 'Z_state/simple_sampling_based'
    # save_dir = 'Z_state/without_delta_min'
    # save_dir = 'Z_state/no_delta_min_all_accel_limited'
    # save_dir = 'Z_state/500_samples_no_delta_acc_limited'
    save_dir = 'Z_state/compared_500_samples_acc_limited'
    # anim_name = '_state_space_animation'
    # anim_name = '_state_space_animation_acc_limited'
    anim_name = '_500_samples_acc_limited_without_delta_min'

    # Initiate state-space and MPC
    distance_range = (5,50)     # Distance range in meters
    rel_speed_range = (-5,5)    # Relative speed range in m/s
    grid_resolution = (10,10)   # Grid resolution
    num_samples = 500
    N = 20

    intitial_state = z_to_real(intitial_state_z)
    target = z_to_real(target_z)

    target_rel = (target[0],target[1]-target[2])


    # model = vehicle_model_without_delay
    model = new_vehicle_model
    state = intitial_state
    # Initialize mpc
    mpc = SimpleSamplingBasedMPC(model,cost_function1, N, num_samples)
    mpc.set_target(target_rel)


    preceding_acc_history = []
    following_acc_history = []

    path = [intitial_state]
    max_step = 500
    for _ in range(max_step):
        optimal_input_sequence = mpc.solve(state)

        optimal_input = optimal_input_sequence[0]
        preceding_acc_history.append(optimal_input)
        d,vp,vf, optimal_following_acc  = new_vehicle_model_for_logging(state,optimal_input)
        # d,vp,vf  = new_vehicle_model(state,optimal_input)
        # d,vp,vf  = vehicle_model_without_delay(state,optimal_input)
        following_acc_history.append(optimal_following_acc)
        state = (d,vp,vf)
        path.append(state)

    # print(state)
    # print(state_distance(state,target))

    df = pd.DataFrame({
        'preceding_acceleration': preceding_acc_history,
        'following_acceleration': following_acc_history
    })
    
    csv_name = get_unique_filepath(save_dir,f'without_delta_min','.csv')
    df.to_csv(csv_name,index=False)

    plot = True
    if plot:
    # ====================================== Plot Animation =============================================#
        path_z = [real_to_z(s) for s in path]

        pov_choices = ['z','actual']
        pov = pov_choices[1]

        plot_path = path if pov == pov_choices[1] else path_z

        # Separate into x and y
        x_data = [s[0] for s in plot_path]
        y_data = [s[1]-s[2] for s in plot_path] if pov == pov_choices[1] else [s[1] for s in plot_path]

        # Set up the figure
        fig, ax = plt.subplots()
        ax.grid(True)

        line, = ax.plot([], [], 'bo-', lw=2)
        
        
        if pov == pov_choices[1]:
            ax.scatter(intitial_state[0],intitial_state[1]-intitial_state[2],color='green')
            ax.scatter(target_rel[0],target_rel[1],color='red')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Relative Velocity (m/s)')
            ax.set_xlim(5,50)
            ax.set_ylim(-5,5)
        else:
            ax.scatter(intitial_state_z[0],intitial_state_z[1],color='green')
            ax.scatter(target_z[0],target_z[1],color='red')
            ax.set_xlim(5,20)
            ax.set_ylim(5,20)
            ax.set_xlabel('Z_1')
            ax.set_ylabel('Z_2')

        ax.set_title(f"2D {pov} state animation")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(x_data[:frame+1], y_data[:frame+1])
            return line,

        ani = animation.FuncAnimation(
            fig, update, frames=len(plot_path),
            init_func=init, blit=True, interval=500, repeat=False
        )

        save_path = get_unique_filepath(save_dir,f'{pov}{anim_name}','.gif')
        ani.save(save_path, writer="pillow")
        save_path = get_unique_filepath(f'{save_dir}/mp4',f'{pov}{anim_name}','.mp4')
        ani.save(save_path, writer="ffmpeg")


        pov = pov_choices[0]

        plot_path = path if pov == pov_choices[1] else path_z

        # Separate into x and y
        x_data = [s[0] for s in plot_path]
        y_data = [s[1]-s[2] for s in plot_path] if pov == pov_choices[1] else [s[1] for s in plot_path]

        # Set up the figure
        fig, ax = plt.subplots()
        ax.grid(True)

        line, = ax.plot([], [], 'bo-', lw=2)
        
        
        if pov == pov_choices[1]:
            ax.scatter(intitial_state[0],intitial_state[1]-intitial_state[2],color='green')
            ax.scatter(target_rel[0],target_rel[1],color='red')
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Relative Velocity (m/s)')
            ax.set_xlim(5,50)
            ax.set_ylim(-5,5)
        else:
            ax.scatter(intitial_state_z[0],intitial_state_z[1],color='green')
            ax.scatter(target_z[0],target_z[1],color='red')
            ax.set_xlim(5,20)
            ax.set_ylim(5,20)
            ax.set_xlabel('Z_1')
            ax.set_ylabel('Z_2')

        ax.set_title(f"2D {pov} state animation")

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            line.set_data(x_data[:frame+1], y_data[:frame+1])
            return line,

        ani = animation.FuncAnimation(
            fig, update, frames=len(plot_path),
            init_func=init, blit=True, interval=500, repeat=False
        )

        save_path = get_unique_filepath(save_dir,f'{pov}{anim_name}','.gif')
        ani.save(save_path, writer="pillow")
        save_path = get_unique_filepath(f'{save_dir}/mp4',f'{pov}{anim_name}','.mp4')
        ani.save(save_path, writer="ffmpeg")

        # plt.show()


