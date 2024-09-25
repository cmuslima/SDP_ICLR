import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
TIMESTEP=0 


def render_trajectory_again(segment1, segment2, trajectory_id):
    trajectory_comps = ['A', 'B']
    segments = [segment1, segment2]


    while(True):

        render = requesting_render_again()
        if render == 1:
            render_trajectory(segments[0], trajectory_id, trajectory_comps[0])
        elif render == 0:
            render_trajectory(segments[1], trajectory_id, trajectory_comps[1])
        else:
            break

def assert_correct_string_input(label):
    if label == 'A' or label == 'a':
        print('Repeating Trajectory A')
        return 1
    elif label == 'B' or label == 'b':
        print('Repeating Trajectory B')
        return 0 
    else:
        print('No repeating any trajectories')
        print('\n')
        return -1 
  
def requesting_render_again():
    print(f'Press A if you want to see Trajectory A again')
    print(f'Press B if you want to see Trajectory B again')
    human_label = input("Press Enter if you dont want to see either: \n")
    print('\n')
    human_label = assert_correct_string_input(human_label)
    return human_label



def render_trajectory(frames, trajectory_id, trajectory_comp, fig=None, ax=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
 

    img = ax.imshow(frames[0])
    
    # Function to update the plot for each frame
    def update(frame):
       
        global TIMESTEP
        ax.axis('on')  # Turn off axis labels      
        img.set_data(frame) 
        if TIMESTEP >= 50:
            ax.set_xlabel(f'Trajectory is over, exit screen', size = 20)
        else:
            ax.set_xlabel(f'Time step {TIMESTEP}', size = 20)
        TIMESTEP+=1
        
        return img
    plt.title(f'Preference Comparsion # {trajectory_id}, Trajectory {trajectory_comp}', size = 25)

    ani = animation.FuncAnimation(fig, update, frames, interval=30, repeat=False)
    plt.show()
  
    global TIMESTEP
    TIMESTEP = 0
 
  
  


        
