# ********************NOT WORKING!**********************
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import interpolate
from tqdm import tqdm
from models.mlp_model import FCNet
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("CUDA is available")
else: 
    print("CUDA is not available")

def forcing_function(time, point):
        time_decay = np.maximum(2.0 - 0.5 * time, 0.0)

        forced_value = (
            time_decay
            *
            np.where(
                (
                    (point[0] > 0.4)      # x-axis
                    & 
                    (point[0] < 0.6)      # x-axis
                    &
                    (point[1] > 0)      # y-axis
                    &
                    (point[1] < 0.2)      # y-axis
                ),
                np.array([0.0, 1.0]),  # force upwards facing 
                np.array([0.0, 0.0]),
            )
        )

        return forced_value

def main(): 
    # constant variables
    DOMAIN_SIZE = 1.0
    N_POINTS = 51
    N_TIME_STEPS = 60
    TIME_STEP_LENGTH = 0.1  # 0.1 blows up - (the smaller the more organized in direction)
    ELEMENT_LENGTH = DOMAIN_SIZE / (N_POINTS - 1) # dx & dy
    VECTOR_SHAPE = ((N_POINTS, N_POINTS, 2))
    SCALAR_SHAPE = (N_POINTS, N_POINTS)
    SCALAR_DOF = N_POINTS**2 

    current_time = 0.0
    velocities_prev = np.zeros(VECTOR_SHAPE)
    pressure_correction = np.zeros(SCALAR_SHAPE)


    # setting coordinate system
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS) 
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    X, Y = np.meshgrid(x, y, indexing="ij")

    coordinates = np.concatenate( (X[..., np.newaxis], Y[..., np.newaxis]), axis = -1)

    # applying force to multiple points at a time
    forcing_function_vectorized = np.vectorize(
        pyfunc = forcing_function, 
        signature = "(),(d)->(d)",  # non-dimensional obj in "time", d dimensional obj in "point"
    )
    
    def partial_derivative_x(field):
        diff = field.copy()
        # central difference
        diff[1:-1, 1:-1] = ((field[2:, 1:-1] - field[0:-2, 1:-1]) / (2 * ELEMENT_LENGTH)) # (field_i+1 - field_i-1) / (2 * dx)

        return diff

    def partial_derivative_y(field):
        diff = field.copy()
        # central difference
        diff[1:-1, 1:-1] = ((field[1:-1, 2:] - field[1:-1, 0:-2]) / (2 * ELEMENT_LENGTH)) # (field_j+1 - field_j-1) / (2 * dy)

        return diff

    def divergence(field):
        divergence_applied = (partial_derivative_x(field[..., 0]) + partial_derivative_y(field[..., 1]))

        return divergence_applied

    def gradient(field):
        gradient_applied = np.concatenate(
            (partial_derivative_x(field)[..., np.newaxis], partial_derivative_y(field)[..., np.newaxis]), axis=-1)    

        return gradient_applied
    
    # 1 Euler backwards step (this could be more advanced)
    def advect(field, vector_field):
        # values must be within 0 & DOMAIN_SIZE
        backtraced_positions = np.clip((coordinates - TIME_STEP_LENGTH * vector_field), 0.0, DOMAIN_SIZE)

        advected_field = interpolate.interpn(points=(x, y), values=field, xi=backtraced_positions)

        return advected_field


    plt.style.use("dark_background")

    # for collecting data
    # training_inputs = []
    training_outputs = [] 


    # Instantiate the model
    model = FCNet(input_size=SCALAR_DOF).to(device)

    # Load the trained model state dict
    model.load_state_dict(torch.load("datasets/mlp/trained_model.pth"))

    
    for _ in tqdm(range(N_TIME_STEPS)):  # the bigger N_TIME_STEPS gets the more likely convection part blows up (so delta t is never small enough for the solution to converge)
        current_time += TIME_STEP_LENGTH
        forces = forcing_function_vectorized(
            current_time,
            coordinates, # points
        )

        # (1) Apply Forces
        velocities_forces_applied = (
            velocities_prev
            +
            TIME_STEP_LENGTH
            *
            forces
        )
        # (2) Nonlinear convection (=self-advection)
        velocities_advected = advect(
            field=velocities_forces_applied,
            vector_field=velocities_forces_applied,
        )
    
        velocities_advected_div = divergence(velocities_advected)
  
        # (3.1) Compute a pressure correction
        velocities_advected_div_tensor = torch.tensor(velocities_advected_div[np.newaxis, ...]).float().to(device) #[1, 51, 51]
        velocities_advected_div_tensor = torch.reshape(velocities_advected_div_tensor, (1, SCALAR_DOF)) #[1, 2601]
        
        pressure_correction_tensor = model(velocities_advected_div_tensor) # [1, 2601]
        pressure_correction = pressure_correction_tensor.detach().cpu().numpy().reshape(SCALAR_SHAPE) # [51, 51]

        # print("velocity div size: ", velocities_advected_div_tensor.shape)    # [1, 2601]
        # print("pressure correction Tensor size: ", pressure_correction_tensor.shape)        # [1, 2601]
        # print("pressure correction size: ", pressure_correction.shape)        # [51, 51]

        # training_inputs.append(velocities_advected_div)  # [51, 51]
        training_outputs.append(pressure_correction)     # [51, 51]
        
        # (3.2) Correct the velocities to be incompressible
        velocities_projected = (velocities_advected - gradient(pressure_correction))

        # Advance to next time step
        velocities_prev = velocities_projected


    # training_inputs = np.array(training_inputs)
    training_outputs = np.array(training_outputs)
    print(training_outputs.shape)
    
    # Choose the indices for the point you want to plot
    x = 30  # X coordinate of the point
    y = 30  # Y coordinate of the point

    # Extract the pressure values of the chosen point over time
    pressure_over_time = training_outputs[:, y, x]
    # print("pressure: ", pressure_over_time[:])
    time = np.arange(pressure_over_time.shape[0])
    plt.plot(time, pressure_over_time)
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.title('Pressure at Point ({}, {}) - Neural Net'.format(x, y))
    plt.grid(True)
    plt.savefig('Figures/pressure_over_time_nn_1_1.png')


if __name__ == "__main__":
    main()    
