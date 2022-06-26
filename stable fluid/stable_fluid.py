# importing libraries
from tokenize import PlainToken
import numpy as np
import scipy.sparse.linalg as splinalg
from scipy import interpolate
import matplotlib.pyplot as plt

# optional libraries
import cmasher as cmr
from tqdm import tqdm


# constant variables
DOMAIN_SIZE = 1.0
N_POINTS = 51
N_TIME_STEPS = 100
TIME_STEP_LENGTH = 0.1

KINEMATIC_VISCOSITY = 0.0001

MAX_ITER_CG = None  # CG = conjugate gradient 

def forcing_function(time, point):
    time_decay = np.maximum(
        2.0 - 0.5 * time,
        0.0,
    )

    forced_value = (
        time_decay
        *
        np.where(
            (
                (point[0] > 0.4)      # x-axis
                &
                (point[0] < 0.6)      # x-axis
                &
                (point[1] > 0.1)      # y-axis
                &
                (point[1] < 0.3)      # y-axis
            ),
            np.array([0.0, 1.0]),  # force upwards facing 
            np.array([0.0, 0.0]),
        )
    )

    return forced_value

def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)
    # these are the points representing on the grid - as scalar or grid  
    scalar_shape = (N_POINTS, N_POINTS)
    scalar_dof = N_POINTS**2                             # degrees of freedom
    vector_shape = (N_POINTS, N_POINTS, 2)
    vector_dof = N_POINTS**2 * 2

    # return evenly spaced N_POINTS over the time interval of [0, DOMAIN_SIZE]
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS) 
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    # Return coordinate matrices from coordinate vectors (example in my notes)
    X, Y = np.meshgrid(x, y, indexing="ij")

    coordinates = np.concatenate(
        (
            X[..., np.newaxis],  # adding a new axis
            Y[..., np.newaxis],
        ),
        axis = -1,  # concatenate over the last axis
    )

    # applying force to multiple points at a time
    forcing_function_vectorized = np.vectorize(
        pyfunc = forcing_function, 
        signature = "(),(d)->(d)",  # non-dimensional obj in "time", d dimensional obj in "point"
    )

    def partial_derivative_x(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[2: , 1:-1]  # x from one step forward to the end, and all the interior points in y
                -
                field[0:-2, 1:-1]  # from the first point to the second last in x, and all the interior points in y
            ) / (
                2 * element_length
            )
        )

        return diff

    def partial_derivative_y(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[1:-1, 2: ]
                -
                field[1:-1, 0:-2]
            ) / (
                2 * element_length
            )
        )

        return diff

    # how do we get this function?
    def laplace(field):
        diff = np.zeros_like(field)

        diff[1:-1, 1:-1] = (
            (
                field[0:-2, 1:-1]  # go back one step in x, take the interior points in y
                +
                field[1:-1, 0:-2]  # go back one step in y, take the interior points in x
                - 4 * 
                field[1:-1, 1:-1]  # subtract 4 times the interior points 
                +
                field[2: , 1:-1]   # from 2 to end in x direction, interior points in y
                +
                field[1:-1, 2: ]   # interior points in x, from 2 to end in y direction
            ) / (
                element_length**2  # getting discretized laplace operator
            )
        )

        return diff

    def divergence(vector_field):
        divergence_applied = (
            partial_derivative_x(vector_field[..., 0])  # 0 for first element components
            +
            partial_derivative_y(vector_field[..., 1])  # 1 for second element components
        )

        return divergence_applied

    def gradient(field):
        gradient_applied = np.concatenate(
            (
                partial_derivative_x(field)[..., np.newaxis],
                partial_derivative_y(field)[..., np.newaxis],
            ),
            axis=-1
        )    

        return gradient_applied

    def curl_2d(vector_field):
        curl_applied = (
            partial_derivative_x(vector_field[..., 1])
            -
            partial_derivative_y(vector_field[..., 0])
        )

        return curl_applied

    # 1 Euler backwards step (this could be more advanced)
    def advect(field, vector_field):
        backtraced_positions = np.clip(
            (
                coordinates
                -
                TIME_STEP_LENGTH
                *
                vector_field
            ),
            0.0,
            DOMAIN_SIZE,
        )

        advected_field = interpolate.interpn(
            points=(x, y),
            values=field,
            xi=backtraced_positions,
        )

        return advected_field

    def diffusion_operator(vector_field_flattened):
        vector_field = vector_field_flattened.reshape(vector_shape)

        diffusion_applied = (
            vector_field-
            KINEMATIC_VISCOSITY
            *
            TIME_STEP_LENGTH
            *
            laplace(vector_field)
        )

        return diffusion_applied.flatten()

    def poisson_operator(field_flattened):
        field = field_flattened.reshape(scalar_shape)

        poisson_applied = laplace(field)

        return poisson_applied.flatten()

    plt.style.use("dark_background")
    plt.figure()

    velocities_prev = np.zeros(vector_shape)

    time_current = 0.0


    # Instantly makes the loop show a smart "progress" meter
    for i in tqdm(range(N_TIME_STEPS)):
        time_current += TIME_STEP_LENGTH
        forces = forcing_function_vectorized(
            time_current,
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

        
        # (3) Diffuse
        velocities_diffused = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(vector_dof, vector_dof), 
                matvec=diffusion_operator,
            ),
            b=velocities_advected.flatten(),
            maxiter=MAX_ITER_CG,
        )[0].reshape(vector_shape)

        # (4.1) Compute a pressure correction
        pressure = splinalg.cg(
            A=splinalg.LinearOperator(
                shape=(scalar_dof, scalar_dof),
                matvec=poisson_operator,
            ),
            b=divergence(velocities_diffused).flatten(),
            maxiter=MAX_ITER_CG,
        )[0].reshape(scalar_shape)

        # (4.2) Correct the velocities to be incompressible
        velocities_projected = (
            velocities_diffused
            -
            gradient(pressure)
        )

        # Advance to next time step
        velocities_prev = velocities_projected
        # plot
        curl = curl_2d(velocities_projected)
        plt.contourf(
            X,
            Y,
            curl,
            cmap=cmr.sepia,
            levels=100,
            
        )
        plt.quiver(
            X,
            Y,
            velocities_projected[..., 0],
            velocities_projected[..., 1],
            color="black",
        )
        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    plt.show()

if __name__ == "__main__":
    main()