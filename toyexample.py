import emg3d
import numpy as np

rec_x = np.arange(26)*100+500
rec_y = np.array([-1000, -500, 0, 500, 1000])
rec_z = 0.0
rec = emg3d.surveys.txrx_coordinates_to_dict(
        emg3d.RxElectricPoint,
        (np.tile(rec_x, rec_y.size), rec_y.repeat(rec_x.size), rec_z, 0, 0)
)
src = emg3d.TxElectricDipole((0, 0, 0, 0, 0))
frequencies = np.array([1.0, ])
survey = emg3d.surveys.Survey(
    sources=src,
    receivers=rec,
    frequencies=frequencies,
    noise_floor=1e-17,
    relative_error=0.05,
)

for case in ['tiny', 'small']:
    if case == 'small':
        hx = np.ones(16)*250.0
        hyz = np.ones(8)*250.0
        grid = emg3d.TensorMesh([hx, hyz, hyz], [-500, -1000, -1750])
    else:
        hx = np.ones(8)*500.0
        hyz = np.ones(4)*500.0
        grid = emg3d.TensorMesh([hx, hyz, hyz], [-500, -1000, -1500])

    model_start = emg3d.Model(grid, 1.0, mapping='Conductivity')
    model_true = emg3d.Model(grid, 1.0, mapping='Conductivity')
    # Target
    if case == 'small':
        model_true.property_x[3:12, 2:-2, 3:-3] = 0.001
    else:
        model_true.property_x[2:5, 1:-1, 1:-2] = 0.001
    # Water
    model_start.property_x[:, :, -1] = 3.3
    model_true.property_x[:, :, -1] = 3.3

    # Create an emg3d Simulation instance
    sim = emg3d.simulations.Simulation(
        survey=survey.copy(),
        model=model_true,
        gridding='both',
        max_workers=2,
        gridding_opts={'center_on_edge': False},
        receiver_interpolation='linear',
        tqdm_opts=False,
    )
    sim.compute(observed=True)
    sim.clean('computed')

    sim.model = model_start
    sim.solver_opts = {'tol': 1e-3}

    sim.compute()
    sim.survey.data['start'] = sim.survey.data.synthetic
    sim.clean('computed')

    emg3d.save(f'toy-{case}.h5', sim=sim, model_true=model_true)
