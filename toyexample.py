import emg3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def create_data():
    rec_x = np.arange(26)*100+500
    rec_y = np.array([-1000, -500, 0, 500, 1000])
    rec_z = 0.0
    rec = emg3d.surveys.txrx_coordinates_to_dict(
            emg3d.RxElectricPoint,
            (np.tile(rec_x, rec_y.size), rec_y.repeat(rec_x.size), rec_z, 0, 0)
    )
    # src_x = np.array([0, 1500, 3000])
    # src_y = np.array([-500, 0, 500])
    # src_z = 50.0
    # src = emg3d.surveys.txrx_coordinates_to_dict(
    #     emg3d.TxElectricDipole,
    #     (np.tile(src_x, src_y.size), src_y.repeat(src_x.size), src_z, 0, 0)
    # )
    # frequencies = np.array([0.5, 1.0, 5.0])
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
            max_workers=10,
            gridding_opts={'center_on_edge': False},
            receiver_interpolation='linear',
            solver_opts={'tol_gradient': 1e-3},
            tqdm_opts=False,
        )
        sim.compute(observed=True, min_offset=400)
        sim.clean('computed')

        sim.model = model_start

        sim.compute()
        sim.survey.data['start'] = sim.survey.data.synthetic
        sim.clean('computed')

        emg3d.save(
            f'toy-{case}.h5',
            sim=sim,
            grid=model_true.grid,
            model_true=model_true,
            model_start=model_start,
        )


def get_data(name):
    toy = emg3d.load(name)
    sim = toy['sim']
    model_true = toy['model_true']
    model_start = toy['model_start']
    grid = toy['grid']

    grid.plot_3d_slicer(
        1/model_true.property_x, zslice=-750,
        pcolor_opts={
            'cmap': 'Spectral_r',
            'norm': LogNorm(vmin=.1, vmax=1000), 'lw': 0.5, 'color': 'c',
        },
    )

    print(grid)
    print(model_true)
    print(sim)

    return sim, model_true, model_start, grid


def plot_models(sim, mstart, mtrue, zind=1):
    depth = np.round(mstart.grid.cell_centers_z[zind], 2)
    print(f"Depth slice: {depth} m")

    popts1 = {'cmap': 'Spectral_r', 'norm': LogNorm(vmin=0.1, vmax=1000)}
    # popts2 = {'edgecolors': 'grey', 'linewidth': 0.5, 'cmap': 'Spectral_r',
    #           'norm': LogNorm(vmin=0.1, vmax=1000)}
    opts = {'v_type': 'CC', 'normal': 'Y'}

    rec_coords = sim.survey.receiver_coordinates()
    src_coords = sim.survey.source_coordinates()

    fig, axs = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
    (ax1, ax2, ax3), (ax4, ax5, ax6) = axs

    grid = sim.model.grid

    # True model
    out1, = grid.plot_slice(1/mtrue.property_x.ravel('F'), ax=ax1,
                            pcolor_opts=popts1, **opts)
    ax1.set_title("True Model (Ohm.m)")
    ax1.plot(rec_coords[0], rec_coords[2], 'bv')
    ax1.plot(src_coords[0], src_coords[2], 'r*')

    # Start model
    out2, = grid.plot_slice(
            1/mstart.property_x.ravel('F'), ax=ax2, pcolor_opts=popts1, **opts)
    ax2.set_title("Start Model (Ohm.m)")

    # Final inversion model
    out3, = grid.plot_slice(
            1/sim.model.property_x.ravel('F'), ax=ax3, pcolor_opts=popts1,
            **opts)
    ax3.set_title("Final Model (Ohm.m)")

    opts['normal'] = 'Z'
    opts['ind'] = zind

    # True model
    out4, = grid.plot_slice(
            1/mtrue.property_x.ravel('F'), ax=ax4, pcolor_opts=popts1, **opts)
    ax4.set_title("True Model (Ohm.m)")
    ax4.plot(rec_coords[0], rec_coords[1], 'bv')
    ax4.plot(src_coords[0], src_coords[1], 'r*')

    # Start model
    out5, = grid.plot_slice(
            1/mstart.property_x.ravel('F'), ax=ax5, pcolor_opts=popts1, **opts)
    ax5.set_title("Start Model (Ohm.m)")

    # Final inversion model
    out6, = grid.plot_slice(
            1/sim.model.property_x.ravel('F'), ax=ax6, pcolor_opts=popts1,
            **opts)
    ax6.set_title("Final Model (Ohm.m)")

    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('')
        ax.set_xticklabels([])

    for ax in [ax2, ax3, ax5, ax6]:
        ax.set_ylabel('')
        ax.set_yticklabels([])

    for ax in axs.ravel():
        ax.axis('equal')

    plt.colorbar(
            out1, ax=axs, orientation='horizontal', fraction=.1, shrink=.8,
            aspect=30)


def plot_responses(sim):
    fig, axs = plt.subplots(
            2, 1, figsize=(9, 6), constrained_layout=True, sharex=True)
    real = [int(k[2:]) for k in sim.data.keys() if k.startswith('it')]
    for i in real:
        n = f"it{i}"
        axs[0].plot(np.abs(sim.data[n].squeeze()), f"C{i % 10}-", label=n)
        rms = 100*np.abs((sim.data[n].squeeze() - sim.data.observed.squeeze()))
        rms /= np.abs(sim.data.observed.squeeze())
        axs[1].plot(rms, f"C{i % 10}-")
    axs[0].plot(np.abs(sim.data.observed.squeeze()), "k.")
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')
    axs[0].legend()
