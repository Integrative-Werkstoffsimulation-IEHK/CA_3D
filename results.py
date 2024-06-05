# from visualisation import *
from visualisation import *
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
database_name = filedialog.askopenfilename()
iter = 4546
visualise = Visualisation(database_name)

# visualise.animate_3d(animate_separate=False, const_cam_pos=False)

visualise.plot_3d(plot_separate=False, const_cam_pos=False)
# visualise.plot_3d(plot_separate=False, const_cam_pos=True, iteration=iter)
# visualise.animate_2d(plot_separate=False)

visualise.plot_2d(plot_separate=False)
# visualise.plot_2d(plot_separate=False, iteration=2080)
# visualise.animate_concentration(conc_type="atomic", analytic_sol=False)

visualise.plot_concentration(plot_separate=False, conc_type="cells", analytic_sol=False)
# visualise.plot_concentration(plot_separate=False, conc_type="cells", analytic_sol=False,
# iteration=29)

# visualise.plot_h()

# for plane_ind in range(0, 61000, 100):
#     visualise.plot_3d(plot_separate=False, iteration=plane_ind, const_cam_pos=True)


# visualise.calculate_phase_size()
