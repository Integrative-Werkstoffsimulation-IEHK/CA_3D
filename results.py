from visualisation import *
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()
database_name = filedialog.askopenfilename()

visualise = Visualisation(database_name)

visualise.animate_3d(animate_separate=False, const_cam_pos=False)
#
# visualise.plot_3d(plot_separate=False, const_cam_pos=False)
# visualise.plot_3d(plot_separate=False, const_cam_pos=False, iteration=20000)

visualise.animate_2d(plot_separate=False)

visualise.plot_2d(plot_separate=False)
# visualise.plot_2d(plot_separate=False, iteration=20000)

# visualise.animate_concentration(conc_type="cells", analytic_sol=False)

# visualise.plot_concentration(plot_separate=False, conc_type="cells", analytic_sol=False)
# visualise.plot_concentration(plot_separate=False, conc_type="cells", analytic_sol=False, iteration=956)

# visualise.plot_h()

