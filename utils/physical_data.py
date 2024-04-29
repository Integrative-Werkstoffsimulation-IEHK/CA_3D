from math import *


DENSITY = {"Ni": 8908, "Fe": 7800, "Ti": 4506, "Cr": 7140, "Al": 2700, "N": 1.17, "O": 1, "None": 1}  # kg/m3
MOLAR_MASS = {"Ti": 0.0479, "Ni": 0.0587, "Fe": 0.0558, "Cr": 0.052, "Al": 0.027, "N": 0.014, "O": 0.016, "TiN": 0.0619,
              "None": 1}  # kg/mol


def get_diff_coeff(temperature, cond):
    diff_coeff = {"N in Ni20Cr2Ti Krupp": 4.7 * 10 ** -6 * exp(-125720 / (8.314 * (273 + temperature))),
                  "O in Ni at 1000 Smithells_Ransley": 2.4 * 10**-13,
                  "O in Ni Krupp": 4.9 * 10 ** -6 * exp(-164000 / (8.314 * (273 + temperature))),
                  "N in Ni Krupp": 1.99 * 10 ** -5 * exp(-132640 / (8.314 * (273 + temperature))),
                  "Al in Ni Krupp": 1.85 * 10 ** -4 * exp(-260780 / (8.314 * (273 + temperature))),
                  "Cr in Ni Krupp": 5.2 * 10 ** -4 * exp(-289000 / (8.314 * (273 + temperature))),
                  "Ti in Ni Krupp": 4.1 * 10 ** -4 * exp(-275000 / (8.314 * (273 + temperature))),
                  "N in Ni Katrin PHD": 1.99 * 10 ** -5 * exp(-132640 / (8.314 * (273 + temperature))),
                  "N in Ni Savva at 1020": 5 * 10 ** -11,
                  "Ti in Ni Savva at 1020": 5 * 10 ** -15,
                  "N in alfa-Fe Rozendaal": 6.6 * 10 ** -7 * exp(-77900 / (8.314 * (273 + temperature))),
                  "Test": 1 * (10 ** -13),
                  "Test_slower": 1.4 * (10 ** -9),
                  "Ti in Ni Krupp boost": 2.13 * 10 ** -13,
                  "Test Diffusion in precipitation": 6.18 * 10 ** -20,
                  "N in FeAlMn": 2.1 * 10 ** -8 * exp(-93517 / (8.314 * (273 + temperature))),
                  "Al in FeAlMn": 2.1 * 10 ** -10 * exp(-93517 / (8.314 * (273 + temperature))),
                  "Cr in NiCr": 0.03 * exp(-40800/(8.314 * (273 + temperature))),
                  "N in NiCr at 800°C": 6.7 * 10 ** -11,
                  # scales_______________________________________________
                  "O in Cr2O3 from [O in Cr2O3]": 15.9 * 10 ** -4 * exp((-100800 * 4.184) / (8.314 * (273 + temperature))),
                  "Cr in Cr2O3 from [Cr in Cr2O3]": 0.137 * 10 ** -4 * exp((-61100 * 4.184) / (8.314 * (273 + temperature))),
                  "None": 1 * (10 ** -13)
                  }
    return diff_coeff[cond]
