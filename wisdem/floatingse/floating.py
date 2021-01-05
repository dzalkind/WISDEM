import numpy as np
import openmdao.api as om
from wisdem.floatingse.member import Member
from wisdem.floatingse.map_mooring import MapMooring
from wisdem.floatingse.floating_frame import FloatingFrame

# from wisdem.floatingse.substructure import Substructure, SubstructureGeometry


class FloatingSE(om.Group):
    def initialize(self):
        self.options.declare("modeling_options")

    def setup(self):
        opt = self.options["modeling_options"]

        # self.set_input_defaults("mooring_type", "chain")
        # self.set_input_defaults("anchor_type", "SUCTIONPILE")
        # self.set_input_defaults("loading", "hydrostatic")
        # self.set_input_defaults("wave_period_range_low", 2.0, units="s")
        # self.set_input_defaults("wave_period_range_high", 20.0, units="s")
        # self.set_input_defaults("cd_usr", -1.0)
        # self.set_input_defaults("zref", 100.0)
        # self.set_input_defaults("number_of_offset_columns", 0)
        # self.set_input_defaults("material_names", ["steel"])

        n_member = opt["floating"]["n_member"]
        mem_prom = [
            "E_mat",
            "G_mat",
            "sigma_y_mat",
            "rho_mat",
            "rho_water",
            "unit_cost_mat",
            "material_names",
            "painting_cost_rate",
            "labor_cost_rate",
        ]
        # mem_prom += ["Uref", "zref", "shearExp", "z0", "cd_usr", "cm", "beta_wind", "rho_air", "mu_air", "beta_water",
        #            "rho_water", "mu_water", "Uc", "Hsig_wave","Tsig_wave","rho_water","water_depth"]
        for k in range(n_member):
            self.add_subsystem(
                "member" + str(k),
                Member(modeling_options=opt, member_options=opt["floating"]["member"][k]),
                promotes=mem_prom,
            )

        # Next run MapMooring
        self.add_subsystem("mm", MapMooring(modeling_options=opt["mooring"]), promotes=["*"])

        # Add in the connecting truss
        self.add_subsystem("load", FloatingFrame(modeling_options=opt), promotes=["*"])

        # Evaluate system constraints
        # self.add_subsystem("cons", FloatingConstraints(modeling_options=opt), promotes=["*"])

        # Connect all input variables from all models
        mem_vars = [
            "nodes_xyz",
            "nodes_r",
            "section_A",
            "section_Asx",
            "section_Asy",
            "section_Ixx",
            "section_Iyy",
            "section_Izz",
            "section_rho",
            "section_E",
            "section_G",
            "idx_cb",
            "buoyancy_force",
            "displacement",
            "center_of_buoyancy",
            "center_of_mass",
            "total_mass",
            "total_cost",
            "Awater",
            "Iwater",
            "added_mass",
        ]
        for k in range(n_member):
            for var in mem_vars:
                self.connect("member" + str(k) + "." + var, "member" + str(k) + ":" + var)

        """
        self.connect("max_offset_restoring_force", "mooring_surge_restoring_force")
        self.connect("operational_heel_restoring_force", "mooring_pitch_restoring_force")
        """


if __name__ == "__main__":
    npts = 5

    opt = {}
    opt["floating"] = {}
    opt["floating"]["n_member"] = 1
    opt["floating"]["member"] = [{}]
    opt["floating"]["member"][0]["n_height"] = npts
    opt["floating"]["member"][0]["n_bulkhead"] = 4
    opt["floating"]["member"][0]["n_layers"] = 1
    opt["floating"]["member"][0]["n_ballast"] = 0
    opt["floating"]["member"][0]["n_ring"] = 10
    opt["floating"]["member"][0]["n_axial"] = 1
    opt["floating"]["tower"] = {}
    opt["floating"]["tower"]["n_height"] = npts
    opt["floating"]["tower"]["n_bulkhead"] = 2
    opt["floating"]["tower"]["n_layers"] = 1
    opt["floating"]["tower"]["n_ballast"] = 0
    opt["floating"]["tower"]["n_ring"] = 0
    opt["floating"]["tower"]["n_axial"] = 0
    opt["floating"]["frame3dd"] = {}
    opt["floating"]["frame3dd"]["shear"] = True
    opt["floating"]["frame3dd"]["geom"] = False
    opt["floating"]["frame3dd"]["tol"] = 1e-6
    opt["floating"]["gamma_f"] = 1.35  # Safety factor on loads
    opt["floating"]["gamma_m"] = 1.3  # Safety factor on materials
    opt["floating"]["gamma_n"] = 1.0  # Safety factor on consequence of failure
    opt["floating"]["gamma_b"] = 1.1  # Safety factor on buckling
    opt["floating"]["gamma_fatigue"] = 1.755  # Not used
    opt["floating"]["run_modal"] = True  # Not used
    opt["mooring"] = {}
    opt["mooring"]["n_nodes"] = 3
    opt["mooring"]["n_anchors"] = 3

    opt["materials"] = {}
    opt["materials"]["n_mat"] = 2

    prob = om.Problem()
    prob.model = FloatingSE(modeling_options=opt)
    prob.setup()

    # Material properties
    prob["rho_mat"] = np.array([7850.0, 5000.0])  # Steel, ballast slurry [kg/m^3]
    prob["E_mat"] = 200e9 * np.ones((2, 3))  # Young's modulus [N/m^2]
    prob["G_mat"] = 79.3e9 * np.ones((2, 3))  # Shear modulus [N/m^2]
    prob["sigma_y_mat"] = 3.45e8 * np.ones(2)  # Elastic yield stress [N/m^2]
    prob["unit_cost_mat"] = np.array([2.0, 1.0])
    prob["material_names"] = ["steel", "slurry"]

    # Mass and cost scaling factors
    prob["labor_cost_rate"] = 1.0  # Cost factor for labor time [$/min]
    prob["painting_cost_rate"] = 14.4  # Cost factor for column surface finishing [$/m^2]
    prob["member0.outfitting_factor_in"] = 0.0  # Fraction of additional outfitting mass for each column

    # Column geometry
    prob["member0.grid_axial_joints"] = [0.384615]  # Fairlead at 70m
    # prob["member0.ballast_grid"] = np.empy((0,2))
    # prob["member0.ballast_volume"] = np.empty(0)
    prob["member0.height"] = np.sum([49.0, 59.0, 8.0, 14.0])  # Length of each section [m]
    prob["member0.s"] = np.cumsum([0.0, 49.0, 59.0, 8.0, 14.0]) / prob["member0.height"]
    prob["member0.outer_diameter_in"] = np.array(
        [9.4, 9.4, 9.4, 6.5, 6.5]
    )  # Diameter at each section node (linear lofting between) [m]
    prob["member0.layer_thickness"] = 0.05 * np.ones(
        (1, npts)
    )  # Shell thickness at each section node (linear lofting between) [m]
    prob["member0.layer_materials"] = ["steel"]
    prob["member0.ballast_materials"] = ["slurry", "seawater"]

    prob["member0.joint2"] = np.array([0.0, 0.0, 10.0])  # Freeboard=10
    prob["member0.joint1"] = np.array([0.0, 0.0, 10.0 - prob["member0.height"]])  # Freeboard=10
    prob["member0.bulkhead_thickness"] = 0.05 * np.ones(4)  # Locations of internal bulkheads
    prob["member0.bulkhead_grid"] = np.array([0.0, 0.37692308, 0.89230769, 1.0])  # Thickness of internal bulkheads

    # Column ring stiffener parameters
    prob["member0.ring_stiffener_web_height"] = 0.10
    prob["member0.ring_stiffener_web_thickness"] = 0.04
    prob["member0.ring_stiffener_flange_width"] = 0.10
    prob["member0.ring_stiffener_flange_thickness"] = 0.02
    prob["member0.ring_stiffener_spacing"] = 2.28

    # Mooring parameters
    prob["line_diameter"] = 0.09  # Diameter of mooring line/chain [m]
    prob["line_length"] = 300 + 902.2  # Unstretched mooring line length
    prob["line_mass_density_coeff"] = 19.9e3
    prob["line_stiffness_coeff"] = 8.54e10
    prob["line_breaking_load_coeff"] = 176972.7
    prob["line_cost_rate_coeff"] = 3.415e4
    prob["fairlead_radius"] = 10.0  # Offset from shell surface for mooring attachment [m]
    prob["fairlead"] = 70.0
    prob["anchor_radius"] = 853.87  # Distance from centerline to sea floor landing [m]
    prob["anchor_cost"] = 1e5

    # Mooring constraints
    prob["max_surge_fraction"] = 0.1  # Max surge/sway offset [m]
    prob["survival_heel"] = 10.0  # Max heel (pitching) angle [deg]
    prob["operational_heel"] = 5.0  # Max heel (pitching) angle [deg]

    # Set environment to that used in OC3 testing campaign
    # prob["rho_air"] = 1.226  # Density of air [kg/m^3]
    # prob["mu_air"] = 1.78e-5  # Viscosity of air [kg/m/s]
    prob["rho_water"] = 1025.0  # Density of water [kg/m^3]
    # prob["mu_water"] = 1.08e-3  # Viscosity of water [kg/m/s]
    prob["water_depth"] = 320.0  # Distance to sea floor [m]
    # prob["Hsig_wave"] = 0.0  # Significant wave height [m]
    # prob["Tsig_wave"] = 1e3  # Wave period [s]
    # prob["shearExp"] = 0.11  # Shear exponent in wind power law
    # prob["cm"] = 2.0  # Added mass coefficient
    # prob["Uc"] = 0.0  # Mean current speed
    # prob["yaw"] = 0.0  # Turbine yaw angle
    # prob["beta_wind"] = prob["beta_wave"] = 0.0
    # prob["cd_usr"] = -1.0  # Compute drag coefficient
    # prob["Uref"] = 10.0
    # prob["zref"] = 100.0

    # Porperties of turbine tower
    nTower = prob.model.options["modeling_options"]["floating"]["tower"]["n_height"] - 1
    prob["tower.height"] = prob["hub_height"] = 77.6
    prob["tower.s"] = np.linspace(0.0, 1.0, nTower + 1)
    prob["tower.outer_diameter_in"] = np.linspace(6.5, 3.87, nTower + 1)
    prob["tower.layer_thickness"] = np.linspace(0.027, 0.019, nTower + 1).reshape((1, nTower + 1))
    prob["tower.layer_materials"] = ["steel"]
    prob["tower.outfitting_factor"] = 1.07

    # Properties of rotor-nacelle-assembly (RNA)
    prob["rna_mass"] = 350e3  # Mass [kg]
    prob["rna_I"] = 1e5 * np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    prob["rna_cg"] = np.zeros(3)
    prob["rna_F"] = np.zeros(3)
    prob["rna_M"] = np.zeros(3)

    prob.run_model()
