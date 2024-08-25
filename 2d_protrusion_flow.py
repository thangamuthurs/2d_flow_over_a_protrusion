import numpy as np
from sympy import Symbol, Eq
import modulus.sym
from modulus.sym.hydra import to_absolute_path, ModulusConfig, instantiate_arch
from modulus.sym.solver import Solver
from modulus.sym.domain import Domain
from modulus.sym.geometry.primitives_2d import Rectangle, Line, Channel2D
from modulus.sym.utils.sympy.functions import parabola
from modulus.sym.eq.pdes.navier_stokes import NavierStokes
from modulus.sym.eq.pdes.basic import NormalDotVec
from modulus.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
    IntegralBoundaryConstraint,
)
from modulus.sym.domain.validator import PointwiseValidator
from modulus.sym.key import Key
from modulus.sym.node import Node
import scipy.interpolate
import matplotlib.pyplot as plt
import pandas as pd
from modulus.sym.utils.io import (
    csv_to_dict,
    ValidatorPlotter,
    InferencerPlotter,
)

@modulus.sym.main(config_path="/content/drive/MyDrive/2d_chip", config_name = "2d_chip_config.yaml")
def run(cfg: ModulusConfig) -> None:
    
#Making list of nodes to unroll the graph
    ns = NavierStokes(nu= 0.02, rho = 1, dim = 2, time = False)
    normal_dot_vel = NormalDotVec(["u" , "v"])                              
    flow_net = instantiate_arch(
        input_keys = [Key("x"),Key("y")], 
        output_keys = [Key("u"), Key("v"), Key("p")],
        cfg= cfg.arch.fully_connected,
    )
    nodes = (
        ns.make_nodes() 
        + normal_dot_vel.make_nodes()                                       
        + [flow_net.make_node(name = "flow_network", jit = cfg.jit)]
    )
    
#Dimensions:
    channel_length= (-2.5,2.5)
    channel_width = (-0.5,0.5)
    chip_width = 1.0
    chip_height = 0.6
    chip_pos = -1.0
    inlet_vel = 1.5
    
#defining symbols:
    x,y = Symbol("x"), Symbol("y")
    
    channel = Channel2D(
        (channel_length[0],channel_width[0]),(channel_length[1],channel_width[1])
    )
    
    inlet = Line(
        (channel_length[0],channel_width[0]),
        (channel_length[0],channel_width[1]),
        normal = 1,                                                     
    )
        
    outlet = Line(
        (channel_length[1],channel_width[0]),
        (channel_length[1],channel_width[1]),
        normal = 1,
    )
    
    rec = Rectangle(
        (chip_pos,channel_width[0]),
        (chip_pos+chip_width,channel_width[0]+0.6),
    )
    
    flow_rec = Rectangle(
        (chip_pos -0.25 , channel_width[0]),
        (chip_pos +chip_width + 0.25 , channel_width[1]),
    )
    geo = channel - rec                                             
    geo_hr = geo & flow_rec                                        
    geo_lr = geo - flow_rec                                         
    
#Integral continuity plane to speed up convergence              
    x_pos = Symbol("x_pos")
    integral_line = Line( (x_pos,channel_width[0]) , (x_pos,channel_width[1]) , 1)                  
    x_pos_range = {
        x_pos: lambda batch_size: np.full(
            (batch_size,1), np.random.uniform(channel_length[0],channel_length[1])                 
        )                                                                                           
    }                                                                                               
                                                                                                    
    
#Putting Constraints
    domain = Domain()
    
  #inlet:
    inlet_parabola = parabola(y, channel_width[0], channel_width[1],inlet_vel)              
    inlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = inlet,
        outvar = {"u": inlet_parabola , "v":0},
        batch_size = cfg.batch_size.inlet,
    )
    domain.add_constraint(inlet,"Inlet")
  
  #outlet  
    outlet = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = outlet,
        outvar = {"p":0},
        batch_size = cfg.batch_size.outlet,
        criteria = Eq(x , channel_length[1])                                                   
    )
    domain.add_constraint(outlet , "Outlet")
  
 #no_slip
    no_slip = PointwiseBoundaryConstraint(
        nodes = nodes,
        geometry = geo,                                                                     
        outvar = {"u":0,"v":0},
        batch_size = cfg.batch_size.no_slip,
    )
    domain.add_constraint(no_slip,"no_slip")
    
 #interior_lr
    interior_lr = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geo_lr,                                                                   
        outvar = {"continuity":0,"momentum_x":0, "momentum_y":0},
        batch_size = cfg.batch_size.interior_lr,
        #bounds = {"x": channel_length, "y": channel_width},
        lambda_weighting={
            "continuity" : 2 * Symbol("sdf"),                                                
            "momentum_x" : 2 * Symbol("sdf"),
            "momentum_y" : 2 * Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_lr, "interior_lr")
    
  #interior_hr
    interior_hr = PointwiseInteriorConstraint(
        nodes = nodes,
        geometry = geo_hr,                                                                   
        outvar = {"continuity":0,"momentum_x":0, "momentum_y":0},
        batch_size = cfg.batch_size.interior_lr,
        #bounds = {"x": channel_length, "y": channel_width},
        lambda_weighting = {
            "continuity" : Symbol("sdf"),
            "momentum_x" : Symbol("sdf"),
            "momentum_y" : Symbol("sdf"),
        },
    )
    domain.add_constraint(interior_hr, "interior_hr")
    
  #integral continuity
    def integral_criteria(invar, params):
        sdf = geo.sdf(invar, params)
        return np.greater(sdf["sdf"], 0)

    integral_continuity = IntegralBoundaryConstraint(
        nodes=nodes,
        geometry=integral_line,                                             
        outvar={"normal_dot_vel": 1},                                      
        batch_size=cfg.batch_size.num_integral_continuity,                  
        integral_batch_size=cfg.batch_size.integral_continuity,             
        lambda_weighting={"normal_dot_vel": 1},                             
        criteria=integral_criteria,
        parameterization=x_pos_range,
    )
    domain.add_constraint(integral_continuity, "integral_continuity")
    
#Custom Validator
    from modulus.sym.utils.io.plotter import ValidatorPlotter
    class CustomValidatorPlotter(ValidatorPlotter):
        def __call__(self, invar, true_outvar, pred_outvar):
            "Custom plotting function for validator"

            # get input variables
            x,y = invar["x"][:,0], invar["y"][:,0]
            extent = (x.min(), x.max(), y.min(), y.max())

            # get and interpolate output variable
            u_true,u_pred = true_outvar["p"][:,0], pred_outvar["p"][:,0]
      
            u_true,u_pred = self.interpolate_output(x, y,
                                                    [u_true, u_pred],
                                                    extent,
            )
            # make plot
            f = plt.figure(figsize=(14,4), dpi=100)
            plt.suptitle("2d Chip flow: True vs PINN solutions")
            plt.subplot(1,3,1)
            plt.title("PINN solution (p)")
            plt.imshow(u_true.T, origin="lower", extent=extent, vmin=-2.8, vmax=8.3,cmap='jet')
            plt.xlabel("x"); plt.ylabel("y")
            plt.colorbar()
            #plt.vlines(-1, -0.5, 0.1, color="k", lw=10, label="No slip boundary")
            #plt.vlines(0, -0.5, 0.5, color="k", lw=10)
            #plt.hlines(0.5, -2.5, 2.5, color="k", lw=10)
            #plt.hlines(-0.5, -2.5, -1, color="k", lw=10)
            #plt.hlines(-0.5, 0, 0.5, color="k", lw=10)
            #plt.hlines(0.1, -1, 0, color="k", lw=10)
            plt.legend(loc="lower right")
            plt.subplot(1,3,2)
            plt.title("PINN solution (p)")
            plt.imshow(u_pred.T, origin="lower", extent=extent, vmin=-2.8, vmax=8.3,cmap='jet')
            plt.xlabel("x"); plt.ylabel("y")
            plt.colorbar()
            plt.subplot(1,3,3)
            plt.title("Difference)")
            plt.imshow((u_true - u_pred).T, origin="lower", extent=extent, vmin=-4.2, vmax=0.4,cmap='jet')
            plt.xlabel("x"); plt.ylabel("y")
            plt.colorbar()
            plt.tight_layout()

            return [(f, "custom_plot"),]

        @staticmethod
        def interpolate_output(x, y, us, extent):
            "Interpolates irregular points onto a mesh"

            # define mesh to interpolate onto
            xyi = np.meshgrid(
                np.linspace(extent[0], extent[1], 300),
                np.linspace(extent[2], extent[3], 300),
                indexing="ij",
            )
            #x_remove = np.logical_and (x>=-1, x<=-0.5)
            #y_remove = np.logical_and(y>=-0.5, y<=0.1)
            
            #xyi[0][np.ix_(x_remove,y_remove)] = np.nan
            #xyi[1][np.ix_(x_remove,y_remove)] = np.nan

            x_remove = (xyi[0] >= -1.0) & (xyi[0] <= 0)
            y_remove = (xyi[1] >= -0.5) & (xyi[1] <= 0.1)
            xyi[0][x_remove & y_remove] = np.nan
            xyi[1][x_remove & y_remove] = np.nan
            # linearly interpolate points onto mesh
            us = [scipy.interpolate.griddata(
                (x, y), (v), tuple(xyi)
                )
                for v in us]
            
            return us
            
# add validation data
    mapping = {"Points:0": "x", "Points:1": "y", "U:0": "u", "U:1": "v", "p": "p"}
    openfoam_var = csv_to_dict(to_absolute_path("/content/drive/MyDrive/2d_chip/2D_chip_fluid0.csv"), mapping)
    openfoam_var["x"] -= 2.5  # TODO normalize pos
    openfoam_var["y"] -= 0.5  # TODO normalize pos
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u", "v", "p"]
    }
    openfoam_validator = PointwiseValidator(
        invar=openfoam_invar_numpy, 
        true_outvar=openfoam_outvar_numpy, 
        nodes=nodes,
        batch_size = 1024,
        plotter = CustomValidatorPlotter(),
    )
    domain.add_validator(openfoam_validator)
    
    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()
    
if __name__ == "__main__":
    run()