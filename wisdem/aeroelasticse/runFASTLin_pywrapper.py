"""
A basic python script that demonstrates how to use the FST8 reader, writer, and wrapper in a purely
python setting. These functions are constructed to provide a simple interface for controlling FAST
programmatically with minimal additional dependencies.
"""
# Hacky way of doing relative imports
from __future__ import print_function
import os, sys, time, pathlib
import multiprocessing as mp
# sys.path.insert(0, os.path.abspath(".."))

from wisdem.aeroelasticse.FAST_reader import InputReader_Common, InputReader_OpenFAST, InputReader_FAST7
from wisdem.aeroelasticse.FAST_writer import InputWriter_Common, InputWriter_OpenFAST, InputWriter_FAST7
from wisdem.aeroelasticse.FAST_wrapper import FastWrapper
from wisdem.aeroelasticse.FAST_post import return_timeseries
from wisdem.aeroelasticse.Util.FileTools import save_yaml, load_yaml

import numpy as np

# pCrunch Modules and instantiation
import matplotlib.pyplot as plt 
from ROSCO_toolbox import utilities as ROSCO_utilites
fast_io = ROSCO_utilites.FAST_IO()
fast_pl = ROSCO_utilites.FAST_Plots()

# WISDEM modules
from wisdem.aeroelasticse.Util import FileTools

# Batch Analysis
from pCrunch import pdTools
from pCrunch import Processing, Analysis

class runFAST_pywrapper(object):

    def __init__(self, **kwargs):
        self.FAST_ver = 'OPENFAST' #(FAST7, FAST8, OPENFAST)

        self.FAST_exe = None
        self.FAST_InputFile = None
        self.FAST_directory = None
        self.FAST_runDirectory = None
        self.FAST_namingOut = None
        self.read_yaml = False
        self.write_yaml = False
        self.fst_vt = {}
        self.case = {}                  # dictionary of variable values to change
        self.channels = {}              # dictionary of output channels to change
        self.debug_level   = 0
        self.dev_branch = False

        # Optional population class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(runFAST_pywrapper, self).__init__()

    def execute(self):
        # FAST version specific initialization
        if self.FAST_ver.lower() == 'fast7':
            reader = InputReader_FAST7(FAST_ver=self.FAST_ver)
            writer = InputWriter_FAST7(FAST_ver=self.FAST_ver)
        elif self.FAST_ver.lower() in ['fast8','openfast']:
            reader = InputReader_OpenFAST(FAST_ver=self.FAST_ver)
            writer = InputWriter_OpenFAST(FAST_ver=self.FAST_ver)
        wrapper = FastWrapper(FAST_ver=self.FAST_ver, debug_level=self.debug_level)

        # Read input model, FAST files or Yaml
        if self.fst_vt == {}:
            if self.read_yaml:
                reader.FAST_yamlfile = self.FAST_yamlfile_in
                reader.read_yaml()
            else:
                reader.FAST_InputFile = self.FAST_InputFile
                reader.FAST_directory = self.FAST_directory
                reader.dev_branch = self.dev_branch
                reader.execute()
        
            # Initialize writer variables with input model
            writer.fst_vt = reader.fst_vt
        else:
            writer.fst_vt = self.fst_vt
        writer.FAST_runDirectory = self.FAST_runDirectory
        writer.FAST_namingOut = self.FAST_namingOut
        writer.dev_branch = self.dev_branch
        # Make any case specific variable changes
        if self.case:
            writer.update(fst_update=self.case)
        # Modify any specified output channels
        if self.channels:
            writer.update_outlist(self.channels)
        # Write out FAST model
        writer.execute()
        if self.write_yaml:
            writer.FAST_yamlfile = self.FAST_yamlfile_out
            writer.write_yaml()

        # Run FAST
        wrapper.FAST_exe = self.FAST_exe
        wrapper.FAST_InputFile = os.path.split(writer.FAST_InputFileOut)[1]
        wrapper.FAST_directory = os.path.split(writer.FAST_InputFileOut)[0]
        wrapper.execute()

        FAST_Output = os.path.join(wrapper.FAST_directory, wrapper.FAST_InputFile[:-3]+'outb')
        return FAST_Output

class runFAST_pywrapper_batch(object):

    def __init__(self, **kwargs):

        self.FAST_ver           = 'OpenFAST'
        self.FAST_exe           = None
        self.FAST_InputFile     = None
        self.FAST_directory     = None
        self.FAST_runDirectory  = None
        self.debug_level        = 0
        self.dev_branch         = False

        self.read_yaml          = False
        self.FAST_yamlfile_in   = ''
        self.fst_vt             = {}
        self.write_yaml         = False
        self.FAST_yamlfile_out  = ''

        self.case_list          = []
        self.case_name_list     = []
        self.channels           = {}

        self.post               = None

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(runFAST_pywrapper_batch, self).__init__()

        
    def run_serial(self):
        # Run batch serially

        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)

        out = [None]*len(self.case_list)
        for i, (case, case_name) in enumerate(zip(self.case_list, self.case_name_list)):
            out[i] = eval(case, case_name, self.FAST_ver, self.FAST_exe, self.FAST_runDirectory, self.FAST_InputFile, self.FAST_directory, self.read_yaml, self.FAST_yamlfile_in, self.fst_vt, self.write_yaml, self.FAST_yamlfile_out, self.channels, self.debug_level, self.dev_branch, self.post)

        return out

    def run_multi(self, cores=None):
        # Run cases in parallel, threaded with multiprocessing module

        if not os.path.exists(self.FAST_runDirectory):
            os.makedirs(self.FAST_runDirectory)

        if not cores:
            cores = mp.cpu_count()
        pool = mp.Pool(cores)

        case_data_all = []
        for i in range(len(self.case_list)):
            case_data = []
            case_data.append(self.case_list[i])
            case_data.append(self.case_name_list[i])
            case_data.append(self.FAST_ver)
            case_data.append(self.FAST_exe)
            case_data.append(self.FAST_runDirectory)
            case_data.append(self.FAST_InputFile)
            case_data.append(self.FAST_directory)
            case_data.append(self.read_yaml)
            case_data.append(self.FAST_yamlfile_in)
            case_data.append(self.fst_vt)
            case_data.append(self.write_yaml)
            case_data.append(self.FAST_yamlfile_out)
            case_data.append(self.channels)
            case_data.append(self.debug_level)
            case_data.append(self.dev_branch)
            case_data.append(self.post)

            case_data_all.append(case_data)

        output = pool.map(eval_multi, case_data_all)
        pool.close()
        pool.join()

        return output

    def run_mpi(self, mpi_comm_map_down):
        # Run in parallel with mpi
        from mpi4py import MPI

        # mpi comm management
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        sub_ranks = mpi_comm_map_down[rank]
        size = len(sub_ranks)

        N_cases = len(self.case_list)
        N_loops = int(np.ceil(float(N_cases)/float(size)))
        
        # file management
        if not os.path.exists(self.FAST_runDirectory) and rank == 0:
            os.makedirs(self.FAST_runDirectory)

        case_data_all = []
        for i in range(N_cases):
            case_data = []
            case_data.append(self.case_list[i])
            case_data.append(self.case_name_list[i])
            case_data.append(self.FAST_ver)
            case_data.append(self.FAST_exe)
            case_data.append(self.FAST_runDirectory)
            case_data.append(self.FAST_InputFile)
            case_data.append(self.FAST_directory)
            case_data.append(self.read_yaml)
            case_data.append(self.FAST_yamlfile_in)
            case_data.append(self.fst_vt)
            case_data.append(self.write_yaml)
            case_data.append(self.FAST_yamlfile_out)
            case_data.append(self.channels)
            case_data.append(self.debug_level)
            case_data.append(self.dev_branch)
            case_data.append(self.post)

            case_data_all.append(case_data)

        output = []
        for i in range(N_loops):
            idx_s    = i*size
            idx_e    = min((i+1)*size, N_cases)

            for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
                data   = [eval_multi, case_data]
                rank_j = sub_ranks[j]
                comm.send(data, dest=rank_j, tag=0)

            # for rank_j in sub_ranks:
            for j, case_data in enumerate(case_data_all[idx_s:idx_e]):
                rank_j = sub_ranks[j]
                data_out = comm.recv(source=rank_j, tag=1)
                output.append(data_out)

        return output



def eval(case, case_name, FAST_ver, FAST_exe, FAST_runDirectory, FAST_InputFile, FAST_directory, read_yaml, FAST_yamlfile_in, fst_vt, write_yaml, FAST_yamlfile_out, channels, debug_level, dev_branch, post):
    # Batch FAST pyWrapper call, as a function outside the runFAST_pywrapper_batch class for pickle-ablility

    fast = runFAST_pywrapper(FAST_ver=FAST_ver)
    fast.FAST_exe           = FAST_exe
    fast.FAST_InputFile     = FAST_InputFile
    fast.FAST_directory     = FAST_directory
    fast.FAST_runDirectory  = FAST_runDirectory
    fast.dev_branch         = dev_branch

    fast.read_yaml          = read_yaml
    fast.FAST_yamlfile_in   = FAST_yamlfile_in
    fast.fst_vt             = fst_vt
    fast.write_yaml         = write_yaml
    fast.FAST_yamlfile_out  = FAST_yamlfile_out

    fast.FAST_namingOut     = case_name
    fast.case               = case
    fast.channels           = channels
    fast.debug_level        = debug_level

    FAST_Output = fast.execute()

    # Post process
    if post:
        out = post(FAST_Output)
    else:
        out = []

    return out

def eval_multi(data):
    # helper function for running with multiprocessing.Pool.map
    # converts list of arguement values to arguments
    return eval(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15])

class LinearFAST(runFAST_pywrapper_batch):

    def __init__(self, **kwargs):

        self.FAST_ver           = 'OpenFAST'
        self.FAST_exe           = None
        self.FAST_InputFile     = None
        self.FAST_directory     = None
        self.FAST_runDirectory  = None
        self.debug_level        = 0
        self.dev_branch         = True

        self.read_yaml          = False
        self.FAST_yamlfile_in   = ''
        self.fst_vt             = {}
        self.write_yaml         = False
        self.FAST_yamlfile_out  = ''

        self.case_list          = []
        self.case_name_list     = []
        self.channels           = {}

        self.post               = None

        # Optional population of class attributes from key word arguments
        for (k, w) in kwargs.items():
            try:
                setattr(self, k, w)
            except:
                pass

        super(LinearFAST, self).__init__()

    def runFAST_steady(self):
        """ 
        Run batch of steady state cases for initial conditions, in serial or in parallel
        """

        self.FAST_runDirectory = self.FAST_steadyDirectory

        case_inputs = {}
        case_inputs[("Fst","TMax")] = {'vals':[self.TMax], 'group':0}
        case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
        case_inputs[("Fst","OutFileFmt")] = {'vals':[2], 'group':0}

        # Wind Speeds
        case_inputs[("InflowWind","HWindSpeed")] = {'vals':self.WindSpeeds, 'group':1}

        # # Initial Conditions: less important, trying to find them here
        # case_inputs[("ElastoDyn","RotSpeed")] = {'vals':[7.55], 'group':0}
        # case_inputs[("ElastoDyn","BlPitch1")] = {'vals':[3.823], 'group':0}
        # case_inputs[("ElastoDyn","BlPitch2")] = case_inputs[("ElastoDyn","BlPitch1")]
        # case_inputs[("ElastoDyn","BlPitch3")] = case_inputs[("ElastoDyn","BlPitch1")]
        
        from CaseGen_General import CaseGen_General
        case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=self.FAST_steadyDirectory, namebase='steady')

        self.case_list = case_list
        self.case_name_list = case_name_list

        if self.parallel:
            self.run_multi()
        else:
            self.run_serial()

        
    def postFAST_steady(self):
        """
        Post process results to get steady state information for all initial conditions at each wind speed
        Save as ss_ops.yaml for 
        """

        # Plot steady states vs wind speed
        PLOT = 0

        # Define input files paths
        output_dir      = self.FAST_steadyDirectory

        # Find all outfiles
        outfiles = []
        for file in os.listdir(output_dir):
            if file.endswith('.outb'):
                outfiles.append(os.path.join(output_dir,file))
            # elif file.endswith('.out') and not file.endswith('.MD.out'):  
            #     outfiles.append(os.path.join(output_dir,file))


        # Initialize processing classes
        fp = Processing.FAST_Processing()

        # Set some processing parameters
        fp.OpenFAST_outfile_list        = outfiles
        fp.t0                           = self.TMax - 400            # make sure this is less than simulation time
        fp.parallel_analysis            = False
        fp.results_dir                  = os.path.join(output_dir, 'stats')
        fp.verbose                      = True
        fp.save_LoadRanking             = True
        fp.save_SummaryStats            = True

        # Load and save statistics and load rankings
        stats =fp.batch_processing()


        windSortInd = np.argsort(stats[0]['Wind1VelX']['mean'])

        #            FAST output name,  FAST IC name
        ssChannels = [['Wind1VelX',     'Wind1VelX'],  
                    ['OoPDefl1',        'OoPDefl'],
                    ['IPDefl1',         'IPDefl'],
                    ['BldPitch1',       'BlPitch1'],
                    ['RotSpeed',        'RotSpeed'],
                    ['TTDspFA',         'TTDspFA'],
                    ['TTDspSS',         'TTDspSS'],
                    ['PtfmSurge',       'PtfmSurge'],
                    ['PtfmSway',        'PtfmSway'],
                    ['PtfmHeave',       'PtfmHeave'],
                    ['PtfmRoll',        'PtfmRoll'],
                    ['PtfmYaw',         'PtfmYaw'],
                    ['PtfmPitch',       'PtfmPitch'],
                    ]

        ssChanData = {}
        for iChan in ssChannels:
            try:
                ssChanData[iChan[1]] = np.array(stats[0][iChan[0]]['mean'])[windSortInd].tolist()
            except:
                print('Warning: ' + iChan + ' is is not in OutList')


        if PLOT:
            fig1 = plt.figure()
            ax1 = fig1.add_subplot(211)
            ax2 = fig1.add_subplot(212)

            ax1.plot(ssChanData['Wind1VelX'],ssChanData['BlPitch1'])
            ax2.plot(ssChanData['Wind1VelX'],ssChanData['RotSpeed'])


            fig2 = plt.figure()
            ax1 = fig2.add_subplot(411)
            ax2 = fig2.add_subplot(412)
            ax3 = fig2.add_subplot(413)
            ax4 = fig2.add_subplot(414)

            ax1.plot(ssChanData['Wind1VelX'],ssChanData['OoPDefl'])
            ax2.plot(ssChanData['Wind1VelX'],ssChanData['IPDefl'])
            ax3.plot(ssChanData['Wind1VelX'],ssChanData['TTDspFA'])
            ax4.plot(ssChanData['Wind1VelX'],ssChanData['TTDspSS'])

            fig3 = plt.figure()
            ax1 = fig3.add_subplot(611)
            ax2 = fig3.add_subplot(612)
            ax3 = fig3.add_subplot(613)
            ax4 = fig3.add_subplot(614)
            ax5 = fig3.add_subplot(615)
            ax6 = fig3.add_subplot(616)

            ax1.plot(ssChanData['Wind1VelX'],ssChanData['PtfmSurge'])
            ax2.plot(ssChanData['Wind1VelX'],ssChanData['PtfmSway'])
            ax3.plot(ssChanData['Wind1VelX'],ssChanData['PtfmHeave'])
            ax4.plot(ssChanData['Wind1VelX'],ssChanData['PtfmRoll'])
            ax5.plot(ssChanData['Wind1VelX'],ssChanData['PtfmPitch'])
            ax6.plot(ssChanData['Wind1VelX'],ssChanData['PtfmYaw'])

            plt.show()


        # output steady states to yaml
        save_yaml(output_dir,'ss_ops.yaml',ssChanData)




    def runFAST_linear(self):
        """ 
        Example of running a batch of cases, in serial or in parallel
        """

        ss_opFile = os.path.join(self.FAST_steadyDirectory,'ss_ops.yaml')
        self.FAST_runDirectory = self.FAST_linearDirectory

        ## Generate case list using General Case Generator
        ## Specify several variables that change independently or collectly
        case_inputs = {}
        case_inputs[("Fst","TMax")] = {'vals':[self.TMax], 'group':0}
        case_inputs[("Fst","Linearize")] = {'vals':['True'], 'group':0}
        case_inputs[("Fst","CalcSteady")] = {'vals':['True'], 'group':0}

        case_inputs[("Fst","OutFileFmt")] = {'vals':[2], 'group':0}
        case_inputs[("Fst","CompMooring")] = {'vals':[0], 'group':0}

        if not self.HydroStates:
            case_inputs[("Fst","CompHydro")] = {'vals':[0], 'group':0}
        
        # InflowWind
        case_inputs[("InflowWind","WindType")] = {'vals':[1], 'group':0}
        if not isinstance(self.WindSpeeds,list):
            self.WindSpeeds = [self.WindSpeeds]
        case_inputs[("InflowWind","HWindSpeed")] = {'vals':self.WindSpeeds, 'group':1}

        # AeroDyn Inputs
        case_inputs[("AeroDyn15","AFAeroMod")] = {'vals':[1], 'group':0}

        # Servodyn Inputs
        case_inputs[("ServoDyn","PCMode")] = {'vals':[0], 'group':0}
        case_inputs[("ServoDyn","VSContrl")] = {'vals':[1], 'group':0}

        # Hydrodyn Inputs, these need to be state-space (2), but they should work if 0
        case_inputs[("HydroDyn","ExctnMod")] = {'vals':[2], 'group':0}
        case_inputs[("HydroDyn","RdtnMod")] = {'vals':[2], 'group':0}
        case_inputs[("HydroDyn","DiffQTF")] = {'vals':[0], 'group':0}
        case_inputs[("HydroDyn","WvDiffQTF")] = {'vals':['False'], 'group':0}
        

        # Degrees-of-freedom: set all to False & enable those defined in self
        case_inputs[("ElastoDyn","FlapDOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","FlapDOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","EdgeDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TeetDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","DrTrDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","GenDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","YawDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwFADOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF1")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","TwSSDOF2")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmSgDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmSwDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmHvDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmRDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmPDOF")] = {'vals':['False'], 'group':0}
        case_inputs[("ElastoDyn","PtfmYDOF")] = {'vals':['False'], 'group':0}

        for dof in self.DOFs:
            case_inputs[("ElastoDyn",dof)] = {'vals':['True'], 'group':0}
        
        # Initial Conditions
        ss_ops = load_yaml(ss_opFile)
        uu = ss_ops['Wind1VelX']

        for ic in ss_ops:
            if ic != 'Wind1VelX':
                case_inputs[("ElastoDyn",ic)] = {'vals': np.interp(case_inputs[("InflowWind","HWindSpeed")]['vals'],uu,ss_ops[ic]).tolist(), 'group': 1}

        case_inputs[('ElastoDyn','BlPitch2')] = case_inputs[('ElastoDyn','BlPitch1')]
        case_inputs[('ElastoDyn','BlPitch3')] = case_inputs[('ElastoDyn','BlPitch1')]

        # Gen Speed to track
        # set for now and update with GB ratio next
        RefGenSpeed = 0.95 * np.array(case_inputs[('ElastoDyn','RotSpeed')]['vals']) * self.GBRatio
        case_inputs[('ServoDyn','VS_RtGnSp')] = {'vals': RefGenSpeed.tolist(), 'group': 1}


        # Lin Times
        # rotPer = 60. / np.array(case_inputs['ElastoDyn','RotSpeed']['vals'])
        # linTimes = np.linspace(self.TMax-100,self.TMax-100 + rotPer,num = self.NLinTimes, endpoint=False)
        # linTimeStrings = []

        # if linTimes.ndim == 1:
        #     linTimeStrings = np.array_str(linTimes,max_line_width=9000,precision=3)[1:-1]
        # else:
        #     for iCase in range(0,linTimes.shape[1]):
        #         linTimeStrings.append(np.array_str(linTimes[:,iCase],max_line_width=9000,precision=3)[1:-1])
        
        case_inputs[("Fst","NLinTimes")] = {'vals':[self.NLinTimes], 'group':0}

        # Trim case depends on rated wind speed (torque below-rated, pitch above)
        TrimCase = 3 * np.ones(len(self.WindSpeeds),dtype=int)
        TrimCase[np.array(self.WindSpeeds) < self.v_rated] = 2

        case_inputs[("Fst","TrimCase")] = {'vals':TrimCase.tolist(), 'group':1}


        case_inputs[("Fst","TrimTol")] = {'vals':[1e-5], 'group':0}
        

        # Generate Cases
        from CaseGen_General import CaseGen_General
        case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix=self.FAST_linearDirectory, namebase='lin')

        self.case_list = case_list
        self.case_name_list = case_name_list

        if self.parallel:
            self.run_multi()
        else:
            self.run_serial()
        


if __name__=="__main__":

    linear = LinearFAST(FAST_ver='OpenFAST', dev_branch=True);

    # fast info
    linear.FAST_exe                 = '/Users/dzalkind/Tools/openfast/install/bin/openfast'   # Path to executable
    linear.FAST_InputFile           = 'IEA-15-240-RWT-UMaineSemi.fst'   # FAST input file (ext=.fst)
    linear.FAST_directory           = '/Users/dzalkind/Tools/IEA-15-240-RWT/OpenFAST/IEA-15-240-RWT-UMaineSemiTrim'   # Path to fst directory files
    linear.FAST_steadyDirectory     = '/Users/dzalkind/Tools/SaveData/UMaine/Steady'
    linear.FAST_linearDirectory     = '/Users/dzalkind/Tools/SaveData/TrimTest/LinearTwrPit_Tol1en5'
    linear.debug_level              = 2
    linear.dev_branch               = True
    linear.write_yaml               = True

    # do a read to get gearbox ratio
    fastRead = InputReader_OpenFAST(FAST_ver='OpenFAST', dev_branch=True)
    fastRead.FAST_InputFile = linear.FAST_InputFile   # FAST input file (ext=.fst)
    fastRead.FAST_directory = linear.FAST_directory   # Path to fst directory files

    fastRead.execute()

    # linearization setup
    linear.v_rated          = 10.74         # needed as input from RotorSE or something, to determine TrimCase for linearization
    linear.GBRatio          = fastRead.fst_vt['ElastoDyn']['GBRatio']
    linear.WindSpeeds       = np.arange(5,25,1,dtype=float).tolist()   #[8.,10.,12.,14.,24.]
    linear.DOFs             = ['GenDOF','TwFADOF1','PtfmPDOF']
    linear.TMax             = 2000.
    linear.NLinTimes        = 12

    #if true, there will be a lot of hydronamic states, equal to num. states in ss_exct and ss_radiation models
    linear.HydroStates      = True  

    # simulation setup
    linear.parallel         = True


    # run steady state sims
    # linear.runFAST_steady()

    # process results 
    # linear.postFAST_steady()

    # run linearizations
    linear.runFAST_linear()
