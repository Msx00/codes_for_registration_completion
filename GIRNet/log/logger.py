import os
import datetime
import math
import torch
import shutil
import yaml

from torch.utils.tensorboard import SummaryWriter

class Logger:

    def __init__( self, base_path, n_samples, comment=None ):
        """Create folder for logging the training process, copy all training scripts to the training folder.
        Args:
            base_path (str): folder where to store all the logs
            n_samples (num): number of samples
            comment (str, optional): description of the training, will be appended to the end of the training folder

        """
        timestamp = datetime.datetime.now().strftime( "%Y-%m-%d_%H-%M-%S" )
        n_samples_formatted = "+".join(str(x) for x in n_samples) if isinstance(n_samples, list) else str(n_samples)
        if comment:
            comment = comment.replace(" ", "_")
            folder_name = f"{timestamp}_N{n_samples_formatted}_{comment}"
        else:
            folder_name = f"{timestamp}_N{n_samples_formatted}"

        self.path = os.path.join( base_path, folder_name )
        assert (not os.path.exists( self.path )), f"Cannot log to {self.path}, directory exists!"

        os.makedirs( self.path )
        print("Created folder:", self.path)

        ######################################
        ## Copy all the python files into the output directory:
        def ignore(path, names):
            ignored = set()
            for name in names:
                if not name in ["data", "log", "metrics", "models", "configs", "visualization"] and not (name.endswith(".py") or name.endswith(".yaml")):
                    ignored.add( name )
                if name[0] == ".":
                    ignored.add( name )
                if not os.path.isdir( os.path.join( path, name ) ):
                    if not name.endswith(".py"):
                        ignored.add( name )

            return ignored
        src_bkup_path = os.path.join( self.path, "src" )
        shutil.copytree(os.getcwd(), src_bkup_path, ignore = ignore )
        Logger.remove_empty_folders( src_bkup_path )

        # Per default, just log into this path. If separate trials are started, will log into
        # sub directories:
        self.current_path = self.path
        self.summary_writer = SummaryWriter( log_dir = self.current_path )
        
        self.trial = -1
        self.trial_params = None
        self.args = None

        #run_name = f"trial{trial.number}"
        #writer = SummaryWriter( log_dir=os.path.join(log_dir, run_name) )

        #checkpoints_dir = exp_dir.joinpath(run_name, 'checkpoints/')
        #checkpoints_dir.mkdir(exist_ok=True, parents=True)

        # Add git commit string to the arguments:
        #git_commit_sha = repo.head.object.hexsha
        #run_details["git_commit"] = git_commit_sha
        #print("current sha:", git_commit_sha)

    def save_run_details( self, args, ):
        """save the command line arguments and some other related information
        to a yaml file in the output directory

        Args:
            args (dict): the command line arguments
        """
        args = vars(args)

        import socket
        hostname = socket.gethostname()
        # append to the args:
        args["hostname"] = hostname

        # get GPU type:
        gpu_type = torch.cuda.get_device_name(0)
        args["gpu"] = gpu_type

        args_file = os.path.join( self.path, "command_line_args.yaml" )
        with open( args_file, 'w' ) as f:
            yaml.dump( args, f )
        
        self.args = args


    def start_new_trial( self ):
        self.trial += 1

        self.current_path = os.path.join( self.path, f"{self.trial}" )

        os.makedirs( self.current_path )
        print("Starting trial:", self.trial)
        print("\tLogging trial data to:", self.current_path)
    
        self.summary_writer = SummaryWriter( log_dir = self.current_path )

        self.trial_params = None


    def continue_old_trial(self, old_trial_folder):
        # copy old trial folder to the current path
        # only copy the trial folder, not the parent folder
        # set the log_dir as the copied folder
        if old_trial_folder[-1] == "/":
            old_trial_folder = old_trial_folder[:-1]
        dst_folder = os.path.join(self.path, os.path.basename(old_trial_folder))
        # print(old_trial_folder )
        self.trial = int(os.path.basename(old_trial_folder))
        print("Copying old weights to:", dst_folder)
        shutil.copytree(old_trial_folder, dst_folder)
        print("Copied old trial data to:", dst_folder)
        self.current_path = dst_folder
        self.summary_writer = SummaryWriter( log_dir = self.current_path )
        self.trial_params = self.get_trial_params()


    def get_commandline_args( self ):
        if self.args is None:
            with open( os.path.join(self.path, "command_line_args.yaml"), 'r') as f:
                args = yaml.load( f, Loader=yaml.FullLoader )
            self.args = args
        return self.args
        

    def get_trial_params( self ):
        with open( os.path.join(self.current_path, "params.yaml"), 'r') as f:
            trial_params = yaml.load( f, Loader=yaml.FullLoader )

        return trial_params


    def set_trial_params( self, trial_params, model_name ):
        trial_params["model_name"] = model_name
        with open( os.path.join(self.current_path, "params.yaml"), 'w') as f:
            yaml.dump( trial_params, f )

        self.trial_params = trial_params


    def finalize_trial( self, metrics = {} ):
        assert self.trial_params, \
                "Before calling 'finalize_trial', you must set the paramters via 'set_trial_params'!"

        self.summary_writer.add_hparams( self.trial_params, metrics )


    def save_model( self, name, epoch, model, optimizer, scheduler, train_mean_displ_err = math.inf, test_mean_displ_err = math.inf ):
        savepath = os.path.join( self.current_path, f"{name}.pth" )
        
        print(f"Saving at {savepath}")
        state = {
            'epoch': epoch,
            'train_mean_displ_err': train_mean_displ_err,
            'test_mean_displ_err': test_mean_displ_err,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            # "logger": logger,
            #'hyperparams': trial.params
        }
        torch.save(state, savepath)


    def save_stats(self, stats):
        with open( os.path.join(self.current_path, "stats.yaml"), 'w') as f:
            yaml.dump( stats, f)


    @staticmethod
    def load_model( model, path ):

        state = torch.load(path)
        model.load_state_dict( state["model_state_dict"] )
        return model, state


    @staticmethod
    def remove_empty_folders( path ):
        for (_path, _dirs, _files) in os.walk(path, topdown=False):
            if len(_files) > 0:
                continue
            try:
                os.rmdir(_path)
            except OSError as ex:
                print("Error, could not remove empty directory:", ex)


if __name__ == "__main__":

    logger = Logger( "/tmp/logger_test", 10, comment="logger_module_test" )

    def optimize():
        logger.start_new_trial()
        print("Simulating trial")

        import random
        for e in range(10):
            print("Epoch:", e)
            logger.summary_writer.add_scalar("AvgErr/train", random.random(), e)
            logger.summary_writer.add_scalar("AvgDisplacementErr/train", random.random(), e)
            logger.summary_writer.add_scalar("AvgErr/test", random.random(), e)
            logger.summary_writer.add_scalar("AvgDisplacementErr/test", random.random(), e)
        
        logger.finalize_trial( {}, {} )

    for i in range(3):
        optimize()



