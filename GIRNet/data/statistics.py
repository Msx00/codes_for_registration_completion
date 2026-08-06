import os
import matplotlib.pyplot as plt
#from data_hhlbm import HHLBMDataset
from .pc_utils import preprocess_data

class Statistics():

    def __init__( self, dataset, output_folder, max_num_samples = 100 ):

        if not os.path.exists( output_folder ):
            os.makedirs( output_folder )

        nearest_neighbor_distances = []
        #preop_num_valid_surface_points = []
        #preop_num_valid_internal_points = []

        preop_num_valid_points = []
        preop_dimensions_x = []
        preop_dimensions_y = []
        preop_dimensions_z = []
        preop_mean_x = []
        preop_mean_y = []
        preop_mean_z = []

        intraop_num_valid_points = []
        intraop_dimensions_x = []
        intraop_dimensions_y = []
        intraop_dimensions_z = []
        intraop_mean_x = []
        intraop_mean_y = []
        intraop_mean_z = []


        preop_features = []
        intraop_features = []

        num = min( max_num_samples, len(dataset))
        for i in range(num):
            print(f"Loading sample {i}/{num}")

            data = dataset[i]

            if isinstance( data, tuple ):
                preop_volume, preop_surface, preop_landmarks, _, \
                        intraop_volume, intraop_surface, intraop_landmarks, _ = data
                preop = preop_volume.cuda().unsqueeze(0)
                intraop = intraop_surface.cuda().unsqueeze(0)
                displ = (intraop_landmarks.cuda() - preop_landmarks.cuda()).unsqueeze(0)
            else:
                preop = data["preop"].cuda()
                intraop = data["intraop"].cuda()
                #preop_internal = data["preop_internal"].cuda()
                #intraop_internal = data["intraop_internal"].cuda()
                displ = data["displ"].cuda()

                preop = preop.unsqueeze(0)
                intraop = intraop.unsqueeze(0)
                displ = displ.unsqueeze(0)
        
            coords_preop, features_preop, coords_intraop, features_intraop = preprocess_data(
                    preop, intraop 
            )

            # From here on, remove the batch shape, because we have Batch size 1:
            assert coords_preop.shape[0] == 1
            coords_preop = coords_preop[0,...]
            features_preop = features_preop[0,...]
            coords_intraop = coords_intraop[0,...]
            features_intraop = features_intraop[0,...]


            ###########
            ## Preop
            
            preop_features.append( features_preop )

            preop_valid_mask = (coords_preop[0,:].abs() < 5000)
            preop_valid_points = coords_preop[:,preop_valid_mask]
            preop_n_valid_points = preop_valid_mask.sum().item()
            preop_num_valid_points.append( preop_n_valid_points )

            preop_n_dummy_points = coords_preop.shape[1] - preop_n_valid_points
            if preop_n_dummy_points > 0:
                print(f"Detected {preop_n_dummy_points} preop dummy points. Removing.")

            dimensions = preop_valid_points.max( dim = 1 )[0] - preop_valid_points.min( dim = 1 )[0]
            mean = preop_valid_points.mean( dim = 0 )
            preop_dimensions_x.append( dimensions[0].item() )
            preop_dimensions_y.append( dimensions[1].item() )
            preop_dimensions_z.append( dimensions[2].item() )
            preop_mean_x.append( mean[0].item() )
            preop_mean_y.append( mean[1].item() )
            preop_mean_z.append( mean[2].item() )


            #########
            ## Intraop
            intraop_features.append( features_intraop )

            intraop_valid_mask = (coords_intraop[0,:].abs() < 5000)
            intraop_valid_points = coords_intraop[:,intraop_valid_mask]
            intraop_n_valid_points = intraop_valid_mask.sum().item()
            intraop_num_valid_points.append( intraop_n_valid_points )

            intraop_n_dummy_points = coords_intraop.shape[1] - intraop_n_valid_points
            if intraop_n_dummy_points > 0:
                print(f"Detected {intraop_n_dummy_points} intraop dummy points. Removing.")

            dimensions = intraop_valid_points.max( dim = 1 )[0] - intraop_valid_points.min( dim = 1 )[0]
            mean = intraop_valid_points.mean( dim = 0 )
            intraop_dimensions_x.append( dimensions[0].item() )
            intraop_dimensions_y.append( dimensions[1].item() )
            intraop_dimensions_z.append( dimensions[2].item() )
            intraop_mean_x.append( mean[0].item() )
            intraop_mean_y.append( mean[1].item() )
            intraop_mean_z.append( mean[2].item() )


        print("Saving statistics plots in: \n\t", output_folder)

        fig, axs = plt.subplots(3, 1, tight_layout=True)
        axs[0].hist( preop_dimensions_x, bins=20 )
        axs[0].set_title( "dimensions X" )
        axs[1].hist( preop_dimensions_y, bins=20 )
        axs[1].set_title( "dimensions Y" )
        axs[2].hist( preop_dimensions_z, bins=20 )
        axs[2].set_title( "dimensions Z" )
        fig.savefig( os.path.join( output_folder, "preop_dimensions.eps" ) )

        fig, axs = plt.subplots(3, 1, tight_layout=True)
        axs[0].hist( preop_mean_x, bins=20 )
        axs[0].set_title( "mean_x X" )
        axs[1].hist( preop_mean_y, bins=20 )
        axs[1].set_title( "mean Y" )
        axs[2].hist( preop_mean_z, bins=20 )
        axs[2].set_title( "mean Z" )
        fig.savefig( os.path.join( output_folder, "preop_mean.eps" ) )

        fig = plt.figure()
        plt.hist( preop_num_valid_points, bins=20 )
        plt.title( "valid points" )
        plt.savefig( os.path.join( output_folder, "preop_num_valid_points.eps" ) )

        ##########################
        fig, axs = plt.subplots(3, 1, tight_layout=True)
        axs[0].hist( intraop_dimensions_x, bins=20 )
        axs[0].set_title( "dimensions X" )
        axs[1].hist( intraop_dimensions_y, bins=20 )
        axs[1].set_title( "dimensions Y" )
        axs[2].hist( intraop_dimensions_z, bins=20 )
        axs[2].set_title( "dimensions Z" )
        fig.savefig( os.path.join( output_folder, "intraop_dimensions.eps" ) )

        fig, axs = plt.subplots(3, 1, tight_layout=True)
        axs[0].hist( intraop_mean_x, bins=20 )
        axs[0].set_title( "mean_x X" )
        axs[1].hist( intraop_mean_y, bins=20 )
        axs[1].set_title( "mean Y" )
        axs[2].hist( intraop_mean_z, bins=20 )
        axs[2].set_title( "mean Z" )
        fig.savefig( os.path.join( output_folder, "intraop_mean.eps" ) )

        fig = plt.figure()
        plt.hist( intraop_num_valid_points, bins=20 )
        plt.title( "valid points" )
        plt.savefig( os.path.join( output_folder, "intraop_num_valid_points.eps" ) )

 
    #/mnt/ceph/tco/TCO-Staff/Homes/pfeiffemi/Deformation/V2SData/NewPipeline/us_tumors_only_static
