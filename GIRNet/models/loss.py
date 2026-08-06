from models.select import select_points
import torch

def subsampled_prediction_loss_3_matching_cues(displ, predictions, cue_mask, weight_cue=1):
    """ Loss for both downsampled preop and intraop points including MSE for all levels of displacement
        and loss item for matching cues 

    """

    displ_current_level = displ

    full_loss = 0
   
    # Traverse in order from high res (original) point cloud to lower res (subsampled) levels:
    predictions = [p for p in reversed(predictions)]
    # Weigh depending on number of levels. This makes losses between runs slightly more comparable.
    # Might not be optimal, though.
    num_predictions = len(predictions)
    weights = [1/num_predictions for p in predictions]
    # Weigh the main, full-scale displacement field with a higher factor:
    weights[0] = 10

    for level, level_result in enumerate( predictions ):
        prediction = level_result["result"]
        #if level == 0:
        #    prediction = torch.cat( (prediction, displ_intraop_dummy), dim=2 )
        #loss = (((prediction - displ_current_level)*preop_point_mask)**2).mean()        # MSE
        loss = (((prediction - displ_current_level))**2).mean()        # MSE
        full_loss += weights[level]*loss

        if level < len(predictions)-1:
            next_level_idx = predictions[level+1]["point_preop_idx"]
            displ_current_level = select_points( points = displ_current_level, idx = next_level_idx )
            #preop_point_mask = select_points( points = preop_point_mask, idx = next_level_idx )

    # loss for matching cues, for now only a single pair of cues is supported
    # cue loss is calculated only using the points in the highest resolution
    # TODO 1. add support for multiple pairs of cues
    # TODO 2. add support for multiple resolutions

    pred_cue = predictions[0]["result"][:, :, cue_mask]
    gt_cue = displ[:, :, cue_mask]
    loss_cue = (((pred_cue - gt_cue))**2).mean()        # MSE
    
    full_loss += weight_cue * loss_cue

    return full_loss


def calc_loss_pinns(results, youngs_modulus, poisson_ratio):
    ### Calculate loss for PINNs, using Youngs modulus and Poisson ratio to calculate Lame parameters,
    # which is used to build constitutive model (matrix) for the material. The the constitutive matrix
    # is used to calculate the stress tensor from the strain tensor. The calculated stress tensor is
    # compared to the stress tensor predicted by the neural network.

    stress_xx_pred = results["PINN"]["stress"]["xx"]
    stress_yy_pred = results["PINN"]["stress"]["yy"]
    stress_zz_pred = results["PINN"]["stress"]["zz"]
    stress_xy_pred = results["PINN"]["stress"]["xy"]
    stress_xz_pred = results["PINN"]["stress"]["xz"]
    stress_yz_pred = results["PINN"]["stress"]["yz"]

    momentum_balance1 = results["PINN"]["momentum"]["balance1"]
    momentum_balance2 = results["PINN"]["momentum"]["balance2"]
    momentum_balance3 = results["PINN"]["momentum"]["balance3"]

    strain_xx = results["PINN"]["strain"]["xx"]
    strain_yy = results["PINN"]["strain"]["yy"]
    strain_zz = results["PINN"]["strain"]["zz"]
    strain_xy = results["PINN"]["strain"]["xy"]
    strain_xz = results["PINN"]["strain"]["xz"]
    strain_yz = results["PINN"]["strain"]["yz"]


    mu = youngs_modulus / (2 * (1 + poisson_ratio))
    lam = youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))

    B = mu.shape[0]
    B_N = strain_xx.shape[0]

    mu = mu.repeat(1, int(B_N / B) ).reshape(-1, 1)
    lam = lam.repeat(1, int(B_N / B) ).reshape(-1, 1)

    # print("mu.shape: ", mu.shape, "lam.shape: ", lam.shape)
    # print("strain_xx.shape: ", strain_xx.shape)

    # mu = torch.ones_like(strain_xx) * mu
    # lam = torch.ones_like(strain_xx) * lam


    stress_xx = lam * (strain_xx + strain_yy + strain_zz) + 2 * mu * strain_xx
    stress_yy = lam * (strain_xx + strain_yy + strain_zz) + 2 * mu * strain_yy
    stress_zz = lam * (strain_xx + strain_yy + strain_zz) + 2 * mu * strain_zz
    stress_xy = 2 * mu * strain_xy
    stress_xz = 2 * mu * strain_xz
    stress_yz = 2 * mu * strain_yz
    

    # print("stress_xx_pred.shape: ", stress_xx_pred.shape, "stress_xx.shape: ", stress_xx.shape, "strain_xx.shape: ", strain_xx.shape)

    # 1. Static Equilibrium Equations
    loss_ses = torch.norm(momentum_balance1, dim=0, p=1) + torch.norm(momentum_balance2, dim=0, p=1) + torch.norm(momentum_balance3,  dim=0, p=1)
    # loss_ses = torch.mean(momentum_balance1**2) + torch.mean(momentum_balance2**2) + torch.mean(momentum_balance3**2)

    # 2. Stress-Strain Relationship

    loss_ss = torch.norm(stress_xx_pred - stress_xx, dim=0, p=1) +\
            torch.norm(stress_yy_pred - stress_yy, dim=0, p=1) +\
            torch.norm(stress_zz_pred - stress_zz, dim=0, p=1) +\
            torch.norm(stress_xy_pred - stress_xy, dim=0, p=1) + \
            torch.norm(stress_xz_pred - stress_xz, dim=0, p=1) +\
            torch.norm(stress_yz_pred - stress_yz, dim=0, p=1)
    # print("stress_xx_pred: ", stress_xx_pred.shape, "torch.mean(stress_xx_pred): ", torch.mean(stress_xx_pred))
    # print("stress_xx: ", stress_xx.shape, "torch.mean(stress_xx): ", torch.mean(stress_xx))
    # loss_ss = torch.mean((stress_xx_pred - stress_xx)**2) + torch.mean((stress_yy_pred - stress_yy)**2) + torch.mean((stress_zz_pred - stress_zz)**2) + torch.mean((stress_xy_pred - stress_xy)**2) + torch.mean((stress_xz_pred - stress_xz)**2) + torch.mean((stress_yz_pred - stress_yz)**2)

    # 3. Elastic energy:
    loss_es = torch.mean(
        0.5 * (
            strain_xx * stress_xx_pred + 
            strain_yy * stress_yy_pred + 
            strain_zz * stress_zz_pred + 
            2 * strain_xy * stress_xy_pred + 
            2 * strain_xz * stress_xz_pred + 
            2 * strain_yz * stress_yz_pred
        ))

    # print("torch.mean(momentum_balance1): ", torch.mean(momentum_balance1))
    # print("torch.mean(momentum_balance2): ", torch.mean(momentum_balance2))
    # print("torch.mean(momentum_balance3): ", torch.mean(momentum_balance3))

    # print("torch.mean(stress_xx_pred)", torch.mean(stress_xx_pred))
    # print("torch.mean(stress_yy_pred)", torch.mean(stress_yy_pred))
    # print("torch.mean(stress_zz_pred)", torch.mean(stress_zz_pred))
    # print("torch.mean(stress_xy_pred)", torch.mean(stress_xy_pred))
    # print("torch.mean(stress_xz_pred)", torch.mean(stress_xz_pred))
    # print("torch.mean(stress_yz_pred)", torch.mean(stress_yz_pred))

    # print("torch.mean(stress_xx)", torch.mean(stress_xx))
    # print("torch.mean(stress_yy)", torch.mean(stress_yy))
    # print("torch.mean(stress_zz)", torch.mean(stress_zz))
    # print("torch.mean(stress_xy)", torch.mean(stress_xy))
    # print("torch.mean(stress_xz)", torch.mean(stress_xz))
    # print("torch.mean(stress_yz)", torch.mean(stress_yz))

    # print("torch.mean(strain_xx)", torch.mean(strain_xx))
    # print("torch.mean(strain_yy)", torch.mean(strain_yy))
    # print("torch.mean(strain_zz)", torch.mean(strain_zz))
    # print("torch.mean(strain_xy)", torch.mean(strain_xy))
    # print("torch.mean(strain_xz)", torch.mean(strain_xz))
    # print("torch.mean(strain_yz)", torch.mean(strain_yz))

    # print("loss_ses: ", loss_ses.shape)
    # print("loss_ss: ", loss_ss.shape)
    # print("loss_es: ", loss_es.shape)

    loss_pinn = loss_ses + 1e-6 * loss_ss + loss_es

    # print("loss_pinn: ", loss_pinn, "loss_ses: ", loss_ses, "loss_ss: ", loss_ss, "loss_es: ", loss_es)

    return loss_pinn


