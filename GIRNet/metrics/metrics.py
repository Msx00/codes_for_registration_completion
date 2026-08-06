import torch
import math
import numpy as np


def MAE( prediction, target ):
    total_loss = torch.mean( torch.abs( prediction-target ) )
    return total_loss

def MeanMagnitude( displacement, batch_reduce = True ):
    """ Given a displacement field, calculate the average magnitude """
    magnitude = torch.linalg.vector_norm( displacement, dim = 1 )
    if batch_reduce:
        return magnitude.mean()
    else:
        return magnitude.mean( dim=1 )  # Only average over points, not batch

def MeanDisplacementError( prediction, target, batch_reduce = True ):
    return MeanMagnitude( target - prediction, batch_reduce=batch_reduce )

def MSE( prediction, target ):
    total_loss = ((prediction - target)**2).mean()
    return total_loss

def RMSE( prediction, target ):

    total_loss = torch.sqrt( MSE( prediction, target ) )
    return total_loss

def RMSE_np( prediction, target):
    """
        prediction: numpy array of shape (N, 3)
        target: numpy array of shape (N, 3)
        return:
            RMSE: float, root mean square error between prediction and target
            sqrt( (1/N) * sum_i ||gt[i] - pred[i]||^2 ).
    """

    rmse = np.sqrt( np.mean( np.sum( (target - prediction)**2, axis=1 ) ) )
    return rmse



def chamfer( prediction, target, mode="SYMMETRIC", reduction="mean" ):
    ''' Computes the distance from the source (prediction) to the target points.

    By default (mode == "SYMETRIC"), this computes the symmetric distance, i.e.
        chamfer = (dist from pred. to target) + (dist from target to pred).
    If mode is set to "P2T", it will only calculate the asymmetric distance from prediction to target.
    If mode is set to "T2P", it will only calculate the asymmetric distance from target to prediction.

    Champfer distances > 0 suggest there is noise and/or registration errors.

    Args:
        prediction: Tensor of shape (B, N1, 3)
        target: Tensor of shape (B, N2, 3)
        mode: string, one of "SYMMETRIC", "P2T" or "T2P", determines calculation of symmetric or 
            asymmetric Chamfer distance.
        reduction: string, "mean" or "sum"
    Returns:
        sum of chamfer distances (if aggregate == "sum") or mean of chamfer distances (if aggregate == "mean")
    '''

    assert reduction == "mean" or reduction == "sum"
    assert mode == "SYMMETRIC" or mode == "P2T" or mode == "T2P"

    # dists = torch.cdist( prediction.permute(0,2,1), target.permute(0,2,1) ) # B x N1 x N2
    dists = torch.cdist(prediction, target)
    # print("dists.shape:", dists.shape)
    if mode == "SYMMETRIC":
        dists_pred_to_target, _ = torch.min( dists, dim=2 )
        dists_target_to_pred, _ = torch.min( dists, dim=1 )

        summed_dists = dists_pred_to_target.sum( dim=1 ) + dists_target_to_pred.sum( dim=1 )
    elif mode == "P2T":
        dists_pred_to_target, _ = torch.min( dists, dim=2 ) # B x N1
        # print("dists_pred_to_target.shape:", dists_pred_to_target.shape)
        summed_dists = dists_pred_to_target.sum( dim=1 )
    elif mode == "T2P":
        dists_target_to_pred, _ = torch.min( dists, dim=1 )
        summed_dists = dists_target_to_pred.sum( dim=1 )
    # print(summed_dists)
    # print("summed_dists.shape:", summed_dists.shape)
    if reduction == "mean":
        return summed_dists.mean()
    elif reduction == "sum":
        return summed_dists.sum()
        
    # dists_min should hold:
    # - for every preop point the minimum distance to the intraop points
    # - for every intraop point the minimum distance to the preop points



# def compute_landmark_eror_unpaired( source_points, target_points ):
#     # Given two VTK point clouds, calculate the TRE between them. Assumes that the landmarks do not correspond.

#     assert source_points.GetNumberOfPoints() == target_points.GetNumberOfPoints()
#     assert source_points.GetNumberOfPoints() > 0

#     dists = []
#     for i in range(source_points.GetNumberOfPoints()):
#         s = source_points.GetPoint(i)
#         t = target_points.GetPoint(i)

#         dist = math.sqrt( (s[0]-t[0])**2 +
#             (s[1]-t[1])**2 +
#             (s[2]-t[2])**2 )

#         dists.append( dist )

#     avg_dist = sum(dists)/len(dists)
#     return avg_dist, dists

def smoothness(prediction, target):
    """
    Calculate smoothness of the predicted displacement field.
    """
    ...
    

def jacobian_determinant(pcd, displ,
                k = 20  # 邻居数量，可根据需要调整
):
    
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    X = pcd  # 原始点云坐标
    U = displ  # 每个点对应的位移向量
    N = X.shape[0]  # 点的数量

    # 示例输入：N个空间点坐标和对应的形变位移
    # X: 原始点云坐标, shape (N, 3)
    # U: 每个点对应的位移向量, shape (N, 3)
    # N = 1000
    # # 随机生成示例数据
    # np.random.seed(4200)
    # X = np.random.rand(N, 3)
    # U = 0.1 * np.random.randn(N, 3)

    # 构建邻域搜索器

    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    # 计算每个点的局部梯度 (Jacobian of deformation field)
    jac_det = np.zeros(N)
    I = np.eye(3)

    for i in range(N):
        neighbor_idx = indices[i, 1:]  # 排除自身
        Xi = X[i]
        Ui = U[i]
        
        # 构建差分矩阵：ΔX, ΔU
        dX = X[neighbor_idx] - Xi  # shape (k, 3)
        dU = U[neighbor_idx] - Ui  # shape (k, 3)
        
        # 最小二乘估计梯度矩阵 G 使得 dU ≈ G @ dX
        # 解 G (3x3) from dX (k x 3) and dU (k x 3)
        G, _, _, _ = np.linalg.lstsq(dX, dU, rcond=None)  # G shape (3,3) after transpose
        G = G.T  # 转置回来，使 G @ ΔX.T ≈ ΔU.T
        
        # 完整变形场Jacobian：J = I + G
        J = I + G
        jac_det[i] = np.linalg.det(J)

    # 计算负Jacobian行列式的百分比
    perc_negative = np.mean(jac_det < 0) * 100
    print(f"Negative Jacobian determinants: {np.sum(jac_det < 0)} / {N} ({perc_negative:.2f}%)")

    return jac_det, perc_negative





if __name__ == "__main__":

    # pred = torch.rand((2,3,4))

    # print("prediction:", pred)

    # target = (pred + torch.rand_like(pred)).detach()

    # print("target:", target)

    # print("MAE:", MAE( pred, target ) )
    # print("MSE:", MSE( pred, target ) )
    # print("RMSE:", RMSE( pred, target ) )
    # print("Chamfer symmetrical:", chamfer( pred, target ) )
    # print("Chamfer P2T:", chamfer( pred, target, mode="P2T" ) )
    # print("Chamfer T2P:", chamfer( pred, target, mode="T2P" ) )

    source = np.random.rand(1000, 3)
    displ = np.random.rand(1000, 3) * 0.1
    target = source + displ

    jacobian_determinant(pcd=source, displacement=displ)



