import torch
from matplotlib import pyplot as plt



def pos_enc_sinusoidal(points, frequency=2 ** -1):
    """
    Compute sinusoidal positional encoding of the input coordinates.

    Args:
        points: coordiantes, shape (B, 3, N)
        frequency: angualr frequency

    Returns:
        pos_enc: sinusoidla positional encoding in shape of (B, 6, N)
    """
    B = points.shape[0]
    x = points[:, 0, ...].reshape(B, 1, -1)
    y = points[:, 1, ...].reshape(B, 1, -1)
    z = points[:, 2, ...].reshape(B, 1, -1)
    sinx = torch.sin( frequency * x)
    siny = torch.sin( frequency * y)
    sinz = torch.sin( frequency * z)
    cosx = torch.cos( frequency * x)
    cosy = torch.cos( frequency * y)
    cosz = torch.cos( frequency * z)
    pos_enc = torch.cat([sinx, cosx, siny, cosy, sinz, cosz], dim=1)
    return pos_enc



def visualize(pos_enc, freq_list=[], batch=0, output_path=None):
    pos_enc_show = pos_enc[batch, :,  0:256].transpose(1, 0)
    fig, ax = plt.subplots(figsize=(20,20))
    cax = ax.matshow(pos_enc_show,)
    # print(freq_list, len(freq_list), pos_enc_show.shape[1])
    if freq_list and len(freq_list) * 6 == pos_enc_show.shape[1]:
        ax.set_xticklabels(freq_list)
    plt.gcf().colorbar(cax)

    plt.show()
    if output_path:
        plt.savefig(output_path)
        print("saved plot to:", output_path)




if __name__ == "__main__":
    points = torch.rand([5, 3, 1000])
    print(points.shape)
    # positional encodings with default frequences 2^-1
    pos_enc = pos_enc_sinusoidal(points=points)
    print(pos_enc.shape)

    # positional encodings with various frequences
    pos_enc_list = []
    freq_list = []
    # for idx_freq in range(-6, 0):
    for idx_freq in range(-5, 8):
        freq = 2 ** idx_freq
        freq_list.append(freq)
        pos_enc = pos_enc_sinusoidal(points=points, frequency=freq)
        pos_enc_list.append(pos_enc)
    pos_enc = torch.cat(pos_enc_list, dim=1)
    print(pos_enc.shape)

    visualize(pos_enc, freq_list)
