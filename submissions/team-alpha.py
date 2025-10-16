## Sample submission file for the VAE + Flow leaderboard challenge     
## Author: Scott H. Hawley, Oct 6 2025 
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
import gdown
import os
import numpy as np


class SimpleVAEModel(nn.Module):
    def __init__(self,
                 latent_dim=3,    # dimensionality of the latent space. bigger=less compression, better reconstruction
                 n_hid=[256,64],  # simple
                 act = nn.LeakyReLU,
                 ):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(28 * 28, n_hid[0]),act(),
                nn.Linear(n_hid[0], n_hid[1]), act(),
                nn.Linear(n_hid[1], latent_dim*2), # *2 b/c mu, log_var
                )
        self.decoder = nn.Sequential(
                nn.Linear(latent_dim, n_hid[1]), act(),
                nn.Linear(n_hid[1], n_hid[0]),  act(),
                nn.Linear(n_hid[0], 28 * 28),
                )
        self.latent_dim, self.n_hid, self.act = latent_dim, n_hid, act # save for possible use later

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten to (batch, 784)
        z = self.encoder(x)
        i_half = z.shape[-1]//2
        mu, log_var = z[:,:i_half],  z[:,i_half:]
        z_hat = mu + torch.randn_like(mu) * torch.exp(0.5*log_var)
        x_hat = self.decoder(z_hat)
        x_hat = x_hat.view(x_hat.size(0), 1, 28, 28)  # Reshape back for BCE loss
        return z, x_hat, mu, log_var, z_hat
    

class SimpleFlowModel(nn.Module):
    def __init__(self, latent_dim=3, n_hidden=32, n_layers=3, act=nn.LeakyReLU):
        super(SimpleFlowModel, self).__init__()
        self.latent_dim = latent_dim
        self.layers = nn.Sequential(
            nn.Linear(latent_dim+1, n_hidden), act(),
            *[nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) for _ in range(n_layers-1)],
            nn.Linear(n_hidden, latent_dim),)
    
    def forward(self, x, t, act=F.gelu):
        t = t.expand(x.size(0), 1)  # Ensure t has the correct dimensions
        x = torch.cat([x, t], dim=1)
        return self.layers(x)




### these next two are identical to blog post
@torch.no_grad()
def fwd_euler_step(model, current_points, current_t, dt):
    velocity = model(current_points, current_t)
    return current_points + velocity * dt 

@torch.no_grad()
def integrate_path(model, initial_points, step_fn=fwd_euler_step, n_steps=100,
                   save_trajectories=False, warp_fn=None):
    """this 'sampling' routine is primarily used for visualization."""
    device = next(model.parameters()).device
    current_points = initial_points.clone()
    ts =  torch.linspace(0,1,n_steps).to(device)
    if warp_fn: ts = warp_fn(ts)
    if save_trajectories: trajectories = [current_points]    
    for i in range(len(ts)-1):
        current_points = step_fn(model, current_points, ts[i], ts[i+1]-ts[i])
        if save_trajectories: trajectories.append(current_points)
    if save_trajectories: return current_points, torch.stack(trajectories).cpu()
    return current_points 
#####




class SubmissionInterface(nn.Module):
    """All teams must implement this for automated evaluation.
    When you subclass/implement these methods, replace the NotImplementedError."""
    
    def __init__(self):
        super().__init__()

        #--- REQUIRED INFO:
        self.info = { 
            'team': 'sample',  # REPLACE with your team name. This will be public
            'names': 'Your Name(s) Here', # or single name. This will be kept private
        }
        self.latent_dim = 3   # TODO: we could just (re)measure this on the fly 
        #---- 

        # keep support for full auto-initialization:
        self.load_vae()
        self.load_flow_model()
        self.device = 'cpu' # we can change this later via .to()
    
    def load_vae(self):
        """this completely specifies the vae model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.vae = SimpleVAEModel(latent_dim=self.latent_dim)
        vae_weights_file = 'downloaded_vae.safetensors'
        if not os.path.exists(vae_weights_file):
            safetensors_link = "https://drive.google.com/file/d/1N4VS3HKBrXnuQhiud1ruMTFZ5jbhsrIn/view?usp=sharing"
            gdown.download(safetensors_link, vae_weights_file, quiet=False, fuzzy=True)
        self.vae.load_state_dict(load_file(vae_weights_file))
        
    def load_flow_model(self):
        """this completely specifies the flow model including configuration parameters,
           downloads/mounts the weights from Google Drive, automatically loads weights"""
        self.flow_model = SimpleFlowModel(latent_dim=self.latent_dim)
        flow_weights_file = 'downloaded_flow.safetensors'
        if not os.path.exists(flow_weights_file):
            safetensors_link = "https://drive.google.com/file/d/13hlolKEc1QB6wA5M_sSgH8wfKR35fjc9/view?usp=sharing"
            gdown.download(safetensors_link, flow_weights_file, quiet=False, fuzzy=True)
        self.flow_model.load_state_dict(load_file(flow_weights_file))
    
    def generate_samples(self, n_samples:int, n_steps=100) -> torch.Tensor:
        z0 = torch.randn([n_samples, self.latent_dim]).to(self.device)
        z1 = integrate_path(self.flow_model, z0, n_steps=n_steps)
        gen_xhat = F.sigmoid(self.decode(z1).view(-1, 28, 28))
        return gen_xhat

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        # if your vae has linear layers, flatten first
        # if your vae has conv layers, comment out next line
        images = images.view(images.size(0), -1)  
        with torch.no_grad():
            z = self.vae.encoder(images.to(self.device))
            mu = z[:, :self.latent_dim]  # return only first half (mu)
            return mu
    
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decoder(latents)

    def to(self, device):
        self.device = device 
        self.vae.to(self.device)
        self.flow_model.to(self.device)
        return self 
    

@torch.no_grad()
def generate_animation_data(model, n_samples=500, n_steps=50, classifier=None):
    """Generate trajectories and metadata for 3D animation.
    Returns dict with trajectories, colors, and alphas."""
    z0 = torch.randn([n_samples, model.latent_dim]).to(model.device)
    z_final, z_history = integrate_path(
        model.flow_model, z0, n_steps=n_steps, save_trajectories=True
    )
    
    # Decode and classify final positions
    final_images = F.sigmoid(model.decode(z_final).view(-1, 28, 28))
    
    if classifier is not None:
        logits = classifier(final_images.unsqueeze(1))
        probs = F.softmax(logits, dim=1)
        predictions = probs.argmax(dim=1)
        confidences = probs.max(dim=1)[0]
    else:
        predictions = torch.zeros(n_samples)
        confidences = torch.ones(n_samples)
    
    return {
        'trajectories': z_history,  # [T, N, 3]
        'colors': predictions.cpu().numpy(),
        'alphas': confidences.cpu().numpy(),
        'final_images': final_images.cpu()
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import imageio.v2 as imageio
    import io
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    mysub = SubmissionInterface().to(device)
    data = generate_animation_data(mysub, n_samples=500, classifier=None)
    
    all_z = data['trajectories']
    xlim = (all_z[:,:,0].min().item(), all_z[:,:,0].max().item())
    ylim = (all_z[:,:,1].min().item(), all_z[:,:,1].max().item())
    zlim = (all_z[:,:,2].min().item(), all_z[:,:,2].max().item())
    n_show = 8
    sample_indices = torch.linspace(0, len(data['trajectories'][0])-1, n_show).long()
    
    all_grids = []
    for frame in tqdm(range(len(data['trajectories'])), desc="Generating frames"):
        z = data['trajectories'][frame]
        z_samples = z[sample_indices].to(mysub.device)
        img_samples = F.sigmoid(mysub.decode(z_samples).view(-1, 28, 28))
        grid_rows = [torch.cat([img_samples[i*4 + j] for j in range(4)], dim=1) for i in range(2)]
        grid = torch.cat(grid_rows, dim=0).detach().cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        all_grids.append(grid)
    
    gif_frames = []
    for frame in tqdm(range(len(data['trajectories'])), desc="Creating GIF frames"):
        fig = plt.figure(figsize=(24, 10))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        z = data['trajectories'][frame]
        ax1.imshow(all_grids[frame], cmap='gray'); ax1.axis('off')
        z_highlight = z[sample_indices]
        ax2.scatter(z[:,0], z[:,1], z[:,2], c=data['colors'], cmap='tab10', alpha=0.3, s=40)
        ax2.scatter(z_highlight[:,0], z_highlight[:,1], z_highlight[:,2], c='red', s=250, marker='*', edgecolors='black', linewidths=2)
        ax2.set_xlim(xlim); ax2.set_ylim(ylim); ax2.set_zlim(zlim)
        ax2.view_init(elev=20, azim=45)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        image = imageio.imread(buf)
        h, w = image.shape[:2]
        image = image[100:h-100, :]  # Crop here
        gif_frames.append(image)
        plt.close()
        buf.close()
    
    imageio.mimsave('flow_animation.gif', gif_frames, fps=10)
    print("Saved flow_animation.gif")
