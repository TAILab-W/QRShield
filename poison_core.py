import torch
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms.functional as TF
PURPLE = '\033[95m'
BLUE = '\033[94m'
RESET = '\033[0m'


class HardMask(torch.nn.Module):
    def __init__(self, temperature=5.0, value = 1.0):
        super(HardMask,self).__init__()
        self.temperature = temperature
        self.value = value

    def forward(self, x):
        soft = torch.tanh(x * self.temperature) * self.value
        hard = torch.sign(x + 1e-6) * self.value
        return soft + (hard - soft).detach()      
    

def high_freq_energy(x, sigma=1.0):
    # x: [B, C, H, W] latent
    with torch.no_grad():
        x_blur = TF.gaussian_blur(x, kernel_size=9, sigma=sigma)
    high = x - x_blur
    energy = high.pow(2).mean()
    return - energy


def latent_anchor_optimize(
    args,
    vae,
    clean_tensors, 
):
    # prepare vae
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")
    vae.to(device, dtype=weight_dtype).requires_grad_(False)
    hard_mask = HardMask(value=1.0)
    
    # prepare tensors
    clean_tensors = clean_tensors.to(device, dtype=weight_dtype).requires_grad_(False)
    
    with torch.no_grad():
        clean_latents = []
        for i in range(clean_tensors.size(0)):
            tensor = clean_tensors[i:i+1]  # shape (1, C, H, W)
            latent = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
            clean_latents.append(latent)
        clean_latents = torch.cat(clean_latents, dim=0)  # shape (N, C, H, W)
    
    # --- Learnable parameters ---
    tar_kernel = torch.zeros(1, 1, args.block_size - args.block_padding, args.block_size - args.block_padding, device=device, dtype=weight_dtype, requires_grad=True)
    # tar_kernel.data.mul_(1e-6)
    optimizer = torch.optim.Adam([tar_kernel], lr=args.ag_lr)
    
    # start anchor optimization 
    tar_latent = None
    pbar = tqdm(range(args.ag_round))
    for step in pbar:
        patch_16 = F.pad(tar_kernel, (0, args.block_padding, 0, args.block_padding))
        tiled = patch_16.expand(1, 1, 64 // args.block_size, 64 // args.block_size, args.block_size, args.block_size)
        tiled = tiled.permute(0, 1, 2, 4, 3, 5).reshape(1, 1, 64, 64)  # shape (1,1,64,64)
        tar_latent = hard_mask(tiled.repeat(1, 4, 1, 1))
        moved_latents = 0.7 * clean_latents + 0.3 * tar_latent     
        # compute LPIPS loss between clean_tensor and moved_tensor
        hf_loss = high_freq_energy(moved_latents)
        # Backpropagate LPIPS loss to update parameters
        optimizer.zero_grad()
        hf_loss.backward()
        optimizer.step()
        pbar.set_description(f"{BLUE}[latent anchor optimizing...]{RESET} timestep:{step + 1} | hf_loss:{hf_loss.item():.5f}")
        
    pbar.close()
    return tar_latent.detach()


def create_gaussian_kernel(k=2, sigma=0.8, channels=3):
    x = torch.arange(-k, k+1, dtype=torch.float32)
    x = torch.exp(-0.5 * (x / sigma) ** 2)
    x = x / x.sum()
    
    kernel2d = x[:, None] @ x[None, :]
    kernel2d = kernel2d / kernel2d.sum()

    kernel = kernel2d.expand(channels, 1, 2 * k + 1, 2 * k + 1).clone()
    return kernel


def pgd_with_momentum(
    args,
    vae,
    clean_tensor, 
    target_latent, 
    tensor_index,
):
    # prepare vae model
    weight_dtype = torch.bfloat16
    device = torch.device("cuda")
    vae.to(device, dtype=weight_dtype).requires_grad_(False)
    
    # prepare tensors in need
    clean_tensor = clean_tensor.to(device, dtype=weight_dtype).requires_grad_(False)
    tar_latent = target_latent.to(device, dtype=weight_dtype).requires_grad_(False)
    adversarial_tensor = clean_tensor.clone().requires_grad_(True)
    # initialize momentum and kernel
    kernel = create_gaussian_kernel().to(device, dtype=weight_dtype)  # 15x15
    padding = 2
    g = torch.zeros_like(adversarial_tensor)
    # start poisoning
    pbar = tqdm(range(args.p_round))
    pgd_lr = args.pgd_lr
    for step in pbar:
        if step <= 70:
            args.pgd_lr = 4/255
        else:
            args.pgd_lr = pgd_lr
        
        adversarial_tensor = adversarial_tensor.detach().clone().requires_grad_(True)
        adv_latent = vae.encode(adversarial_tensor).latent_dist.sample() * vae.config.scaling_factor
        utility = - F.mse_loss(adv_latent.float(), tar_latent.float())
        
        grad = torch.autograd.grad(utility, adversarial_tensor)[0]
        grad_conv = F.conv2d(grad, kernel, padding=padding, groups=3)
            
        # normalize grad and apply momentum
        grad_norm = grad_conv / (grad_conv.abs().mean() + 1e-8)
        g = args.p_mu * g + grad_norm

        with torch.no_grad():
            # updating adversarial_tensor using PGD with momentum
            step_tensor = adversarial_tensor + args.pgd_lr * g.sign()
            delta = torch.clamp(step_tensor - clean_tensor, min=-args.delta_range, max=+args.delta_range)
            adversarial_tensor = torch.clamp(clean_tensor + delta, min=-1.0, max=1.0).detach_()
            pbar.set_description(f"{PURPLE}[{tensor_index}: pgd with momentum...]{RESET} step:{step + 1} | utility:{utility.item():.5f}")
        
    pbar.close()
    return adversarial_tensor.detach()












