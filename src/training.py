import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import save_samples


def train_discriminator(model, generator, real_images, opt_d, latent_size):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through discriminator
    real_preds = model.forward(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator.forward(latent)

    # Pass fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = model.forward(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    # Update discriminator weights
    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

def train_generator(model, opt_g, discriminator, latent_size):
    # Clear generator gradients
    opt_g.zero_grad()
    
    # Generate fake images
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = model.forward(latent)
    
    # Try to fool the discriminator
    preds = discriminator.forward(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)
    
    # Update generator weights
    loss.backward()
    opt_g.step()
    
    return loss.item()

def fit(discriminator, generator, epochs, lr, train, stats, start_idx=1):
    sample_dir = 'generated'
    os.makedirs(sample_dir, exist_ok=True)
    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(
        discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(
        generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_images in tqdm(train):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(
                discriminator, generator, real_images, opt_d, latent_size)
            # Train generator
            loss_g = train_generator(
                generator, opt_g, discriminator, latent_size)
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))

        torch.save(generator.state_dict(), 'model_backups/G.pth')
        torch.save(discriminator.state_dict(), 'model_backups/D.pth')
    
        # Save generated images
        save_samples(sample_dir, epoch+start_idx, fixed_latent, stats, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores
