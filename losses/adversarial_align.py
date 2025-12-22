# losses/adversarial_align.py
import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """一个简单的判别器，用于判断输入是来自真实SigLIP还是生成的e_t"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class AdversarialAlignLoss:
    """
    RDT接口分布对齐的对抗性损失。
    包含生成器损失和判别器损失的计算逻辑。
    """
    def __init__(self, discriminator_input_dim=768):
        self.discriminator = Discriminator(input_dim=discriminator_input_dim)
        self.adversarial_loss_fn = nn.BCELoss() # 二元交叉熵

    def get_discriminator(self):
        return self.discriminator

    def calculate_discriminator_loss(self, generated_e_t, real_siglip_features):
        """计算判别器的损失"""
        # .detach() 是为了在训练判别器时不计算生成器的梯度
        d_real = self.discriminator(real_siglip_features)
        d_fake = self.discriminator(generated_e_t.detach())
        
        loss_real = self.adversarial_loss_fn(d_real, torch.ones_like(d_real))
        loss_fake = self.adversarial_loss_fn(d_fake, torch.zeros_like(d_fake))
        
        d_loss = (loss_real + loss_fake) / 2
        return d_loss

    def calculate_generator_loss(self, generated_e_t):
        """计算生成器（即我们的FusionEncoder）的损失"""
        # 目标是让判别器将生成的e_t误判为1（真实）
        d_fake = self.discriminator(generated_e_t)
        g_loss = self.adversarial_loss_fn(d_fake, torch.ones_like(d_fake))
        return g_loss