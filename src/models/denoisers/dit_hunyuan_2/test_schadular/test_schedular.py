"""
用于测试不同的 flowmatching schedular
"""
from src.models.denoisers.dit_hunyuan_2.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers import  FlowMatchHeunDiscreteScheduler
from diffusers.schedulers import  DDPMScheduler
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # 可换成 FlowMatchingScheduler
import numpy as np

def load_image_and_addnoise(scheduler):
    # ===== 1. 读取图片并调整为 128x128 =====
    img_path = "src/models/denoisers/dit_hunyuan_2/test_schadular/test_image.png"  # 改为你的图片路径
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),  # [0,1]
    ])
    image_tensor = transform(image)  # shape: (3,128,128)
    image_tensor = image_tensor.unsqueeze(0)  # 加 batch 维度，(1,3,128,128)

    # ===== 2. 初始化 scheduler，并获取加噪图像 =====
    scheduler.set_timesteps(1000)  # 50 个时间步（可改为更多）

    for t in [50, 100, 200, 400, 800]:
        # 任意选择一个 timestep，比如 25 步的噪声程度：
        timestep = scheduler.timesteps[t][None, ...]
        noise = torch.randn_like(image_tensor)
        noisy_image = scheduler.scale_noise(sample=image_tensor, noise=noise, timestep=timestep)

        # ===== 3. 可视化原图和加噪图像 =====
        def show_image(img_tensor, title):
            img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(np.clip(img, 0, 1))
            plt.title(title)
            plt.axis("off")

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        show_image(image_tensor, f"original")
        plt.subplot(1, 2, 2)
        show_image(noisy_image, f"t={t}, shift={scheduler. shift}")
        plt.show()

def vis_shift(schedular):
    import matplotlib.pyplot as plt
    x = np.arange(0, len(schedular.sigmas))
    y=schedular.sigmas.detach().cpu().numpy()
    plt.plot(x, y, marker='o')
    plt.xlabel('timestep')
    plt.ylabel('sigma')
    plt.title(f'shift = {schedular.shift}')
    plt.grid(True)
    plt.show()


def add_noise_and_run_steps(scheduler):
    img_path = "src/models/denoisers/dit_hunyuan_2/test_schadular/test_image.png"  # 改为你的图片路径
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),  # [0,1]
    ])
    image_tensor = transform(image)  # shape: (3,128,128)
    image_tensor = image_tensor.unsqueeze(0)  # 加 batch 维度，(1,3,128,128)

    scheduler.set_timesteps(1000)  # 50 个时间步（可改为更多）

    timestep = scheduler.timesteps[500][None, ...]
    noise = torch.randn_like(image_tensor)
    noisy_image = scheduler.scale_noise(sample=image_tensor, noise=noise, timestep=timestep)




    # ===== 3. 可视化原图和加噪图像 =====
    def show_image(img_tensor, title):
        img = img_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(np.clip(img, 0, 1))
        plt.title(title)
        plt.axis("off")

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    show_image(image_tensor, f"original")
    plt.subplot(1, 2, 2)
    show_image(noisy_image, f"t={t}, shift={scheduler. shift}")
    plt.show()



if __name__ == "__main__":
    SCHEDULAR = FlowMatchEulerDiscreteScheduler

    scheduler = SCHEDULAR(num_train_timesteps=1000)

    # # test different shift
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=5)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=3)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=2)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=1)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=0.8)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=0.6)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=0.4)
    # )
    # vis_shift(
    #     SCHEDULAR(num_train_timesteps=1000, shift=0.2)
    # )

    # # test scale_noise ===
    # load_image_and_addnoise(SCHEDULAR(num_train_timesteps=1000, shift=1))
    # load_image_and_addnoise(SCHEDULAR(num_train_timesteps=1000, shift=0.6))
    # load_image_and_addnoise(SCHEDULAR(num_train_timesteps=1000, shift=3))

    # 测试多图加噪 ===
    img_path = "src/models/denoisers/dit_hunyuan_2/test_schadular/test_image.png"  # 改为你的图片路径
    image = Image.open(img_path).convert("RGB")
    transform = T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),  # [0,1]
    ])
    image_tensor = transform(image)  # shape: (3,128,128)
    image_tensor = image_tensor.unsqueeze(0)  # 加 batch 维度，(1,3,128,128)
    image_tensors = torch.cat([image_tensor, image_tensor], dim=0)
    scheduler.set_timesteps(1000)  # 50 个时间步（可改为更多）
    timestep = torch.tensor([scheduler.timesteps[100], scheduler.timesteps[200]])
    noise = torch.randn_like(image_tensor)
    noisy_image = scheduler.scale_noise(sample=image_tensors, noise=noise, timestep=timestep)
