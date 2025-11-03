import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys 


class PINN(nn.Module):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)-1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.net.add_module(f"tanh_{i}", nn.Tanh())
        
    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        print(inputs.shape)
        return self.net(inputs)


def physics_loss(model, x, t, alpha=0.1):
    x.requires_grad_(True)
    t.requires_grad_(True)
    
    u = model(x, t)
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                            create_graph=True, retain_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                             create_graph=True, retain_graph=True)[0]
    
    physics = u_t - alpha * u_xx
    print(physics)
    return torch.mean(physics**2)

def boundary_loss(model):
    # Левая граница: x=0, t∈[0,1]
    t_left = torch.rand(100, 1, requires_grad=True)
    x_left = torch.zeros_like(t_left)
    u_left = model(x_left, t_left)
    loss_left = torch.mean(u_left**2)  # u(0,t) = 0
    
    # Правая граница: x=1, t∈[0,1]
    t_right = torch.rand(100, 1, requires_grad=True)
    x_right = torch.ones_like(t_right)
    u_right = model(x_right, t_right)
    loss_right = torch.mean(u_right**2)  # u(1,t) = 0
    
    return loss_left + loss_right

def initial_loss(model):

    x_initial = torch.rand(100, 1, requires_grad=True)
    t_initial = torch.zeros_like(x_initial)
    u_pred = model(x_initial, t_initial)
    u_exact = torch.sin(np.pi * x_initial)  # u(x,0) = sin(πx)
    
    return torch.mean((u_pred - u_exact)**2)

def total_loss(model, x_collocation, t_collocation):
    loss_physics = physics_loss(model, x_collocation, t_collocation)
    loss_bc = boundary_loss(model)
    loss_ic = initial_loss(model)
    
    return loss_physics + 10.0 * loss_bc + 10.0 * loss_ic

print("✓ Функции потерь определены")

# 3. Основной код обучения
def train_pinn():
    print("Начинаем обучение...")
    
    # Инициализация модели
    layers = [2, 20, 20, 20, 1]
    model = PINN(layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Коллокационные точки
    N_collocation = 1000
    x_collocation = torch.rand(N_collocation, 1, requires_grad=True)
    t_collocation = torch.rand(N_collocation, 1, requires_grad=True)
    
    # Процесс обучения (упрощенный для теста)
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = total_loss(model, x_collocation, t_collocation)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Эпоха {epoch}, Потери: {loss.item():.6f}")
    
    # print("Обучение завершено!")
    return model

if __name__ == "__main__":
    model = train_pinn()
    

    x_test = torch.tensor([[0.5]], dtype=torch.float32)
    t_test = torch.tensor([[0.5]], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(x_test, t_test)
    
    print(f"\nТестовое предсказание: u(0.5, 0.5) = {prediction.item():.4f}")
    # print(f"new changes")
    print("done")
