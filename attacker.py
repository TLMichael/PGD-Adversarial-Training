import torch
import torch.nn as nn


class LinfStep(object):
    def __init__(self, orig_input, eps, step_size):
        self.orig_input = orig_input
        self.eps = eps
        self.step_size = step_size
  
    def project(self, x):
        diff = x - self.orig_input
        diff = torch.clamp(diff, -self.eps, self.eps)
        x = diff + self.orig_input
        return torch.clamp(x, 0, 1)
    
    def step(self, x, g):
        step = torch.sign(g) * self.step_size
        return x + step
    
    def random_perturb(self, x):
        new_x = x + 2 * (torch.rand_like(x) - 0.5) * self.eps
        return torch.clamp(new_x, 0, 1)


class AttackerModel(nn.Module):
    def __init__(self, model, config):
        super(AttackerModel, self).__init__()
        self.model = model
        self.random_start = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        self.step_sign = -1 if config['targeted'] else 1
        self.classifier_criterion = nn.CrossEntropyLoss()
    
    def calc_loss(self, x, target_label):
        logits = self.model(x)
        loss = self.step_sign * self.classifier_criterion(logits, target_label)
        return loss
    
    def attack(self, x, target):
        orig_x = x.clone().detach()
        
        step = LinfStep(orig_input=orig_x, eps=self.epsilon, step_size=self.step_size)
        if self.random_start:
            x = step.random_perturb(x)

        for _ in range(self.num_steps):
            x = x.clone().detach().requires_grad_(True)
            
            loss = self.calc_loss(x, target)
            grad = torch.autograd.grad(loss, [x])[0]
            
            with torch.no_grad():
                x = step.step(x, grad)
                x = step.project(x)

        return x.clone().detach()
        
    def forward(self, inp, make_adv=False, target=None):
        adv = None
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attack(inp.clone().detach(), target)
            if prev_training:
                self.train()
            inp = adv

        output = self.model(inp)
        return output, adv
        




