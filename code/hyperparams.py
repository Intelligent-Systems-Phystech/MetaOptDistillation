# This code is a modiciation of https://github.com/khanrc/pt.darts/blob/master/architect.py
import copy
import torch
import math



def adam(param,
         grad,
         exp_avg,
         exp_avg_sq,
         step,
         beta1: float,
         beta2: float,
         lr: float,
         eps: float):
    r"""non-inplace version of https://github.com/pytorch/pytorch/blob/master/torch/optim/_functional.py
    weight_decay = 0,
    amsgrad = False
    max_exp_avg_sq = None
    """   
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step

    exp_avg = exp_avg.clone()
    exp_avg_sq = exp_avg_sq.clone()
    param = param.clone()
    
    # Decay the first and second moment running average coefficient
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
  
    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

    step_size = lr / bias_correction1

    param.addcdiv_(exp_avg, denom, value=-step_size)
    return param


class AdamHyperGradCalculator():
    """ Compute gradients of hyperparameters wrt parameters are optimizaed by Adam """
    def __init__(self,  net, parameters_loss_function, hyperparameters_loss_function, optimizer, h):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net =  None # lazy 
        self.w = list(self.net.parameters())
        self.w_loss = parameters_loss_function #data,model, h
        self.h_loss = hyperparameters_loss_function #x,y,model
        self.optimizer = optimizer
        self.h = list(h)



    def virtual_step(self, trn):
        """
        Compute unrolled weight w' (virtual step)
        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient
        """
        # forward & calc loss
        lr = self.optimizer.param_groups[0]['lr']
        h = self.h 
        optimizer = self.optimizer
        
        loss = self.w_loss(trn, self.net, h) # L_trn(w)

        # compute gradient
        gradients = torch.autograd.grad(loss, self.net.parameters())

        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        with torch.no_grad():
            # dict key is not the value, but the pointer. So original network weight have to
            # be iterated also.
            for w, vw, g in zip(self.net.parameters(), self.v_net.parameters(), gradients):           
                state = optimizer.state[w]
                
                # Lazy state initialization: not ready yet                    
                if len(state) == 0:
                    return                 
                vw.copy_(adam(w,
                             g,
                             state['exp_avg'],
                             state['exp_avg_sq'],                            
                             state['step'],
                             optimizer.param_groups[0]['betas'][0],
                             optimizer.param_groups[0]['betas'][1],
                             optimizer.param_groups[0]['lr'],
                             optimizer.param_groups[0]['eps']))
      
    def calc_gradients(self, trn, val):
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        lr = self.optimizer.param_groups[0]['lr']
        h = self.h 
        optimizer = self.optimizer
        
        if self.v_net is None:
            self.v_net = copy.deepcopy(self.net)        
        # do virtual step (calc w`)
        self.virtual_step(trn)

        # calc unrolled loss
        loss = self.h_loss(val, self.v_net) # L_val(w`)
           
        v_grads = torch.autograd.grad(loss,list(self.v_net.parameters()))
        dw = v_grads

        hessian = self.compute_hessian(dw, trn)

        # update final gradient = dalpha - xi*hessian
        with torch.no_grad():
            for alpha,  he in zip(h,  hessian):
                alpha.grad =  lr*he

    def compute_hessian(self, dw, trn):
        """
        dw = dw` { L_val(w`, alpha) }
        w+ = w + eps * dw
        w- = w - eps * dw
        hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        eps = 0.01 / ||dw||
        """
        h = self.h
        norm = torch.cat([w.view(-1) for w in dw]).norm()
        eps = 0.01 / norm

        # w+ = w + eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                p += eps * d
        loss = self.w_loss(trn, self.net, h)
        dalpha_pos = torch.autograd.grad(loss, h) # dalpha { L_trn(w+) }

        # w- = w - eps*dw`
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                p -= 2. * eps * d
        loss = self.w_loss(trn, self.net, h)
        dalpha_neg = torch.autograd.grad(loss, h) # dalpha { L_trn(w-) }

        # recover w
        with torch.no_grad():
            for p, d in zip(self.net.parameters(), dw):
                p += eps * d

        hessian = [(p-n) / 2.*eps for p, n in zip(dalpha_pos, dalpha_neg)]
        return hessian
