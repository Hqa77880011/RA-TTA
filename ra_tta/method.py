import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .layers import Encoder


class Model(nn.Module):
    def __init__(self,c):
        super().__init__();self.c=c;w=c.get('width',24);self.encoder=Encoder(3,w)
        self.language=nn.Linear(w,w);self.adapter=nn.Parameter(torch.eye(w));self.box=nn.Linear(w,4)
    def forward(self,b):
        frames=b['x'].float();n,time,_,height,width=frames.shape;language=F.normalize(self.language(b['language'].float()),dim=-1)
        outputs=[];risks=[];updates=[]
        for row in range(n):
            feature=self.encoder(frames[row,:1]);box=b['initial_box'][row];yy,xx=torch.meshgrid(torch.linspace(0,1,height,device=frames.device),torch.linspace(0,1,width,device=frames.device),indexing='ij')
            initial=((xx>=box[0])&(xx<=box[2])&(yy>=box[1])&(yy<=box[3])).float();template=(feature[0]*initial).sum((1,2))/initial.sum().clamp_min(1)
            memory=[template];quality=[torch.tensor(1.,device=frames.device)];adapter=self.adapter.clone();predictions=[];risk=[];changed=[]
            for t in range(time):
                token=self.encoder(frames[row,t:t+1]).flatten(2)[0].T;token=F.normalize(token@adapter,dim=-1)
                mem=torch.stack(memory);weights=torch.stack(quality).softmax(0);prototype=(weights[:,None]*mem).sum(0)
                visual=F.normalize(template,dim=-1);similarity=token@visual;prob=(similarity*10).softmax(0)
                entropy=-(prob*prob.clamp_min(1e-8).log()).sum()/np.log(len(prob));peak=(prob.max()-prob.mean())/(prob.max()+1e-8)
                current=prob@token;rm=(1+F.cosine_similarity(current[None],prototype[None]))[0]/2
                rv=(1-entropy+peak+rm)/3;language_score=token@language[row];ordered=language_score.topk(min(2,len(token))).values
                rl=torch.sigmoid((ordered[0]-ordered[-1])/.1);occlusion=1-(peak+rm)/2
                if self.c['variant']=='no_language':rl=rl*0
                fusion=torch.stack([rv,rl,rm]).softmax(0)
                if self.c['variant']=='no_language':
                    fusion=fusion*frames.new_tensor([1.,0.,1.]);fusion=fusion/fusion.sum()
                reference=fusion[0]*visual+fusion[1]*language[row]+fusion[2]*prototype
                score=(token@F.normalize(reference,dim=-1)*10).softmax(0);center=score@torch.stack([xx.flatten(),yy.flatten()],-1)
                size=(box[2:]-box[:2]).clamp_min(.02);delta=self.box(score@token).tanh()*.05
                prediction=torch.cat([center-size/2,center+size/2])+delta;predictions.append(prediction.clamp(0,1))
                gate=(rv>.35)&(occlusion<.8)
                if self.c['variant']!='no_language':gate=gate&(1-rl<.8)
                if self.c['variant']=='no_gate':gate=torch.tensor(True,device=frames.device)
                if self.c['variant']=='baseline':gate=torch.tensor(False,device=frames.device)
                if bool(gate) and not self.training:
                    augmented=self.encoder((frames[row,t:t+1]*1.05+.02).clamp(0,1)).flatten(2)[0].T
                    transformed=F.normalize(augmented@adapter,dim=-1)
                    consistency=(transformed-token.detach()).square().mean()+.01*(adapter-self.adapter.detach()).square().mean()
                    gradient=torch.autograd.grad(consistency,adapter,retain_graph=True)[0];adapter=(self.adapter.detach()+.995*(adapter-self.c.get('adapt_lr',.01)*gradient-self.adapter.detach())).detach().requires_grad_(True)
                if bool(gate) and self.c['variant']!='no_memory':
                    memory.append(current.detach());quality.append((peak*rv*rl*(1-occlusion)).detach());limit=self.c.get('memory_size',8);memory=memory[-limit:];quality=quality[-limit:]
                risk.append((1-rv)+(1-rl)+occlusion);changed.append(gate.float())
            outputs.append(torch.stack(predictions));risks.append(torch.stack(risk));updates.append(torch.stack(changed))
        return {'prediction':torch.stack(outputs),'risk':torch.stack(risks),'adapted':torch.stack(updates)}
    def objective(self,b,p):return F.smooth_l1_loss(p['prediction'],b['y'].float())


def demo(seed,n,c):
    rng=np.random.default_rng(seed);box=np.tile([.2,.2,.7,.7],(n,1)).astype('float32')
    return {'x':rng.random((n,4,3,16,16),dtype=np.float32),'y':np.tile(box[:,None],(1,4,1)),'initial_box':box,
            'language':rng.normal(size=(n,c.get('width',24))).astype('float32'),'group':np.arange(n)}


def measure(p,d,c):
    a=p['prediction'];b=d['y'];intersection=np.maximum(0,np.minimum(a[...,2:],b[...,2:])-np.maximum(a[...,:2],b[...,:2])).prod(-1)
    union=np.maximum(0,a[...,2:]-a[...,:2]).prod(-1)+np.maximum(0,b[...,2:]-b[...,:2]).prod(-1)-intersection;iou=intersection/np.maximum(union,1e-8)
    error=np.linalg.norm((a[...,:2]+a[...,2:]-b[...,:2]-b[...,2:])/2,axis=-1)
    return {'success_auc':float(np.mean([np.mean(iou>=t) for t in np.linspace(0,1,101)])),
            'normalized_precision':float(np.mean(error<=.05)),'mean_iou':float(iou.mean()),'update_fraction':float(p['adapted'].mean())}
