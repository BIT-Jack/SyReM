import sys
sys.path.append("..")

from traj_predictor import MammothBackbone
from  traj_predictor.baselayers import *
from traj_predictor.encoder import *
from traj_predictor.decoder import *

class UQnet(MammothBackbone):   # MammothBackbone 
    def __init__(self, para, test=False, drivable=True):
        super(UQnet, self).__init__()
        self.xmax = para['xmax']
        self.ymin = para['ymin']
        self.ymax = para['ymax']
        self.resolution = para['resolution']
        self.test = test
        self.prob_mode = para['prob_mode']
        self.inference = para['inference']
        
        self.encoder = VectorEncoder(para)
        decoder_dims = para['encoder_attention_size']
        
        lateral = torch.tensor([i+0.5 for i in range(int(-self.xmax/self.resolution), 
                                                         int(self.xmax/self.resolution))])*self.resolution
        longitudinal = torch.tensor([i+0.5 for i in range(int(self.ymin/self.resolution), 
                                                     int(self.ymax/self.resolution))])*self.resolution

        self.len_x = lateral.size(0)
        self.len_y = longitudinal.size(0)
        x1 = lateral.repeat(self.len_y, 1).transpose(1,0)
        y1 = longitudinal.repeat(self.len_x, 1)
        self.mesh = nn.Parameter(torch.stack((x1,y1),-1),requires_grad = False)

        self.decoder = VectorDecoder(para, drivable)
        if not self.inference:
            self.reg_decoder = RegularizeDecoder(para, drivable)
       
    def forward(self, x):
        trajectory, maps, masker, lanefeatures, adj, af, ar, c_mask = x
        if self.inference:
            hlane, hmid, hinteraction = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)
        else:
            hlane, hmid, hinteraction, hmae = self.encoder(maps, trajectory, lanefeatures, adj, af, c_mask)
        grid = self.mesh.reshape(-1, 2)

        log_lanescore, heatmap = self.decoder(hlane, hmid, hinteraction, grid, c_mask, masker)
        heatmap = heatmap.reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
        
        if not self.inference:
            heatmap_reg = self.reg_decoder(hmae, grid, ar, c_mask, masker)
            heatmap_reg = heatmap_reg.reshape(maps.size(0), self.mesh.size(0), self.mesh.size(1))
            return log_lanescore, heatmap, heatmap_reg
        else:
            if not self.test:
                return log_lanescore, heatmap
            else:
                if self.prob_mode=='nll':
                    out = torch.exp(heatmap)#*masker
                else:
                    out = torch.sigmoid(heatmap)
                    out = torch.clamp(out, min=1e-7)
                    if self.resolution==0.5:
                        out = nn.AvgPool2d(3,stride=1,padding=1)(out.unsqueeze(0))
                return torch.exp(log_lanescore), out.squeeze()
        

