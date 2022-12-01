import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import time
import cv2
from torchvision.models import resnet
from functools import reduce
from operator import add
if __name__ == '__main__':
    import sys
    sys.path.append('/home/user4/notebookdir/projs/PFENet-Modified')
    print('path appended')
    import os 
    os.environ["CUDA_VISIBLE_DEVICES"]="3"


def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area  
    return supp_feat
  
def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

class PFENet(nn.Module):
    def __init__(self, layers=50, classes=2, zoom_factor=8, \
        criterion=nn.CrossEntropyLoss(ignore_index=255), BatchNorm=nn.BatchNorm2d, \
        pretrained=True, sync_bn=True, shot=1, ppm_scales=[60, 30, 15, 8], vgg=False):
        super(PFENet, self).__init__()
        assert layers in [50, 101, 152]
        print(ppm_scales)
        assert classes > 1
        from torch.nn import BatchNorm2d as BatchNorm        
        self.zoom_factor = zoom_factor
        self.criterion = criterion
        self.shot = shot
        self.ppm_scales = ppm_scales
        self.vgg = vgg

        '''123'''
        backbone = 'resnet50'
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        '''123'''

        reduce_dim = 256
        if self.vgg:
            fea_dim = 512 + 256
        else:
            fea_dim = 1024 + 512       

        self.cls = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),                 
            nn.Conv2d(reduce_dim, classes, kernel_size=1)
        )                 

        self.down_query = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                  
        )
        self.down_supp = nn.Sequential(
            nn.Conv2d(fea_dim, reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)                   
        )  

        self.pyramid_bins = ppm_scales
        self.avgpool_list = []
        for bin in self.pyramid_bins:
            if bin > 1:
                self.avgpool_list.append(
                    nn.AdaptiveAvgPool2d(bin)
                )


        factor = 1
        mask_add_num = 1
        self.init_merge = []
        self.beta_conv = []
        self.inner_cls = []        
        for bin in self.pyramid_bins:
            self.init_merge.append(nn.Sequential(
                nn.Conv2d(reduce_dim*2 + mask_add_num, reduce_dim, kernel_size=1, padding=0, bias=False),
                nn.ReLU(inplace=True),
            ))                      
            self.beta_conv.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ))            
            self.inner_cls.append(nn.Sequential(
                nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.1),                 
                nn.Conv2d(reduce_dim, classes, kernel_size=1)
            ))            
        self.init_merge = nn.ModuleList(self.init_merge) 
        self.beta_conv = nn.ModuleList(self.beta_conv)
        self.inner_cls = nn.ModuleList(self.inner_cls)                             


        self.res1 = nn.Sequential(
            nn.Conv2d(reduce_dim*len(self.pyramid_bins), reduce_dim, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),                          
        )              
        self.res2 = nn.Sequential(
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),   
            nn.Conv2d(reduce_dim, reduce_dim, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                             
        )                        
     
        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.alpha_conv = []
        for idx in range(len(self.pyramid_bins)-1):
            self.alpha_conv.append(nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0, bias=False),
                nn.ReLU()
            ))     
        self.alpha_conv = nn.ModuleList(self.alpha_conv)
     
    def extract_feat(self,x):
        # 分别对应layer2,3的输出  即8s和16s
        self.feat_ids = [7,13,16]
        x_feats = self.extract_feat_res(x, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        layer2_out = x_feats[0] # 1 512 60 60
        layer3_out = x_feats[1] # 1 1024 30 30
        layer4_out = x_feats[2] # 1 2048 15 15

        layer3_out = F.interpolate(layer3_out, size=(layer2_out.size(2),layer2_out.size(3)), mode='bilinear', align_corners=True)
        layer4_out = F.interpolate(layer4_out, size=(layer2_out.size(2),layer2_out.size(3)), mode='bilinear', align_corners=True)

        return layer2_out , layer3_out , layer4_out

    def extract_feat_res(self,img, backbone, feat_ids, bottleneck_ids, lids):
        r""" Extract intermediate features from ResNet"""
        feats = []

        # Layer 0
        feat = backbone.conv1.forward(img)
        feat = backbone.bn1.forward(feat)
        feat = backbone.relu.forward(feat)
        feat = backbone.maxpool.forward(feat)

        if 0 in feat_ids:
            feats.append(feat.clone())

      # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
            res = feat
            feat = backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            if hid + 1 in feat_ids:
                feats.append(feat.clone())

            feat = backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def forward(self, x, s_x=torch.FloatTensor(1,1,3,473,473).cuda(), s_y=torch.FloatTensor(1,1,473,473).cuda(), y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        self.extract_feat(x)

        #   Query Feature
        with torch.no_grad():
            query_feat_2 ,query_feat_3, query_feat_4 = self.extract_feat(x)
            if self.vgg:
                query_feat_2 = F.interpolate(query_feat_2, size=(query_feat_3.size(2),query_feat_3.size(3)), mode='bilinear', align_corners=True)

        query_feat = torch.cat([query_feat_3, query_feat_2], 1)
        query_feat = self.down_query(query_feat)

        #   Support Feature     
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            with torch.no_grad():
                supp_feat_2 = self.layer2(supp_feat_1)
                supp_feat_3 = self.layer3(supp_feat_2)
                mask = F.interpolate(mask, size=(supp_feat_3.size(2), supp_feat_3.size(3)), mode='bilinear', align_corners=True)
                supp_feat_4 = self.layer4(supp_feat_3*mask)
                final_supp_list.append(supp_feat_4)
                if self.vgg:
                    supp_feat_2 = F.interpolate(supp_feat_2, size=(supp_feat_3.size(2),supp_feat_3.size(3)), mode='bilinear', align_corners=True)
            
            supp_feat = torch.cat([supp_feat_3, supp_feat_2], 1)
            supp_feat = self.down_supp(supp_feat)
            supp_feat = Weighted_GAP(supp_feat, mask)
            supp_feat_list.append(supp_feat)


        corr_query_mask_list = []
        cosine_eps = 1e-7
        for i, tmp_supp_feat in enumerate(final_supp_list):
            resize_size = tmp_supp_feat.size(2)
            tmp_mask = F.interpolate(mask_list[i], size=(resize_size, resize_size), mode='bilinear', align_corners=True)

            tmp_supp_feat_4 = tmp_supp_feat * tmp_mask                    
            q = query_feat_4
            s = tmp_supp_feat_4
            bsize, ch_sz, sp_sz, _ = q.size()[:]

            tmp_query = q
            tmp_query = tmp_query.contiguous().view(bsize, ch_sz, -1)
            tmp_query_norm = torch.norm(tmp_query, 2, 1, True) 

            tmp_supp = s               
            tmp_supp = tmp_supp.contiguous().view(bsize, ch_sz, -1) 
            tmp_supp = tmp_supp.contiguous().permute(0, 2, 1) 
            tmp_supp_norm = torch.norm(tmp_supp, 2, 2, True) 

            similarity = torch.bmm(tmp_supp, tmp_query)/(torch.bmm(tmp_supp_norm, tmp_query_norm) + cosine_eps)   
            similarity = similarity.max(1)[0].view(bsize, sp_sz*sp_sz)   
            similarity = (similarity - similarity.min(1)[0].unsqueeze(1))/(similarity.max(1)[0].unsqueeze(1) - similarity.min(1)[0].unsqueeze(1) + cosine_eps)
            corr_query = similarity.view(bsize, 1, sp_sz, sp_sz)
            corr_query = F.interpolate(corr_query, size=(query_feat_3.size()[2], query_feat_3.size()[3]), mode='bilinear', align_corners=True)
            corr_query_mask_list.append(corr_query)  
        corr_query_mask = torch.cat(corr_query_mask_list, 1).mean(1).unsqueeze(1)     
        corr_query_mask = F.interpolate(corr_query_mask, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)  

        if self.shot > 1:
            supp_feat = supp_feat_list[0]
            for i in range(1, len(supp_feat_list)):
                supp_feat += supp_feat_list[i]
            supp_feat /= len(supp_feat_list)

        out_list = []
        pyramid_feat_list = []

        for idx, tmp_bin in enumerate(self.pyramid_bins):
            if tmp_bin <= 1.0:
                bin = int(query_feat.shape[2] * tmp_bin)
                query_feat_bin = nn.AdaptiveAvgPool2d(bin)(query_feat)
            else:
                bin = tmp_bin
                query_feat_bin = self.avgpool_list[idx](query_feat)
            supp_feat_bin = supp_feat.expand(-1, -1, bin, bin)
            corr_mask_bin = F.interpolate(corr_query_mask, size=(bin, bin), mode='bilinear', align_corners=True)
            merge_feat_bin = torch.cat([query_feat_bin, supp_feat_bin, corr_mask_bin], 1)
            merge_feat_bin = self.init_merge[idx](merge_feat_bin)

            if idx >= 1:
                pre_feat_bin = pyramid_feat_list[idx-1].clone()
                pre_feat_bin = F.interpolate(pre_feat_bin, size=(bin, bin), mode='bilinear', align_corners=True)
                rec_feat_bin = torch.cat([merge_feat_bin, pre_feat_bin], 1)
                merge_feat_bin = self.alpha_conv[idx-1](rec_feat_bin) + merge_feat_bin  

            merge_feat_bin = self.beta_conv[idx](merge_feat_bin) + merge_feat_bin   
            inner_out_bin = self.inner_cls[idx](merge_feat_bin)
            merge_feat_bin = F.interpolate(merge_feat_bin, size=(query_feat.size(2), query_feat.size(3)), mode='bilinear', align_corners=True)
            pyramid_feat_list.append(merge_feat_bin)
            out_list.append(inner_out_bin)
                 
        query_feat = torch.cat(pyramid_feat_list, 1)
        query_feat = self.res1(query_feat)
        query_feat = self.res2(query_feat) + query_feat           
        out = self.cls(query_feat)
        

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y.long())
            aux_loss = torch.zeros_like(main_loss).cuda()    

            for idx_k in range(len(out_list)):    
                inner_out = out_list[idx_k]
                inner_out = F.interpolate(inner_out, size=(h, w), mode='bilinear', align_corners=True)
                aux_loss = aux_loss + self.criterion(inner_out, y.long())   
            aux_loss = aux_loss / len(out_list)
            return out.max(1)[1], main_loss, aux_loss
        else:
            return out


if __name__ == '__main__':
    img_size=473
    model = PFENet()
    model.train()
    query_x = torch.randn(1,3,img_size,img_size)
    query_y = torch.rand(1,img_size,img_size)
    support_x = torch.rand(1,1,3,img_size,img_size)
    support_y = torch.rand(1,1,img_size,img_size)
    predict_logit , _ , _ = model(query_x,support_x,support_y,query_y)
    print(predict_logit.shape)
    print(predict_logit.unique())




