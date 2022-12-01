import torch
from torch import  nn,einsum
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg
from model.resnet import Bottleneck,BasicBlock
import numpy as np
from functools import reduce
from operator import add

import model.svf as svf


class ASPP(nn.Module):
	def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
		super(ASPP, self).__init__()
		self.branch1 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate,bias=True),
				# nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),
		)
		self.branch2 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=6*rate, dilation=6*rate, bias=True),
				# nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch3 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=12*rate, dilation=12*rate, bias=True),
				# nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch4 = nn.Sequential(
				nn.Conv2d(dim_in, dim_out, 3, 1, padding=18*rate, dilation=18*rate, bias=True),
				# nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),	
		)
		self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0,bias=True)
		self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
		self.branch5_relu = nn.ReLU(inplace=True)

		self.conv_cat = nn.Sequential(
				nn.Conv2d(dim_out*5, dim_out, 1, 1, padding=0,bias=True),
				# nn.BatchNorm2d(dim_out, momentum=bn_mom),
				nn.ReLU(inplace=True),		
		)

	def forward(self, x):
		[b, c, row, col] = x.size()
        #-----------------------------------------#
        #   一共五个分支
        #-----------------------------------------#
		conv1x1 = self.branch1(x)
		conv3x3_1 = self.branch2(x)
		conv3x3_2 = self.branch3(x)
		conv3x3_3 = self.branch4(x)
        #-----------------------------------------#
        #   第五个分支，全局平均池化+卷积
        #-----------------------------------------#
		global_feature = torch.mean(x,2,True)
		global_feature = torch.mean(global_feature,3,True)
		global_feature = self.branch5_conv(global_feature)
		# global_feature = self.branch5_bn(global_feature)
		global_feature = self.branch5_relu(global_feature)
		global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
		
        #-----------------------------------------#
        #   将五个分支的内容堆叠起来
        #   然后1x1卷积整合特征。
        #-----------------------------------------#
		feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
		result = self.conv_cat(feature_cat)
		return result


class GAU(nn.Module):
    def __init__(self,inchannel=1536,outchannel=256,equalweight=False):
        super(GAU,self).__init__()
        self.equalweight=equalweight
        self.supp_linear = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1,padding=0, bias=False),
            )
        self.query_linear = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1,padding=0, bias=False),
            )
        self.final_linear = nn.Sequential(
            nn.Conv2d(outchannel*2, outchannel, kernel_size=1, stride=1,padding=0, bias=True),
            nn.ReLU(inplace=True)
            )
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,feat_supp,feat_query,shared_LPF,mask_supp):
        b , d , h , w = feat_supp.shape
        v_supp = shared_LPF(feat_supp)
        v_query = shared_LPF(feat_query)

        if not self.equalweight:
            k = self.supp_linear(feat_supp)
            q = self.query_linear(feat_query)
            
            # k = rearrange(k,'b d h w -> b (h w) d')
            k = k.permute(0,2,3,1).view(b,h*w,-1)
            # q = rearrange(q,'b d h w -> b (h w) d')
            q = q.permute(0,2,3,1).view(b,h*w,-1)
            # i for q index
            # j for k index
            # attn = einsum('b i d , b j d -> b i j',q,k)
            attn = torch.bmm(q,k.transpose(1,2))
            attn = self.suppress_bg_attn(mask_supp,attn,h,w)
            attn = self.softmax(attn)
        else:
            attn = torch.zeros(b,h*w,h*w).cuda()
            attn = self.suppress_bg_attn(mask_supp,attn,h,w)
            attn = self.softmax(attn)
        
        # v_supp = rearrange(v_supp,'b d h w -> b (h w) d')
        v_supp = v_supp.permute(0,2,3,1).view(b,h*w,-1)

        # out = einsum('b i j,b j d -> b i d' , attn , v_supp,optimize=True)
        # out = torch.bmm(attn,v_supp)
        
        # out = einsum('b i j,b j d -> b i d' , attn , v_supp,optimize=True)
        out = torch.bmm(attn,v_supp)
        
        # out = rearrange(out,'b (h w) d -> b d h w',h=h) 
        out = out.view(b,h,w,-1).permute(0,3,1,2)

        attnedfeat_query =  torch.cat((out,v_query),dim=1)
        attnedfeat_query = self.final_linear(attnedfeat_query)

        return attnedfeat_query, attn.view(b,h,w,h,w)
    def suppress_bg_attn(self,mask_supp,attn,h,w):
        # attn b h*w h*w
        # mask_supp b c h w
        fg_mask = F.interpolate(mask_supp.float(), (h,w), mode='bilinear', align_corners=True)
        bg_suppress = (1-fg_mask)*-999
        bg_suppress = bg_suppress.view(attn.shape[0],1,-1)
        return attn + bg_suppress


class PGNet(nn.Module):
    def __init__(self,backbone='resnet50',featsize=(50,50),num_classes=2,svf_on=True):
        super(PGNet,self).__init__()
        self.feat_size=featsize
        self.backbone_type = backbone
        if backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            # 分别对应layer2,3的输出  即8s和16s
            self.feat_ids = [7,13]
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)
        
        if svf_on:
            # 将卷积进行奇异值分解
            self.backbone = svf.resolver(self.backbone)
            self.svf_freeze(self.backbone)
        else:
            # 完全冻结
            self.model_freeze(self.backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])

        self.feat2value = nn.Sequential(
            nn.Conv2d(1536, 512, kernel_size=1, stride=1,padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1,padding=0, bias=True)
            )
            
        self.gau1s_ew = GAU(equalweight=True) # equal weight attention
        self.gau1s = GAU()
        self.gau2s = GAU()
        self.gau4s = GAU()
        self.aap2s = nn.AdaptiveAvgPool2d((self.feat_size[0]//2,self.feat_size[1]//2))
        self.aap4s = nn.AdaptiveAvgPool2d((self.feat_size[0]//4,self.feat_size[1]//4))


        self.resconv1 = BasicBlock(256,256)
        self.resconv2 = BasicBlock(256,256)
        self.resconv3 = BasicBlock(256,256)
        self.aspp = ASPP(dim_in=256, dim_out=256, rate=1)

        self.cls_conv = nn.Conv2d(256, num_classes, 1, stride=1)

    def svf_freeze(self,model):
        count=0
        for name, param in model.named_parameters():
            param.requires_grad = False
            if ('vector_S' in name) and ('layer2' in name or 'layer3' in name or 'layer4' in name):
                param.requires_grad = True
                count+=1
        print(f'{count} params requires_grad')
        
    
    def model_freeze(self,model):
        for name, param in model.named_parameters():
            param.requires_grad = False

    def extract_feat(self,x):
        x_feats = self.extract_feat_res(x, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
        l2_out = x_feats[0]
        l3_out = x_feats[1]
        assert l2_out.size()[-2] == self.feat_size[0]
        l3_x2 = F.interpolate(l3_out,l2_out.shape[-2:], mode='bilinear', align_corners=True)
        feat = torch.cat((l3_x2,l2_out),dim=1)
        return feat

    def mask_feature(self, features, mask):
        # features: B C h w
        # mask B 1 H W
        # 注意会污染原来的features（不过问题不大）
        for idx, feature in enumerate(features):
            curr_mask = F.interpolate(mask[idx].unsqueeze(0).float(), feature.size()[-2:], mode='nearest')
            features[idx] = features[idx] * curr_mask[0]
        return features

    def extract_feat_res(self,img, backbone, feat_ids, bottleneck_ids, lids):
      r""" Extract intermediate features from ResNet"""
      feats = []

      # Layer 0
      feat = backbone.conv1.forward(img)
      feat = backbone.bn1.forward(feat)
      feat = backbone.relu.forward(feat)
      feat = backbone.maxpool.forward(feat)

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

    def forward(self,support_x,support_y,query_x):
        
        b,c,h,w = support_x.shape
        ##################
        # support_x: B x C x H x W
        # support_y: B x 1 x H x W
        # query_x: B x C x H x W
        # RETURN : B X C X H X W
        ##################

        feat_supp = self.extract_feat(support_x) # 1 , 1536 , 28 , 28
        # feat_supp还需要mask掉
        # feat_supp = self.mask_feature(feat_supp,support_y)
        feat_query = self.extract_feat(query_x) # 1 , 1536 , 28 , 28

        gau1s_out , attn_1s = self.gau1s(feat_supp,feat_query,self.feat2value,support_y) # torch.Size([1, 256, 28, 28])
        gau1s_ew_out , attn_1s_ew = self.gau1s_ew(feat_supp,feat_query,self.feat2value,support_y) # torch.Size([1, 256, 28, 28])
        gau2s_out , attn_2s = self.gau2s(self.aap2s(feat_supp),self.aap2s(feat_query),self.feat2value,support_y) # torch.Size([1, 256, 14, 14])
        gau4s_out , attn_4s = self.gau4s(self.aap4s(feat_supp),self.aap4s(feat_query),self.feat2value,support_y) # torch.Size([1, 256, 7, 7])

        gau2s_out_ip = F.interpolate(gau2s_out,gau1s_out.shape[-2:], mode='bilinear', align_corners=True)
        gau4s_out_ip = F.interpolate(gau4s_out,gau1s_out.shape[-2:], mode='bilinear', align_corners=True)

        # 融合
        gau_out=gau1s_out+gau1s_ew_out+gau2s_out_ip+gau4s_out_ip
        # gau_out=gau1s_out+gau2s_out_ip+gau4s_out_ip
        # print(gau_out.shape) torch.Size([1, 256, 28, 28])
        resconv_out=self.resconv1(gau_out)
        resconv_out=self.resconv2(resconv_out)
        resconv_out=self.resconv3(resconv_out)
        # print(resconv_out.shape) torch.Size([1, 256, 28, 28])
        aspp_out = self.aspp(resconv_out) 
        # print(aspp_out.shape) torch.Size([1, 256, 28, 28])
        out = self.cls_conv(aspp_out)

        out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)

        return out,[attn_1s_ew,attn_1s,attn_2s,attn_4s]


# 导入其他模块时候不会运行
if __name__ == '__main__':
    model = PGNet()
    import time
    print(time.time())
    x = torch.randn(1,3,400,400)
    mask = torch.randn(1,1,400,400)
    y = model(x,mask,x)
    print(time.time())