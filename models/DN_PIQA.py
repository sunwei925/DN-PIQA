import torch
import torch.nn as nn
import timm

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x



class Swin_b_384_in22k(torch.nn.Module):
    def __init__(self):
        
        super(Swin_b_384_in22k, self).__init__()
        swin_b = timm.create_model('swin_base_patch4_window12_384_in22k', pretrained=False)
        swin_b.head = Identity()

        self.feature_extraction = swin_b
        self.quality = self.quality_regression(1024+256, 128,1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):
        
        x = self.feature_extraction(x)

        x = self.quality(x)
            
        return x


class Swin_b_384_in22k_pretrained(torch.nn.Module):
    def __init__(self, pretrained_path):
        
        super(Swin_b_384_in22k_pretrained, self).__init__()
        swin_b = Swin_b_384_in22k()
        if pretrained_path!=None:
            swin_b.load_state_dict(torch.load(pretrained_path))
        swin_b.quality = Identity()

        self.feature_extraction = swin_b
        self.quality = self.quality_regression(1024, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x):
        x_size = x.shape
        # x = x.unsqueeze(1)
        x = self.feature_extraction(x)

        x = self.quality(x)
            
        return x













class PIQ_model(torch.nn.Module):
    def __init__(self, pretrained_path, pretrained_path_face):
        
        super(PIQ_model, self).__init__()
        # Overall
        swin_b = Swin_b_384_in22k()
        if pretrained_path!=None:
            print('load overall model')
            swin_b.load_state_dict(torch.load(pretrained_path))
        swin_b.quality = Identity()

        self.feature_extraction = swin_b

        # face
        swin_b_face = Swin_b_384_in22k_pretrained(None)
        if pretrained_path_face != None:
            print('load face model')
            swin_b_face.load_state_dict(torch.load(pretrained_path_face))
        swin_b_face.quality = Identity()

        self.feature_extraction_face = swin_b_face

        self.quality = self.quality_regression(1024+1024+495, 128, 1)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.ReLU(),
            nn.Dropout(0.1),            
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block

    def forward(self, x, x_face, x_feature):

        x = self.feature_extraction(x)
        x_face = self.feature_extraction_face(x_face)

        x = torch.cat((x,x_face,x_feature), dim = 1)

        x = self.quality(x)
            
        return x



