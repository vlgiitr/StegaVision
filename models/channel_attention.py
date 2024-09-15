import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# Channel attention block
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class CBAM_ChannelOnly(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM_ChannelOnly, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out
    
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        # Prep Network
        self.conv1 = nn.Conv2d(3, 50, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, 4, padding=1)
        self.conv3 = nn.Conv2d(3, 5, 5, padding=2)
        
        self.conv4 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv6 = nn.Conv2d(65, 5, 5, padding=2)
        
        # Hiding Network
        self.conv7 = nn.Conv2d(68, 50, 3, padding=1)
        self.conv8 = nn.Conv2d(68, 10, 4, padding=1)
        self.conv9 = nn.Conv2d(68, 5, 5, padding=2)
        
        self.conv10 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv12 = nn.Conv2d(65, 5, 5, padding=2)
        
        self.conv13 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv15 = nn.Conv2d(65, 5, 5, padding=2)
        
        self.conv16 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv17 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv18 = nn.Conv2d(65, 5, 5, padding=2)
        
        self.conv19 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv20 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv21 = nn.Conv2d(65, 5, 5, padding=2)
        
        self.conv22 = nn.Conv2d(65, 3, 3, padding=1)
        
        # Channel Gate Modules
        self.channel_attention1 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention2 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention3 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention4 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention5 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention6 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention7 = CBAM_ChannelOnly(gate_channels=65)
        
    def forward(self, input_S, input_C):
        # Prep Network
        x1 = F.relu(self.conv1(input_S))
        x2 = F.relu(self.conv2(input_S))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv3(input_S))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention1(x4)
        
        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention2(x4)
        
        x4 = torch.cat((input_C, x4), 1)
        
        # Hiding Network
        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention3(x4)
        
        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention4(x4)
        
        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention5(x4)
        
        x1 = F.relu(self.conv16(x4))
        x2 = F.relu(self.conv17(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv18(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention6(x4)
        
        x1 = F.relu(self.conv19(x4))
        x2 = F.relu(self.conv20(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv21(x4))
        x4 = torch.cat((x1, x2, x3), 1)
        
        # Apply channel attention
        x4 = self.channel_attention7(x4)
        
        # Output layer
        output = torch.tanh(self.conv22(x4))
        
        return output
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 50, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 10, 4, padding=1)
        self.conv3 = nn.Conv2d(3, 5, 5, padding=2)

        self.conv4 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv5 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv6 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv7 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv8 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv9 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv10 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv11 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv12 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv13 = nn.Conv2d(65, 50, 3, padding=1)
        self.conv14 = nn.Conv2d(65, 10, 4, padding=1)
        self.conv15 = nn.Conv2d(65, 5, 5, padding=2)

        self.conv16 = nn.Conv2d(65, 3, 3, padding=1)

        # Channel Gate Modules
        self.channel_attention1 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention2 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention3 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention4 = CBAM_ChannelOnly(gate_channels=65)
        self.channel_attention5 = CBAM_ChannelOnly(gate_channels=65)


    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv3(x))
        x4 = torch.cat((x1, x2, x3), 1)

        # Apply channel attention
        x4 = self.channel_attention1(x4)

        x1 = F.relu(self.conv4(x4))
        x2 = F.relu(self.conv5(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv6(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        # Apply channel attention
        x4 = self.channel_attention2(x4)

        x1 = F.relu(self.conv7(x4))
        x2 = F.relu(self.conv8(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv9(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        # Apply channel attention
        x4 = self.channel_attention3(x4)

        x1 = F.relu(self.conv10(x4))
        x2 = F.relu(self.conv11(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv12(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        # Apply channel attention
        x4 = self.channel_attention4(x4)

        x1 = F.relu(self.conv13(x4))
        x2 = F.relu(self.conv14(x4))
        x2 = F.pad(x2, (0, 1, 0, 1), 'constant', value=0)
        x3 = F.relu(self.conv15(x4))
        x4 = torch.cat((x1, x2, x3), 1)

        # Apply channel attention
        x4 = self.channel_attention5(x4)

        output = torch.tanh(self.conv16(x4))

        return output

class Make_model(nn.Module):
    def __init__(self):
        super(Make_model,self).__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self,input_S,input_C):
        output_Cprime = self.encoder(input_S,input_C)
        output_Sprime = self.decoder(output_Cprime)
        
        return output_Cprime,output_Sprime

#if want to run this model independently
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Make_model()
    model.to(device)
    original_model = model