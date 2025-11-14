import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SiameseUNetPP(nn.Module):
   
    def __init__(self, n_channels=3, n_classes=1, base_ch=64):
        super().__init__()
        self.base_ch = base_ch

        # ---------- Encoder (shared weights) ----------
        self.enc_conv0 = DoubleConv(n_channels, base_ch)
        self.enc_conv1 = DoubleConv(base_ch, base_ch * 2)
        self.enc_conv2 = DoubleConv(base_ch * 2, base_ch * 4)
        self.enc_conv3 = DoubleConv(base_ch * 4, base_ch * 8)
        self.enc_conv4 = DoubleConv(base_ch * 8, base_ch * 16)
        self.pool = nn.MaxPool2d(2)

        # ---------- Decoder (nested dense connections) ----------
        # channels[i] = base_ch * (2 ** i)
        ch = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]

        # For Siamese fusion, encoder outputs are doubled in channels
        # Define decoder convolutions following full U-Net++ dense pattern
        self.dconv3_0 = DoubleConv(ch[3]*2 + ch[4]*2, ch[3])               # X_3,0
        self.dconv2_0 = DoubleConv(ch[2]*2 + ch[3], ch[2])                 # X_2,0
        self.dconv2_1 = DoubleConv(ch[2] + ch[3], ch[2])                   # X_2,1
        self.dconv1_0 = DoubleConv(ch[1]*2 + ch[2], ch[1])                 # X_1,0
        self.dconv1_1 = DoubleConv(ch[1] + ch[2], ch[1])                   # X_1,1
        self.dconv1_2 = DoubleConv(ch[1] + ch[2], ch[1])                   # X_1,2
        self.dconv0_0 = DoubleConv(ch[0]*2 + ch[1], ch[0])                 # X_0,0
        self.dconv0_1 = DoubleConv(ch[0] + ch[1], ch[0])                   # X_0,1
        self.dconv0_2 = DoubleConv(ch[0] + ch[1], ch[0])                   # X_0,2
        self.dconv0_3 = DoubleConv(ch[0] + ch[1], ch[0])                   # X_0,3

        # ---------- Output layers (deep supervision) ----------
        self.out_convs = nn.ModuleList([
            nn.Conv2d(ch[0], n_classes, kernel_size=1),
            nn.Conv2d(ch[0], n_classes, kernel_size=1),
            nn.Conv2d(ch[0], n_classes, kernel_size=1),
            nn.Conv2d(ch[0], n_classes, kernel_size=1)
        ])

    # Utility upsampling
    def _upsample(self, x, ref):
        """Upsample x to match spatial size of ref using bilinear interpolation"""
        return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)

    def forward(self, x1, x2, return_deep=False):
        # ---------- Encoder (Siamese shared weights) ----------
        e0_1 = self.enc_conv0(x1)
        e1_1 = self.enc_conv1(self.pool(e0_1))
        e2_1 = self.enc_conv2(self.pool(e1_1))
        e3_1 = self.enc_conv3(self.pool(e2_1))
        e4_1 = self.enc_conv4(self.pool(e3_1))

        e0_2 = self.enc_conv0(x2)
        e1_2 = self.enc_conv1(self.pool(e0_2))
        e2_2 = self.enc_conv2(self.pool(e1_2))
        e3_2 = self.enc_conv3(self.pool(e2_2))
        e4_2 = self.enc_conv4(self.pool(e3_2))

        # Siamese fusion (concat pre & post)
        e0 = torch.cat([e0_1, e0_2], dim=1)
        e1 = torch.cat([e1_1, e1_2], dim=1)
        e2 = torch.cat([e2_1, e2_2], dim=1)
        e3 = torch.cat([e3_1, e3_2], dim=1)
        e4 = torch.cat([e4_1, e4_2], dim=1)

        # ---------- Decoder (full dense path) ----------
        # Level 3
        X_3_0 = self.dconv3_0(torch.cat([e3, self._upsample(e4, e3)], dim=1))

        # Level 2
        X_2_0 = self.dconv2_0(torch.cat([e2, self._upsample(X_3_0, e2)], dim=1))
        X_2_1 = self.dconv2_1(torch.cat([X_2_0, self._upsample(X_3_0, X_2_0)], dim=1))

        # Level 1
        X_1_0 = self.dconv1_0(torch.cat([e1, self._upsample(X_2_0, e1)], dim=1))
        X_1_1 = self.dconv1_1(torch.cat([X_1_0, self._upsample(X_2_1, X_1_0)], dim=1))
        X_1_2 = self.dconv1_2(torch.cat([X_1_1, self._upsample(X_2_1, X_1_1)], dim=1))

        # Level 0
        X_0_0 = self.dconv0_0(torch.cat([e0, self._upsample(X_1_0, e0)], dim=1))
        X_0_1 = self.dconv0_1(torch.cat([X_0_0, self._upsample(X_1_1, X_0_0)], dim=1))
        X_0_2 = self.dconv0_2(torch.cat([X_0_1, self._upsample(X_1_2, X_0_1)], dim=1))
        X_0_3 = self.dconv0_3(torch.cat([X_0_2, self._upsample(X_1_2, X_0_2)], dim=1))

        # ---------- Deep supervision outputs ----------
        out0 = self.out_convs[0](X_0_0)
        out1 = self.out_convs[1](X_0_1)
        out2 = self.out_convs[2](X_0_2)
        out3 = self.out_convs[3](X_0_3)

        deep_outs = [out0, out1, out2, out3]

        if return_deep:
            return deep_outs
        else:
            # Average the deep supervision maps
            return torch.sigmoid(torch.mean(torch.stack(deep_outs), dim=0))


# # # ---------- Example usage ----------
# if __name__ == "__main__":
#     model = SiameseUNetPP(n_channels=3, n_classes=1, base_ch=64)
#     x1 = torch.randn(1, 3, 256, 256)
#     x2 = torch.randn(1, 3, 256, 256)
#     y = model(x1, x2)
#     print("Output shape:", y.shape)
