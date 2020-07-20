import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modeling.discriminator.SubLayers import MultiHeadAttention, PositionwiseFeedForward

class D_Net(nn.Module):

	def __init__(self, channel, d_model, d_inner, n_head, d_k, d_v, dropout):
		super(D_Net, self).__init__()
		self.d_model = d_model
		# self.first = nn.Linear(1,d_model)
		# self.MaxPool = nn.MaxPool1d(d_model)
		self.Answer = nn.Linear(d_model,1)
		self.slf_attn_1 = MultiHeadAttention(
			n_head, d_model, d_k, d_v, dropout=dropout)
		self.pos_ffn_1 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

		# self.slf_attn_2 = MultiHeadAttention(
		# 	n_head, d_model, d_k, d_v, dropout=dropout)
		# self.pos_ffn_2 = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

	def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
		enc_input = enc_input.unsqueeze(1)
		# enc_input = self.first(enc_input)
		enc_output, enc_slf_attn = self.slf_attn_1(
			enc_input, enc_input, enc_input, mask=slf_attn_mask)
		# enc_output *= non_pad_mask

		enc_output = self.pos_ffn_1(enc_output)
		# # enc_output *= non_pad_mask
		# enc_output, enc_slf_attn = self.slf_attn_2(
		# 	enc_output, enc_output, enc_output, mask=slf_attn_mask)
		# enc_output *= non_pad_mask

		# enc_output = self.pos_ffn_2(enc_output)
		# enc_output *= non_pad_mask
		# out = self.MaxPool(enc_output).squeeze(1)
		# print(enc_output.size())
		out = self.Answer(enc_output)
		# print(out.size())
		out = out.view(-1)
		out = F.sigmoid(out)
		return out

if __name__ == '__main__':
	CHANALS_MAP = [512, 1024, 512, 256, 256, 256]
	net = D_Net(CHANALS_MAP[0],d_model=64,d_inner=64,n_head=4,d_k=32,d_v=32,dropout=0.2)
	x = torch.rand(15,1,512)
	out = net(x)
	# print(att.size())
	print(out.size())
	print(out)
