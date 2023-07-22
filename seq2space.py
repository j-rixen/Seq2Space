import torch.nn as nn
import torch


class Seq2Space(nn.Module):
    def __init__(self, inchannels, outchannels, S, Z=1, heads=4, useConv=False, toSeq=False):
        super().__init__()
        '''
        inchannels: channel size of the input
        outchannels: channel size of the output
        S: stride of the first pooling layer, applied on the sequence axis of a and b
        Z: stride of the second pooling layer, applied on the channel axis of a
        heads: number of parallel heads
        useConv: whether or not convolutional or linear layers are used
        toSeq: whether or not the Seq2Space2Seq architecture is used or not; if False, seq axis will be reduced to 1 
        '''
        self.heads = heads
        self.toSeq = toSeq
        self.useConv = useConv

        self.depth = outchannels // self.heads

        if useConv:
            self.wa = nn.Conv1d(inchannels, outchannels, 3, 1, padding='same')
            self.wb = nn.Conv1d(inchannels, outchannels, 3, 1, padding='same')
            self.wc = nn.Conv1d(inchannels, outchannels, 3, 1, padding='same')
        else:
            self.wa = nn.Linear(inchannels, outchannels)
            self.wb = nn.Linear(inchannels, outchannels)
            self.wc = nn.Linear(inchannels, outchannels)

        self.linear_out = nn.Linear(outchannels, outchannels)

        self.softmax = nn.Softmax(-1)
        self.S = S
        self.pool = nn.MaxPool1d(S)
        self.pool2 = nn.MaxPool1d(Z)
        self.Z = Z


    def split_heads(self, x, batch_size, is_A=False):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        if is_A:
            x = torch.reshape(x, (batch_size, -1, self.heads, self.depth//self.Z))
        else:
            x = torch.reshape(x, (batch_size, -1, self.heads, self.depth)) #tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return x.transpose(1, 2)#tf.transpose(x, perm=[0, 2, 1, 3])

    def looped_matmul(self, a, k):
        acols = torch.split(a, 1, -2)
        outl = []

        for i in range(self.depth):
            entry = acols[i] - torch.roll(acols[i], i, -1)
            outl.append(entry)

        out = torch.stack(outl, -3)
        k = k.transpose(-2, -1)
        k = torch.unsqueeze(k, -1)

        out = torch.matmul(out, k)
        out = torch.squeeze(out, -1)
        out = out.transpose(-2, -1)

        return out



    def forward(self, x):
        '''
        :param x: input, shape is [batchsize, seqlen, channelsize]
        :return: output, shape is [batchsize, 1, channelsize] if toSeq False, otherwise shape is [batchsize, seqlen, channelsize]
        '''

        bsz = x.size(0)

        x = x.transpose(-2, -1)

        x_pool = self.pool(x)

        if self.useConv:

            x_a = self.wa(x_pool)
            x_b = self.wb(x_pool)

            x_a = x_a.transpose(-2, -1)
            x_b = x_b.transpose(-2, -1)

        else:
            x = x.transpose(-2, -1)
            x_pool = x_pool.transpose(-2, -1)
            x_a = self.wa(x_pool)
            x_b = self.wb(x_pool)

        x_a = self.pool2(x_a)

        a = self.split_heads(x_a, bsz, True)
        b = self.split_heads(x_b, bsz)

        a = a.transpose(-2, -1)
        a = self.softmax(a)

        x_ab = self.looped_matmul(a, b)
        #x_ab = self.softmax(x_ab)

        if self.toSeq:
            c = self.wc(x)
            if self.useConv:
                c = c.transpose(-2, -1)
            c = self.split_heads(c, bsz)
            c = c.transpose(-2, -1)
            x_ab = x_ab.transpose(-2, -1)
            d = c * x_ab
            d = d.transpose(-2, -1)
        else:
            d = x_ab

        d = d.transpose(1, 2)

        d = torch.split(d, 1, -1)
        d = torch.cat(d, 2)
        d = torch.squeeze(d, -1)

        output = self.linear_out(d)

        return output


