import torch
import torch.nn as nn
import torch.nn.functional as F
device='cuda' if torch.cuda.is_available() else None
class Encoder(nn.Module):
    """
    Encoder for Pointer Net
    """
    def __init__(self,d_input=128,d_hidden=128,num_layer=1,bi=False,dropout=0,name='gru'):
        """
        :param d_input: input size
        :param d_hidden: hidden size
        :param num_layer: num layers,Default: 1
        :param bi: If True, becomes a bidirectional rnn. Default: False
        :param dropout: dropout rate. Default: 0
        :param name: rnn. Default: 'gru'
        """
        super().__init__()
        self.bi=bi
        if bi:d_hidden=d_hidden//2
        model=nn.GRU if name in 'gru' else nn.LSTM
        self.rnn = model(input_size=d_input, hidden_size=d_hidden, num_layers=num_layer, batch_first=True,
                          bidirectional=bi, dropout=dropout)
        # if name in 'gru':
        #     self.rnn=nn.GRU(input_size=d_input,hidden_size=d_hidden,num_layers=num_layer,batch_first=True,bidirectional=bi,dropout=dropout)
        # elif name in 'lstm':
        #     self.rnn=nn.LSTM(input_size=d_input,hidden_size=d_hidden,num_layers=num_layer,batch_first=True,bidirectional=bi,dropout=dropout)
    def forward(self,input):
        """

        :param input:   [N,L,H_in]
        :return:output: [N,L,H_hidden],(h: [N,H_hidden],) or (h: [N,H_hidden],c: [N,H_hidden],)
        """
        output,state=self.rnn(input)
        state=state.view(output.shape[0],-1)
        return output,state
class PtrAttScore(nn.Module):
    """
    get attention score for ptr net
    """
    def __init__(self,d_q=128,d_hidden=128,is_q=False,is_k=False,bias=False,att_name='sum'):
        """
        q,k from <<attention is all you need>>
        :param d_q:     q size
        :param d_hidden:hidden size
        :param is_q:    transform q or not
        :param is_k:    transform k or not
        :param att_name:attention name
        """
        super().__init__()
        self.att_name=att_name
        self.wq=nn.Linear(d_q,d_hidden,bias=bias) if not is_q else None
        self.wk=nn.Linear(d_hidden,d_hidden,bias=bias) if not is_k else None
        self.w_score=nn.Linear(d_hidden,1,bias=bias)
    def att(self,q,k,mask):
        if self.att_name=='sum':
            q = q.unsqueeze(1).expand_as(k)
            t=torch.tanh(q+k)
            score=self.w_score(t).squeeze(-1)
        if mask is not None:
            mask = mask.bool()
            score = score.masked_fill(~mask, -float('inf'))
        score=torch.softmax(score,-1)
        return score
    def forward(self,q,k,mask=None):
        """

        :param q:       [N,H_in]
        :param k:       [N,L,H_hidden]
        :param mask:       [N,L]
        :return: score: [N,L]
        """
        if self.wq:q=self.wq(q)         #[N,H_hidden]
        if self.wk:k=self.wk(k)         #[N,L,H_hidden]
        score=self.att(q,k,mask)
        return score
class Decoder(nn.Module):
    """
    Decoder for Pointer Net
    """
    def __init__(self,d_input=128,d_hidden=128,step=2,dropout=0,name='gru'):
        """

        :param d_input: input size
        :param d_hidden: hidden size
        :param step: num step,Default: 2
        :param dropout: dropout rate. Default: 0
        :param name: rnn. Default: 'gru'
        """
        super().__init__()
        self.step=step
        model=nn.GRUCell if name in 'gru' else nn.LSTMCell
        self.rnn = model(input_size=d_input, hidden_size=d_hidden)
        self.att=PtrAttScore(d_hidden,d_hidden)
        self.dp=nn.Dropout(dropout)
    def forward(self,encoder_input,encoder_output,state,mask=None,d0=None,split=False):
        """

        :param encoder_input:   [N,L,H_in]
        :param encoder_output:  [N,L,H_hidden]
        :param state:           [N,H_hidden]
        :param d0:           [N,H_in]
        :param split:           split output at dim 1 or not
        :return: output         [N,step,L]
        """
        bs,d_input=encoder_input.shape[0],encoder_input.shape[2]
        if d0 is None:d0=torch.ones((bs,d_input),device=encoder_input.device)

        score=[]
        decoder_input=d0
        for i  in range(self.step):
            state=self.rnn(decoder_input,state)
            cur_score=self.att(state,encoder_output,mask)
            indice=cur_score.argmax(1)
            indice=indice.view(-1,1,1).expand(-1,-1,d_input)
            decoder_input=encoder_input.gather(1,indice).squeeze(1)
            score.append(cur_score)
        if not split:
            score=torch.stack(score,1)
        return score

class PtrNet(nn.Module):
    """
    Pointer Net
    """
    def __init__(self,d_input=128,d_hidden=128,num_layer=1,bi=False,dropout=0,bias=True,name='gru'):
        """
        :param d_input: input size
        :param d_hidden: hidden size
        :param num_layer: num layers,Default: 1
        :param bi: If True, becomes a bidirectional rnn. Default: False
        :param dropout: dropout rate. Default: 0
        :param bias: bias. Default: True
        :param name: rnn. Default: 'gru'
        """
        super().__init__()
        self.encoder=Encoder(d_input,d_hidden,num_layer,bi,dropout,name)
        self.decoder=Decoder(d_input,d_hidden,step=2,dropout=dropout,name=name)
        self.trans_state=nn.Linear(d_hidden,d_hidden,bias=bias)
    def forward(self,encoder_input,mask=None,d0=None,split=False):
        """

        :param encoder_input:   [N,L,H_in]
        :param d0:           [N,H_in]
        :return: output         [N,step,L]
        """
        encoder_output,hidden_state=self.encoder(encoder_input)
        init_state=self.trans_state(hidden_state).squeeze(-1)
        output=self.decoder(encoder_input,encoder_output,state=init_state,mask=mask,d0=d0,split=split)
        return output
