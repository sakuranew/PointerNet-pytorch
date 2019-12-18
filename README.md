# PointerNet-pytorch
Pointer Network pytorch version,decoder use boundary model

Can be added to bert as output layer
Current version support gru only
## Test
```python
from PointerNet import PtrNet
import torch
x=torch.randn(2,5,256)
m=PtrNet(256,128,bi=True)
print(m(x))
```
### Output: 
    
    tensor([[[0.2175, 0.2090, 0.2347, 0.1744, 0.1645],
             [0.2174, 0.2089, 0.2356, 0.1731, 0.1650]],
            [[0.2420, 0.2518, 0.1624, 0.1893, 0.1545],
             [0.2407, 0.2503, 0.1661, 0.1893, 0.1537]]], grad_fn=<StackBackward>)
## With Bert

```python
from PointerNet import PtrNet
import torch
from pytorch_transformers import BertModel
x=torch.range(1,5).unsqueeze(0).long()
bert=BertModel.from_pretrained('base model path')
embd,cls=bert(x)
m=PtrNet(768,128,bi=True)
print(m(embd))
```
