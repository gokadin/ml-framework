package tensor

type operation struct {
    tensor *Tensor
    operationName string
    creators []*Tensor
    metadata float64
}

func newOperation(tensor *Tensor, operationName string, creators []*Tensor, metadata float64) *operation {
    return &operation{
        tensor: tensor,
        operationName: operationName,
        creators: creators,
        metadata: metadata,
    }
}

