package tensor

import "testing"

func Test_NewRandomTensor_size(t *testing.T) {
    tensor := NewRandomTensor(2, 3)

    if len(tensor.mat) != 2 && len(tensor.mat[0]) != 2 {
        t.Fatalf("tensor size is incorrect")
	}
}
