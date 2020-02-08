package datasets

import "github.com/gokadin/ml-framework/mat"

type set struct {
	data *mat.Mat32f
}

func (s *set) Data() *mat.Mat32f {
	return s.data
}

func (s *set) Normalize(min, max float32) *set {
	normalize(s.data, min, max)

	return s
}

func (s *set) OneHot(depth int) *set {
	oneHotEncode(s.data, depth)

	return s
}
