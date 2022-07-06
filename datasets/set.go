package datasets

import "ml-framework/mat"

type set struct {
	data *mat.M32f
}

func (s *set) Data() *mat.M32f {
	return s.data
}

func (s *set) Normalize(initialMin, initialMax, min, max float32) *set {
	normalize(s.data, initialMin, initialMax, min, max)

	return s
}

func (s *set) OneHot(depth int) *set {
	s.data = oneHotEncode(s.data, depth)

	return s
}
