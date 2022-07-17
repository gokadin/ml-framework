package mat

import (
	"fmt"
	"strconv"
	"strings"
)

type Shape struct {
	D []int
}

func Dim(dimensions ...int) Shape {
	for _, d := range dimensions {
		if d <= 0 {
			panic("cannot have a negative or size zero dimension")
		}
	}
	return Shape{dimensions}
}

func (s Shape) Size() int {
	result := 1
	for _, d := range s.D {
		result *= d
	}
	return result
}

func (s Shape) Count() int {
	return len(s.D)
}

func (s Shape) Equals(other Shape) bool {
	if len(other.D) != len(s.D) {
		return false
	}

	for i := 0; i < len(s.D); i++ {
		if s.D[i] != other.D[i] {
			return false
		}
	}

	return true
}

// IndexOf
//	for 2 dimensions: index = d[0].index * d[1].size + d[1].index
//	for 3 dimensions: index = (d[0].index * d[1].size * d[2].size) + (d[1].index * d[2].size) + d[2].index
///**
func (s Shape) IndexOf(values ...int) (index int) {
	if len(values) == 0 || len(values) > s.Count() {
		panic("invalid argument for IndexOf operation on shape")
	}

	if len(values) < s.Count() {
		for i := 0; i < s.Count()-len(values)+1; i++ {
			values = append(values, 0)
		}
	}

	mul := 1
	for i := s.Count() - 1; i >= 0; i-- {
		if i == s.Count()-1 {
			index += values[i]
		} else {
			mul *= s.D[i+1]
			index += values[i] * mul
		}
	}

	return index
}

func (s Shape) Copy() Shape {
	dimensions := make([]int, s.Count())
	for i := 0; i < len(dimensions); i++ {
		dimensions[i] = s.D[i]
	}
	return Dim(dimensions...)
}

func (s Shape) Expand(dimension int) Shape {
	s.D = append([]int{dimension}, s.D...)
	return s
}

func AddShapes(a, b Shape) Shape {
	if a.Count() != b.Count() {
		panic("cannot add shapes of different dimensions")
	}

	result := a.Copy()
	for i := 0; i < a.Count(); i++ {
		result.D[i] = a.D[i] + b.D[i]
	}

	return result
}

func (s Shape) Print() string {
	str := make([]string, len(s.D))
	for i, d := range s.D {
		str[i] = strconv.Itoa(d)
	}
	return fmt.Sprintf("(%s)", strings.Join(str, ","))
}
