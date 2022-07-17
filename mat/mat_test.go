package mat

import (
	"github.com/stretchr/testify/assert"
	"testing"
)

func Test_mat_sliceOneDim(t *testing.T) {
	mat := FromSlice32f(Dim(10), []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})

	result := mat.Slice(Dim(1), Dim(3))

	expected := FromSlice32f(Dim(3), []float32{2, 3, 4})
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_mat_sliceTwoDims(t *testing.T) {
	mat := FromSlice32f(Dim(3, 5), []float32{
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
		30, 31, 32, 33, 34,
	})

	result := mat.Slice(Dim(1, 1), Dim(2, 3))

	expected := FromSlice32f(Dim(3), []float32{
		21, 22, 23,
		31, 32, 33,
	})
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_mat_sliceTwoDimsWhenOverNumberOfElements(t *testing.T) {
	mat := FromSlice32f(Dim(3, 5), []float32{
		10, 11, 12, 13, 14,
		20, 21, 22, 23, 24,
		30, 31, 32, 33, 34,
	})

	result := mat.Slice(Dim(1, 1), Dim(10, 3))

	expected := FromSlice32f(Dim(3), []float32{
		21, 22, 23,
		31, 32, 33,
	})
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_mat_sliceThreeDims(t *testing.T) {
	mat := FromSlice32f(Dim(3, 3, 5), []float32{
		110, 111, 112, 113, 114,
		120, 121, 122, 123, 124,
		130, 131, 132, 133, 134,

		210, 211, 212, 213, 214,
		220, 221, 222, 223, 224,
		230, 231, 232, 233, 234,

		310, 311, 312, 313, 314,
		320, 321, 322, 323, 324,
		330, 331, 332, 333, 334,
	})

	result := mat.Slice(Dim(1, 1, 1), Dim(2, 2, 3))

	expected := FromSlice32f(Dim(3), []float32{
		221, 222, 223,
		231, 232, 233,

		321, 322, 323,
		331, 332, 333,
	})
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_mat_sliceThreeDimsWithMissingDimension(t *testing.T) {
	mat := FromSlice32f(Dim(3, 3, 5), []float32{
		110, 111, 112, 113, 114,
		120, 121, 122, 123, 124,
		130, 131, 132, 133, 134,

		210, 211, 212, 213, 214,
		220, 221, 222, 223, 224,
		230, 231, 232, 233, 234,

		310, 311, 312, 313, 314,
		320, 321, 322, 323, 324,
		330, 331, 332, 333, 334,
	})

	result := mat.Slice(Dim(1, 1), Dim(2, 2))

	expected := FromSlice32f(Dim(3), []float32{
		220, 221, 222, 223, 224,
		230, 231, 232, 233, 234,

		320, 321, 322, 323, 324,
		330, 331, 332, 333, 334,
	})
	assert.Equal(t, expected.Data(), result.Data())
}

func Test_mat_sliceThreeDimsWithTwoMissingDimensions(t *testing.T) {
	mat := FromSlice32f(Dim(3, 3, 5), []float32{
		110, 111, 112, 113, 114,
		120, 121, 122, 123, 124,
		130, 131, 132, 133, 134,

		210, 211, 212, 213, 214,
		220, 221, 222, 223, 224,
		230, 231, 232, 233, 234,

		310, 311, 312, 313, 314,
		320, 321, 322, 323, 324,
		330, 331, 332, 333, 334,
	})

	result := mat.Slice(Dim(1), Dim(1))

	expected := FromSlice32f(Dim(3), []float32{
		210, 211, 212, 213, 214,
		220, 221, 222, 223, 224,
		230, 231, 232, 233, 234,
	})
	assert.Equal(t, expected.Data(), result.Data())
}

//func Test_mat_slice_multiDimension(t *testing.T) {
//	mat := FromSlice32f(WithShape(4, 2), []float32{1, 2, 3, 4, 5, 6, 7, 8})
//
//	result := mat.Slice(2, 4)
//
//	expected := FromSlice32f(WithShape(2, 2), []float32{5, 6, 7, 8})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_slice_multiDimensionDifferent(t *testing.T) {
//	mat := FromSlice32f(WithShape(4, 5), []float32{1, 2, 3, 4, 5,
//		6, 7, 8, 9, 10,
//		11, 12, 13, 14, 15,
//		16, 17, 18, 19, 20})
//
//	result := mat.Slice(1, 3)
//
//	expected := FromSlice32f(WithShape(2, 5), []float32{6, 7, 8, 9, 10,
//		11, 12, 13, 14, 15})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_add_singleDimension(t *testing.T) {
//	m1 := FromSlice32f(WithShape(1, 3), []float32{1, 2, 3})
//	m2 := FromSlice32f(WithShape(1, 3), []float32{1, 2, 3})
//
//	result := Add(m1, m2)
//
//	expected := FromSlice32f(WithShape(1, 3), []float32{2, 4, 6})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_add_singleDimensionInPlace(t *testing.T) {
//	m1 := FromSlice32f(WithShape(1, 3), []float32{1, 2, 3})
//	m2 := FromSlice32f(WithShape(1, 3), []float32{1, 2, 3})
//
//	m1.Add(m2)
//
//	expected := FromSlice32f(WithShape(1, 3), []float32{2, 4, 6})
//	assert.True(t, expected.Equals32f(m1))
//}
//
//func Test_mat_add_multiDimension(t *testing.T) {
//	m1 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//	m2 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//
//	result := Add(m1, m2)
//
//	expected := FromSlice32f(WithShape(2, 1), []float32{2, 4})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_addScalar_multiDimension(t *testing.T) {
//	m1 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//
//	result := AddScalar(m1, 4)
//
//	expected := FromSlice32f(WithShape(2, 1), []float32{5, 6})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_sub_multiDimension(t *testing.T) {
//	m1 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//	m2 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//
//	result := Sub(m1, m2)
//
//	expected := FromSlice32f(WithShape(2, 1), []float32{0, 0})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_div_multiDimension(t *testing.T) {
//	m1 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//	m2 := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//
//	result := Div(m1, m2)
//
//	expected := FromSlice32f(WithShape(2, 1), []float32{1, 1})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_transpose(t *testing.T) {
//	mat := FromSlice32f(WithShape(3, 1), []float32{1, 2, 3})
//
//	result := Transpose(mat)
//
//	expected := FromSlice32f(WithShape(1, 3), []float32{1, 2, 3})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_transpose_multiDimension(t *testing.T) {
//	mat := FromSlice32f(WithShape(3, 2), []float32{1, 2, 3, 4, 5, 6})
//
//	result := Transpose(mat)
//
//	expected := FromSlice32f(WithShape(2, 3), []float32{1, 3, 5, 2, 4, 6})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_dot_multiDimension(t *testing.T) {
//	m1 := FromSlice32f(WithShape(2, 2), []float32{1, 0, 0, 1})
//	m2 := FromSlice32f(WithShape(2, 2), []float32{4, 1, 2, 2})
//
//	result := MatMul(m1, m2)
//
//	expected := FromSlice32f(WithShape(2, 2), []float32{4, 1, 2, 2})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_dot_multiDimensionOfDifferentSizes(t *testing.T) {
//	m1 := FromSlice32f(WithShape(2, 3), []float32{1, 0, 3, 0, 1, 2})
//	m2 := FromSlice32f(WithShape(3, 2), []float32{4, 1, 2, 2, 1, 5})
//
//	result := MatMul(m1, m2)
//
//	expected := FromSlice32f(WithShape(2, 2), []float32{7, 16, 4, 12})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_sum0_multiDimension(t *testing.T) {
//	mat := FromSlice32f(WithShape(2, 3), []float32{1, 0, 3, 0, 1, 2})
//
//	result := Sum(mat, 0)
//
//	expected := FromSlice32f(WithShape(1, 3), []float32{1, 1, 5})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_sum1_multiDimension(t *testing.T) {
//	mat := FromSlice32f(WithShape(2, 3), []float32{1, 0, 3, 0, 1, 2})
//
//	result := Sum(mat, 1)
//
//	expected := FromSlice32f(WithShape(2, 1), []float32{4, 3})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_expand0_multiDimension(t *testing.T) {
//	mat := FromSlice32f(WithShape(1, 3), []float32{1, 0, 3})
//
//	result := Expand(mat, 0, 3)
//
//	expected := FromSlice32f(WithShape(3, 3), []float32{1, 0, 3, 1, 0, 3, 1, 0, 3})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_expand0_multiDimensionDifferent(t *testing.T) {
//	mat := FromSlice32f(WithShape(1, 2), []float32{1, 0})
//
//	result := Expand(mat, 0, 4)
//
//	expected := FromSlice32f(WithShape(4, 2), []float32{1, 0, 1, 0, 1, 0, 1, 0})
//	assert.True(t, expected.Equals32f(result))
//}
//
//func Test_mat_expand1_multiDimension(t *testing.T) {
//	mat := FromSlice32f(WithShape(2, 1), []float32{1, 2})
//
//	result := Expand(mat, 1, 3)
//
//	expected := FromSlice32f(WithShape(2, 3), []float32{1, 1, 1, 2, 2, 2})
//	assert.True(t, expected.Equals32f(result))
//}
