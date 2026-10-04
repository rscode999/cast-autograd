TESTS='tests.cpp'
ADHOC='tests_adhoc.cpp'

OUTPUT_EXECUTABLE_NAME='tests' #put your desired executable name here

XTENSOR_PATH='../lib/xtensor/include/'
XTL_PATH='../lib/xtl/include/'
XTENSOR_BLAS_PATH='../lib/xtensor-blas/include'

t:
	g++ ${TESTS}  -o ${OUTPUT_EXECUTABLE_NAME} --std=c++20  -I ${XTENSOR_PATH} -I ${XTL_PATH} -I ${XTENSOR_BLAS_PATH}

adhoc:
	g++ ${ADHOC}  -o ${OUTPUT_EXECUTABLE_NAME} --std=c++20  -I ${XTENSOR_PATH} -I ${XTL_PATH} -I ${XTENSOR_BLAS_PATH}