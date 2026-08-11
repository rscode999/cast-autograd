MAIN='main.cpp' #Put your path to a file containing `main` here

OUTPUT_EXECUTABLE_NAME='a' #put your desired executable name here

XTENSOR_PATH='lib/xtensor/include/'
XTL_PATH='lib/xtl/include/'
XTENSOR_BLAS_PATH='lib/xtensor-blas/include'

c:
	g++ ${MAIN}  -o ${OUTPUT_EXECUTABLE_NAME} --std=c++20  -I ${XTENSOR_PATH} -I ${XTL_PATH} -I ${XTENSOR_BLAS_PATH}