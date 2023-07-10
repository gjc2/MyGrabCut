#pragma once

#include <iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/detail/gcgraph.hpp>
#include<limits>
#include<opencv2/imgproc.hpp>
#include<opencv2/core/types_c.h>
#include<opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;
using namespace detail;

void MygrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode);

class MyGMM {
public:
	static const int componentsCount = 5;
	double operator()(const Vec3d color) const;//混合gauss概率
	double operator()(int ci, const Vec3d color)const;//某个gauss分量概率
	int whichComponent(const Vec3d color)const;//pixel属于哪个分量概率最大
	void initLearning();
	void addSample(int ci, const Vec3d color);
	void endLearning();
	void calInverseCovAndDeterm(int ci);


	MyGMM(Mat& m);
	void test() { cout << coefs[0]<<" "<<coefs[1]<<" "<<coefs[2]<<" "<<coefs[3]<<" "<<coefs[4]<<endl; }
private:
	double coefs[componentsCount];//gauss分量的系数
	double mean[componentsCount][3];//均值
	double cov[componentsCount][3][3];//协方差矩阵

	double inverseCovs[componentsCount][3][3];//协方差矩阵的逆矩阵
	double covDeterms[componentsCount];//协方差矩阵的行列式

	double sums[componentsCount][3];
	double prods[componentsCount][3][3];
	int sampleCounts[componentsCount];
	int totalSampleCount;
};