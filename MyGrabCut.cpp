#include "MyGrabCut.h"


MyGMM::MyGMM(Mat& m) {
	
	for (int ci = 0; ci < componentsCount; ci++)
		if (coefs[ci] > 0)
			calInverseCovAndDeterm(ci);
}

double MyGMM::operator()(const Vec3d color) const
{
	double res = 0;
	for (int ci = 0; ci < componentsCount; ci++)
		res += coefs[ci] * (*this)(ci, color);
	return res;
}

double MyGMM::operator()(int ci, const Vec3d color)const
{
	double res = 0;
	if (coefs[ci] > 0) {
		Vec3d n = color;
		n[0] = mean[ci][0] - color[0];
		n[1] = mean[ci][1] - color[1];
		n[2] = mean[ci][2] - color[2];

		double mult = n[0] * (
			n[0] * inverseCovs[ci][0][0] +
			n[1] * inverseCovs[ci][1][0] +
			n[2] * inverseCovs[ci][2][0]
			) +
			n[1] * (
				n[0] * inverseCovs[ci][0][1] +
				n[1] * inverseCovs[ci][1][1] +
				n[2] * inverseCovs[ci][2][1]
				) +
			n[2] * (
				n[0] * inverseCovs[ci][0][2] +
				n[1] * inverseCovs[ci][1][2] +
				n[2] * inverseCovs[ci][2][2]
				);
		res = 1.0f / sqrt(covDeterms[ci]) * exp(-0.5 * mult);
	}
	return res;
}

int MyGMM::whichComponent(const Vec3d color)const
{
	int k = 0;
	double the_max = 0;
	for (int ci = 0; ci < componentsCount; ci++)
	{
		double p = (*this)(ci, color);
		if (p > the_max) {
			k = ci;
			the_max = p;
		}
	}
	return k;
}

void MyGMM::initLearning()
{
	for (int ci = 0; ci < componentsCount; ci++) {
		sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
		prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
		prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
		prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
		sampleCounts[ci] = 0;
	}
	totalSampleCount = 0;
}

void MyGMM::addSample(int ci, const Vec3d color)
{
	sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
	prods[ci][0][0] += color[0] * color[0]; prods[ci][0][1] += color[0] * color[1]; prods[ci][0][2] += color[0] * color[2];
	prods[ci][1][0] += color[1] * color[0]; prods[ci][1][1] += color[1] * color[1]; prods[ci][1][2] += color[1] * color[2];
	prods[ci][2][0] += color[2] * color[0]; prods[ci][2][1] += color[2] * color[1]; prods[ci][2][2] += color[2] * color[2];
	sampleCounts[ci]++;
	totalSampleCount++;
}

void MyGMM::endLearning()
{
	for (int ci = 0; ci < componentsCount; ci++) {
		int n = sampleCounts[ci];
		if (n == 0)
			coefs[ci] = 0;
		else {
			//ci 的系数
			coefs[ci] = (double)n / totalSampleCount;
			//ci 的均值
			mean[ci][0] = sums[ci][0] / n;
			mean[ci][1] = sums[ci][1] / n;
			mean[ci][2] = sums[ci][2] / n;
			//ci 的协方差
			cov[ci][0][0] = prods[ci][0][0] / n - mean[ci][0] * mean[ci][0];
			cov[ci][0][1] = prods[ci][0][1] / n - mean[ci][0] * mean[ci][1];
			cov[ci][0][2] = prods[ci][0][2] / n - mean[ci][0] * mean[ci][2];
			cov[ci][1][0] = prods[ci][1][0] / n - mean[ci][1] * mean[ci][0];
			cov[ci][1][1] = prods[ci][1][1] / n - mean[ci][1] * mean[ci][1];
			cov[ci][1][2] = prods[ci][1][2] / n - mean[ci][1] * mean[ci][2];
			cov[ci][2][0] = prods[ci][2][0] / n - mean[ci][2] * mean[ci][0];
			cov[ci][2][1] = prods[ci][2][1] / n - mean[ci][2] * mean[ci][1];
			cov[ci][2][2] = prods[ci][2][2] / n - mean[ci][2] * mean[ci][2];
		}
		double tt= cov[ci][0][0] * (cov[ci][1][1] * cov[ci][2][2] - cov[ci][1][2] * cov[ci][2][1])
			- cov[ci][0][1] * (cov[ci][1][0] * cov[ci][2][2] - cov[ci][1][2] * cov[ci][2][0])
			+ cov[ci][0][2] * (cov[ci][1][0] * cov[ci][2][1] - cov[ci][1][1] * cov[ci][2][0]);
		if (tt <= std::numeric_limits<double>::epsilon())
		{
			double variance = 0.01;
			//相当于如果行列式小于等于0，（对角线元素）增加白噪声，避免其变
			//为退化（降秩）协方差矩阵（不存在逆矩阵，但后面的计算需要计算逆矩阵）。
			// Adds the white noise to avoid singular covariance matrix.
			cov[ci][0][0] += variance;
			cov[ci][1][1] += variance;
			cov[ci][2][2] += variance;
		}
		calInverseCovAndDeterm(ci);
	}
}

void MyGMM::calInverseCovAndDeterm(int ci)
{
	if (coefs[ci] > 0) {
		covDeterms[ci] =
			cov[ci][0][0] * (cov[ci][1][1] * cov[ci][2][2] - cov[ci][1][2] * cov[ci][2][1])
			- cov[ci][0][1] * (cov[ci][1][0] * cov[ci][2][2] - cov[ci][1][2] * cov[ci][2][0])
			+ cov[ci][0][2] * (cov[ci][1][0] * cov[ci][2][1] - cov[ci][1][1] * cov[ci][2][0]);

		inverseCovs[ci][0][0] = (cov[ci][1][1] * cov[ci][2][2] - cov[ci][2][1] * cov[ci][1][2]) / covDeterms[ci];
		inverseCovs[ci][0][1] = -(cov[ci][0][1] * cov[ci][2][2] - cov[ci][0][2] * cov[ci][2][1]) / covDeterms[ci];
		inverseCovs[ci][0][2] = (cov[ci][0][1] * cov[ci][1][2] - cov[ci][0][2] * cov[ci][1][1]) / covDeterms[ci];
		inverseCovs[ci][1][0] = -(cov[ci][1][0] * cov[ci][2][2] - cov[ci][1][2] * cov[ci][2][0]) / covDeterms[ci];
		inverseCovs[ci][1][1] = (cov[ci][0][0] * cov[ci][2][2] - cov[ci][0][2] * cov[ci][2][0]) / covDeterms[ci];
		inverseCovs[ci][1][2] = -(cov[ci][0][0] * cov[ci][1][2] - cov[ci][0][2] * cov[ci][1][0]) / covDeterms[ci];
		inverseCovs[ci][2][0] = (cov[ci][1][0] * cov[ci][2][1] - cov[ci][1][1] * cov[ci][2][0]) / covDeterms[ci];
		inverseCovs[ci][2][1] = -(cov[ci][0][0] * cov[ci][2][1] - cov[ci][0][1] * cov[ci][2][0]) / covDeterms[ci];
		inverseCovs[ci][2][2] = (cov[ci][0][0] * cov[ci][1][1] - cov[ci][0][1] * cov[ci][1][0]) / covDeterms[ci];

	}
}


static double MycalcBeta(const Mat& img)
{
	double v = 0;
	int cot = 0;
	for (int x = 0; x < img.rows; x++) {
		for (int y = 0; y < img.cols; y++) {
			Vec3d color = img.at<Vec3b>(x, y);
			if (x + 1 < img.rows) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(x + 1, y);
				v += color1.dot(color1);
				cot++;
			}
			if (y + 1 < img.rows) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(x, y + 1);
				v += color1.dot(color1);
				cot++;
			}
			if (x + 1 < img.rows && y + 1 < img.cols) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(x + 1, y + 1);
				v += color1.dot(color1);
				cot++;
			}
			if (x - 1 >= 0 && y + 1 < img.cols) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(x - 1, y + 1);
				v += color1.dot(color1);
				cot++;
			}
		}
	}
	v = 1.0f / (2 * (v) / cot);
	return v;
}

static void MycalcWeight(const Mat& img, Mat& left, Mat& upleft, Mat& up, Mat& upright, double beta, double gamma)
{
	left.create(img.rows, img.cols, CV_64FC1);
	upleft.create(img.rows, img.cols, CV_64FC1);
	up.create(img.rows, img.cols, CV_64FC1);
	upright.create(img.rows, img.cols, CV_64FC1);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3d color = img.at<Vec3b>(i, j);
			if (i > 0) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(i - 1, j);
				up.at<double>(i, j) = gamma * exp(-beta * color1.dot(color1));
			}
			else {
				up.at<double>(i, j) = 0;
			}
			if (i > 0 && j < img.cols - 1) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(i - 1, j + 1);
				upright.at<double>(i, j) = (gamma / std::sqrt(2.0f)) * exp(-beta * color1.dot(color1));
			}
			else {
				upright.at<double>(i, j) = 0;
			}
			if (j > 0) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(i, j - 1);
				left.at<double>(i, j) = gamma * exp(-beta * color1.dot(color1));
			}
			else {
				left.at<double>(i, j) = 0;
			}
			if (j > 0 && i > 0) {
				Vec3d color1 = color - (Vec3d)img.at<Vec3b>(i - 1, j - 1);
				upleft.at<double>(i, j) = (gamma / std::sqrt(2.0f)) * exp(-beta * color1.dot(color1));
			}
			else {
				upleft.at<double>(i, j) = 0;
			}
		}
	}
}
static void MyinitMask(Mat& mask, Size imgsize, Rect rect)
{
	mask.create(imgsize, CV_8UC1);
	mask.setTo(GC_BGD);
	(mask(rect)).setTo(Scalar(GC_PR_FGD)); 
}

static void MyinitGMM(const Mat& img, const Mat& mask, MyGMM& fgdGMM, MyGMM& bgdGMM)
{
	const int kMeansItCount = 10;  //迭代次数
	const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii

	Mat bgdLabels, fgdLabels; //记录背景和前景的像素样本集中每个像素对应GMM的哪个高斯模型，论文中的kn
	std::vector<Vec3f> bgdSamples, fgdSamples; //背景和前景的像素样本集
	Point p;
	for (p.y = 0; p.y < img.rows; p.y++)
	{
		for (p.x = 0; p.x < img.cols; p.x++)
		{
			//mask中标记为GC_BGD和GC_PR_BGD的像素都作为背景的样本像素
			if (mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD)
				bgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
			else // GC_FGD | GC_PR_FGD
				fgdSamples.push_back((Vec3f)img.at<Vec3b>(p));
		}
	}
	CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());

	//kmeans中参数_bgdSamples为：每行一个样本
	//kmeans的输出为bgdLabels，里面保存的是输入样本集中每一个样本对应的类标签（样本聚为componentsCount类后）
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, MyGMM::componentsCount, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, MyGMM::componentsCount, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

	//经过上面的步骤后，每个像素所属的高斯模型就确定的了，那么就可以估计GMM中每个高斯模型的参数了。
	bgdGMM.initLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++) {
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	}
		bgdGMM.endLearning();

	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();
	fgdGMM.test();
	
}


static void MyassignGMMComponent(const Mat& img, const Mat& mask, const MyGMM& bgdGMM, const MyGMM& fgdGMM, Mat& ID)
{
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			Vec3d color = img.at<Vec3b>(i, j);
			if (mask.at<uchar>(i, j) == GC_BGD || mask.at<uchar>(i,j) == GC_PR_BGD) {
				ID.at<int>(i, j) = bgdGMM.whichComponent(color);
			}
			else {
				ID.at<int>(i, j) = fgdGMM.whichComponent(color);
			}
		}
	}
}

static void MylearnGMM(const Mat& img, const Mat& mask, const Mat& ID, MyGMM& bgdGMM, MyGMM& fgdGMM)
{
	bgdGMM.initLearning();
	fgdGMM.initLearning();
	for (int ci = 0; ci < MyGMM::componentsCount; ci++) {
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (ID.at<int>(i, j) == ci) {
					if (mask.at<uchar>(i, j) == GC_BGD||
						mask.at<uchar>(i, j) == GC_PR_BGD) {
						bgdGMM.addSample(ci, img.at<Vec3b>(i, j));
					}
					else {
						fgdGMM.addSample(ci, img.at<Vec3b>(i, j));
					}
				}
			}
		}
	}
	bgdGMM.endLearning();
	fgdGMM.endLearning();
	fgdGMM.test();
}


static void MyconstructGraph(const Mat& img, const Mat& mask, const MyGMM& bgdGMM, const MyGMM& fgdGMM, const Mat& left, const Mat& upleft, const Mat& up, const Mat& upright, GCGraph<double>& graph, double lambda) {
	int vertex = img.cols * img.rows;
	int edge = 2 * ((img.rows - 1) * (img.cols) + (img.cols - 1) * (img.rows) + 2 * (img.cols - 1) * (img.rows - 1));
	graph.create(vertex, edge);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			int vertexID = graph.addVtx();
			Vec3b color = img.at<Vec3b>(i, j);

			double fromSource, toSink;
			if (mask.at<uchar>(i, j) == GC_PR_BGD || mask.at<uchar>(i, j) == GC_PR_FGD) {
				fromSource = -log(bgdGMM(color));
				toSink = -log(fgdGMM(color));
			}
			else if (mask.at<uchar>(i, j) == GC_BGD) {
				fromSource = 0;
				toSink = lambda;
			}
			else {
				fromSource = lambda;
				toSink = 0;
			}
			graph.addTermWeights(vertexID, fromSource, toSink);
			if (i > 0) {
				double w = up.at<double>(i, j);
				graph.addEdges(vertexID, vertexID - img.cols, w, w);
			}
			if (j > 0) {
				double w = left.at<double>(i, j);
				graph.addEdges(vertexID, vertexID - 1, w, w);
			}
			if (i > 0 && j > 0) {
				double w = upleft.at<double>(i, j);
				graph.addEdges(vertexID, vertexID - img.cols - 1, w, w);
			}
			if (j < img.cols - 1 && i>0) {
				double w = upright.at<double>(i, j);
				graph.addEdges(vertexID, vertexID - img.cols + 1, w, w);
			}

		}
	}
}


static void MySegmentation(GCGraph<double>& graph, Mat& mask)
{
	graph.maxFlow();
	int summ = 0;
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<uchar>(i, j) == GC_PR_BGD || mask.at < uchar >(i, j) == GC_PR_FGD) {
				if (graph.inSourceSegment(i * mask.cols + j)) {
					mask.at<uchar>(i, j) = GC_PR_FGD;
					summ++;
				}
				else {
					mask.at<uchar>(i, j) = GC_PR_BGD;
				}
			}
		}
	}
	//cout << summ << endl;
}


void MygrabCut(InputArray _img, InputOutputArray _mask, Rect rect,
	InputOutputArray _bgdModel, InputOutputArray _fgdModel,
	int iterCount, int mode)
{
	Mat img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	MyGMM bgdGMM(bgdModel);
	MyGMM fgdGMM(fgdModel);
	
	Mat ID(img.size(), CV_32SC1);

	if (mode == GC_INIT_WITH_RECT) {
		MyinitMask(mask, img.size(), rect);
		MyinitGMM(img, mask, bgdGMM, fgdGMM);
		//cout << mask;
	}
	cout << 1;
	const double gamma = 50;
	const double lambda = 9 * gamma;
	const double beta = MycalcBeta(img);

	Mat left, upleft, up, upright;
	MycalcWeight(img, left, upleft, up, upright, beta, gamma);

	for (int i = 0; i < iterCount; i++) {
		GCGraph<double> graph;
		MyassignGMMComponent(img, mask, bgdGMM, fgdGMM, ID);
		MylearnGMM(img, mask, ID, bgdGMM, fgdGMM);
		MyconstructGraph(img, mask, bgdGMM, fgdGMM, left, upleft, up, upright, graph, lambda);
		MySegmentation(graph, mask);	
	}

}