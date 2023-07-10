#include <time.h>
#include "MyGrabCut.h"
 

using namespace std;
using namespace cv;

Point p1;
Point p2;
int t = 0;
void mouse(int events, int x, int y, int flag, void*);
int main()
{
	cv::Mat image = cv::imread("2.png");
	cv::namedWindow("origin");
	cv::imshow("origin", image);
	setMouseCallback("origin", mouse);
	waitKey();
	// 定义前景、背景和分割结果
	cv::Mat bgModel, fgModel, result;

	cv::Rect rectangle(p1,p2);
	double start = clock();
	MygrabCut(image,
		result,
		rectangle,
		bgModel,
		fgModel,
		5,
		cv::GC_INIT_WITH_RECT); // use rectangle
	double end = clock();
	printf("%f", (end - start) / 1000.0);

	// 标记可能属于前景的区域
	cv::compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);
	// or:
	//	result= result&1;

	// 创建前景图像
	cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(255, 255, 255));
	image.copyTo(foreground, result); // 复制前景图像

	// 在原图像绘制矩形区域
	cv::rectangle(image, rectangle, cv::Scalar(200, 0, 200), 4);
	cv::namedWindow("origin");
	cv::imshow("origin", image);
	cv::namedWindow("Foreground");
	cv::imshow("Foreground", foreground);
	waitKey();

	return 0;
}


void mouse(int events, int x, int y, int flag, void*)
{
	if (events == EVENT_LBUTTONDOWN) {
		p1 = Point(x, y);
	}
	if (events == EVENT_LBUTTONUP) {
		p2 = Point(x, y);
		t = 1;
	}
}