// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <cmath>
using namespace std;


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}



void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

Mat_<uchar> nearestNeighbor(Mat_<uchar> src, float s) {
	int height = src.rows;
	int width = src.cols;

	int newheight = round(s * height);
	int newwidth = round(s * width);


	Mat_<uchar> dest = Mat(newheight, newwidth, CV_8UC1);
	for (int i = 0; i < newheight; i++) {
		for (int j = 0; j < newwidth; j++) {
			dest(i, j) = src(min(round(i / s),height-1), min(round(j / s),width-1));
		}
	}

	return dest;
}

void nearestNeighbor(float s) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);


		Mat_<uchar> nearest = nearestNeighbor(src, s);

		imshow("input image", src);
		imshow("Nearest neighbor interpolation", nearest);
		waitKey();
	}
}

Mat_<uchar> bilinear(Mat_<uchar> src, float s) {
	int height = src.rows;
	int width = src.cols;

	int newheight = round(s * height);
	int newwidth = round(s * width);


	Mat_<uchar> dest = Mat(newheight, newwidth, CV_8UC1);

	float isrc, jsrc;
	int ifloor, iceil, jfloor, jceil, ul, ur, dl, dr,u,d,val;

	for (int i = 0; i < newheight; i++) {
		for (int j = 0; j < newwidth; j++) {
			isrc = i / s;
			jsrc = j / s;
			ifloor = floor(isrc); 
			iceil = min(ceil(isrc),height-1);
			jfloor = floor(jsrc);
			jceil = min(ceil(jsrc),width-1);

			if (ifloor == iceil && jfloor == jceil) {
				val = src((int)isrc, (int)jsrc);
			}
			else 
				if (jfloor == jceil) {
					u = src(ifloor, (int)jsrc);
					d = src(iceil, (int)jsrc);
					val = u * (iceil - isrc) + d * (isrc - ifloor);
				}
				else 
					if (ifloor == iceil) {
						u = src((int)isrc, jfloor);
						d = src((int)isrc, jceil);
						val = u * (jceil - jsrc) + d * (jsrc - jfloor);
					}
					else {
						ul = src(iceil, jceil);
						ur = src(ifloor, jceil);
						dl = src(iceil, jfloor);
						dr = src(iceil, jceil); //u-up,d-down,l-left,r-right
						u = ul * (jceil - jsrc) + ur * (jsrc - jfloor);
						d = dl * (jceil - jsrc) + dr * (jsrc - jfloor);
						val = u * (iceil - isrc) + d * (isrc - ifloor);
					}
			
			dest(i, j) = val;
		}
	}

	return dest;
}

void bilinear(float s) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		Mat_<uchar> bilinearmat = bilinear(src, s);

		imshow("input image", src);
		imshow("Bilinear interpolation", bilinearmat);
		waitKey();
	}
}

float u(float s, float a) {
	if (s < 0) s = -s;
	if (s <= 1)
		return (a + 2) * s * s * s - (a + 3) * s * s + 1;
	if (s <= 2)
		return (a * s * s * s) - (5 * a * s * s) + (8 * a * s) - (4 * a);
	return 0;
}

Mat_<uchar> bicubic(Mat_<uchar> src, float s, float a) {
	int height = src.rows;
	int width = src.cols;


	int newheight = round(s * height);
	int newwidth = round(s * width);

	

	Mat_<uchar> dest = Mat(newheight, newwidth, CV_8UC1);
	float x, y, x1, x2, x3, x4, y1,y2,y3,y4;
	int f11, f12, f13, f14, f21, f22, f23, f24, f31, f32, f33, f34, f41, f42, f43, f44;

	for (int i = 0; i < newheight; i++) {
		for (int j = 0; j < newwidth; j++) {
			//printf("original height: %d, original width: %d\n", height, width);
			//printf("new height: %d, new width: %d\n", newheight, newwidth);
			y = i / s;
			x = j / s;

			x1 = 1 + x - floor(x);  //distance from x to point x1
			x2 = x - floor(x);
			x3 = floor(x) + 1 - x;
			x4 = floor(x) + 2 - x;

			y1 = 1 + y - floor(y);
			y2 = y - floor(y);
			y3 = floor(y) + 1 - y;
			y4 = floor(y) + 2 - y;

			// p1  p2  p3  p4
			// p5  p6  p7  p8
			// p9  p10 p11 p12
			// p13 p14 p15 p16
			
			//instead of addig a margin to the src I check not to get out of the image
			//and if it would get out then just take the margin pixel like it would be 
			//duplicated on the border
			Point p1(max(min((x - x1), width - 1), 0), max(min((y - y1), height - 1), 0));
			Point p2(max(min((x - x2), width - 1), 0), max(min((y - y1), height - 1), 0));
			Point p3(max(min((x + x3), width - 1), 0), max(min((y - y1), height - 1), 0));
			Point p4(max(min((x + x4), width - 1), 0), max(min((y - y1), height - 1), 0));

			Point p5(max(min((x - x1), width - 1), 0), max(min((y - y2), height - 1), 0));
			Point p6(max(min((x - x2), width - 1), 0), max(min((y - y2), height - 1), 0));
			Point p7(max(min((x + x3), width - 1), 0), max(min((y - y2), height - 1), 0));
			Point p8(max(min((x + x4), width - 1), 0), max(min((y - y2), height - 1), 0));


			Point p9(max(min((x - x1), width - 1), 0), max(min((y + y3), height - 1), 0));
			Point p10(max(min((x - x2), width - 1), 0), max(min((y + y3), height - 1), 0));
			Point p11(max(min((x + x3), width - 1), 0), max(min((y + y3), height - 1), 0));
			Point p12(max(min((x + x4), width - 1), 0), max(min((y + y3), height - 1), 0));

			Point p13(max(min((x - x1), width - 1), 0), max(min((y + y4), height - 1), 0));
			Point p14(max(min((x - x2), width - 1), 0), max(min((y + y4), height - 1), 0));
			Point p15(max(min((x + x3), width - 1), 0), max(min((y + y4), height - 1), 0));
			Point p16(max(min((x + x4), width - 1), 0), max(min((y + y4), height - 1), 0));

			f11 = src(p1);


			//printf("j:%d i:%d\n", j, i);
			//printf("x:%f y:%f\n", x, y);
			
			

			//matrix multiplications
			float m1[4] = { u(x1,a), u(x2,a), u(x3,a), u(x4,a) };
			//printf("x1: %f, x2: %f, x3:%f, x4:%f\n", x1, x2, x3, x4);
			//printf("ux1: %f, ux2: %f, ux3:%f, ux4:%f\n", m1[0], m1[1], m1[2], m1[3]);
			/*float sum = m1[0] + m1[1] + m1[2] + m1[3];
			m1[0] = m1[0] / sum;
			m1[1] = m1[1] / sum;
			m1[2] = m1[2] / sum;
			m1[3] = m1[3] / sum;*/
			//printf("ux1: %f, ux2: %f, ux3:%f, ux4:%f\n", m1[0], m1[1], m1[2], m1[3]);

			float m2[4][4] = {  src(p1), src(p5), src(p9), src(p13), 
								src(p2), src(p6), src(p10), src(p14), 
								src(p3), src(p7), src(p11), src(p15), 
								src(p4), src(p8), src(p12), src(p16)
			};
			float m3[4] = { u(y1,a), u(y2,a), u(y3,a), u(y4,a) };
			/*loat sum2 = m3[0] + m3[1] + m3[2] + m3[3];
			m3[0] = m3[0] / sum;
			m3[1] = m3[1] / sum;
			m3[2] = m3[2] / sum;
			m3[3] = m3[3] / sum;*/


			//printf("y1: %f, y2: %f, y3:%f, y4:%f\n", y1, y2, y3, y4);
			//printf("uy1: %f, uy2: %f, uy3:%f, uy4:%f\n", m3[0], m3[1], m3[2], m3[3]);

			float intermediate[4] = { 0,0,0,0 };
			for (int k = 0; k < 4; k++) {
				float sum = 0;
				for (int l = 0; l < 4; l++) {
					sum += m1[l] * m2[l][k];
				}
				intermediate[k] = sum;
			}

			float val = 0;
			for (int k = 0; k < 4; k++) {
				val += intermediate[k] * m3[k];
			}
			//printf("%d %d %d %d\n%d %d %d %d\n%d %d %d %d\n%d %d %d %d\n", src(p1), src(p2), src(p3), src(p4), src(p5), src(p6), src(p7), src(p8), src(p9), src(p10), src(p11), src(p12), src(p13), src(p14), src(p15), src(p16) );
			//printf("(%d %d) (%d %d) (%d %d) (%d %d)\n(%d %d) (%d %d) (%d %d) (%d %d)\n(%d %d) (%d %d) (%d %d) (%d %d)\n(%d %d) (%d %d) (%d %d) (%d %d)\n\n", p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y, p5.x, p5.y, p6.x, p6.y, p7.x, p7.y, p8.x, p8.y, p9.x, p9.y, p10.x, p10.y, p11.x, p11.y, p12.x, p12.y, p13.x, p13.y, p14.x, p14.y, p15.x, p15.y, p16.x, p16.y);

			//printf("final value: %f %d\n\n", val, (int)val);
			dest(i, j) = (int)val;
			//if (i == 10) return dest;
		}
	}

	return dest;
}



void bicubic(float s) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		Mat_<uchar> bicubicmat1 = bicubic(src, s,-0.5);

		imshow("input image", src);
		imshow("Bicubic interpolation -0.5", bicubicmat1);
		waitKey();
	}
}

void compare(float s) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat_<uchar> src = imread(fname, IMREAD_GRAYSCALE);

		double t = (double)getTickCount();
		Mat_<uchar> nearest = nearestNeighbor(src, s);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Nearest neighbor interpolation time: %.3f ms\n", t * 1000);

		t = (double)getTickCount();
		Mat_<uchar> bilinearMat = bilinear(src, s);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Bilinear interpolation time: %.3f ms\n", t * 1000);

		t = (double)getTickCount();
		Mat_<uchar> bicubicmat = bicubic(src, s,-0.5);
		t = ((double)getTickCount() - t) / getTickFrequency();
		printf("Bicubic interpolation time: %.3f ms\n", t * 1000);

		imshow("input image", src);
		imshow("Nearest neighbor interpolation", nearest);
		imshow("Bilinear interpolation", bilinearMat);
		imshow("Bicubic interpolation", bicubicmat);
		waitKey();
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("%f %f\n", u(0.7, -0.5), u(1.4, -0.5));
		float val = 0;
		for (int i = 0; i < 20; i++) {
			printf("%f, u(f)= %f\n", val + i * 0.1, u(val + i * 0.1,-0.5));
		}
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Nearest neighbor interpolation\n");
		printf(" 11 - Bilinear interpolation\n");
		printf(" 12 - Bicubic interpolation\n");
		printf(" 13 - Compare methods\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				float s1;
				printf("Scale: ");
				scanf("%f", &s1);
				nearestNeighbor(s1);
				break;
			case 11:
				float s2;
				printf("Scale: ");
				scanf("%f", &s2);
				bilinear(s2);
				break;
			case 12:
				float s3;
				printf("Scale: ");
				scanf("%f", &s3);
				bicubic(s3);
				break;
			case 13:
				float s4;
				printf("Scale: ");
				scanf("%f", &s4);
				compare(s4);
				break;
		}
	}
	while (op!=0);
	return 0;
}