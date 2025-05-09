#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

cv::Mat createHighPassKernel(int size) {
    if(size == 3) {
        return (cv::Mat_<float>(3,3) << 
            0.0f, -1.0f,  0.0f,
            -1.0f,  4.0f, -1.0f,
            0.0f, -1.0f,  0.0f
            );
    }   
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) * -1;
    int center = size / 2;
    kernel.at<float>(center, center) = size * size - 1;
    return kernel;
}

