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
cv::Mat replicatePadding(
    const cv::Mat& src, 
    int top, 
    int bottom,
    int left,
    int right
) {
    if (src.empty()) {
        std::cerr << "Error: Empty input image for padding." << std::endl;
        return cv::Mat();
    }
    
    if (top < 0 || bottom < 0 || left < 0 || right < 0) {
        std::cerr << "Error: Padding values must be non-negative." << std::endl;
        return src.clone();
    }
    
    // Calculate dimensions
    int srcRows = src.rows;
    int srcCols = src.cols;
    
    // Calculate new dimensions
    int dstRows = srcRows + top + bottom;
    int dstCols = srcCols + left + right;
    
    // If no padding requested, return a copy of the original
    if (dstRows == srcRows && dstCols == srcCols) {
        return src.clone();
    }
    
    // Create the padded image
    cv::Mat dst = cv::Mat::zeros(dstRows, dstCols, src.type());
    
    // Copy the original image to the center of dst
    src.copyTo(dst(cv::Rect(left, top, srcCols, srcRows)));
    
    // Replicate top edge
    for (int i = 0; i < top; i++) {
        for (int j = 0; j < dstCols; j++) {
            int srcCol = j - left;
            
            // Handle corners by clamping the source column
            srcCol = std::max(0, std::min(srcCols - 1, srcCol));
            
            dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(0, srcCol);
        }
    }
    
    // Replicate bottom edge
    for (int i = srcRows + top; i < dstRows; i++) {
        for (int j = 0; j < dstCols; j++) {
            int srcCol = j - left;
            
            // Handle corners by clamping the source column
            srcCol = std::max(0, std::min(srcCols - 1, srcCol));
            
            dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(srcRows - 1, srcCol);
        }
    }
    
    // Replicate left edge (excluding corners that were already handled)
    for (int i = top; i < srcRows + top; i++) {
        for (int j = 0; j < left; j++) {
            dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i - top, 0);
        }
    }
    
    // Replicate right edge (excluding corners that were already handled)
    for (int i = top; i < srcRows + top; i++) {
        for (int j = srcCols + left; j < dstCols; j++) {
            dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i - top, srcCols - 1);
        }
    }
    
    return dst;
}