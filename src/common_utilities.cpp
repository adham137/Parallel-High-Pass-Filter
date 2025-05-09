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
cv::Mat replicatePadding(const cv::Mat& input, int top, int bottom, int left, int right) {
    if (input.empty()) {
        return cv::Mat();
    }

    // Ensure padding values are non-negative
    top = std::max(0, top);
    bottom = std::max(0, bottom);
    left = std::max(0, left);
    right = std::max(0, right);

    int paddedRows = input.rows + top + bottom;
    int paddedCols = input.cols + left + right;
    cv::Mat paddedImage = cv::Mat::zeros(paddedRows, paddedCols, input.type());

    // Copy original image to center (safe access)
    for (int i = 0; i < input.rows; i++) {
        for (int j = 0; j < input.cols; j++) {
            paddedImage.at<cv::Vec3b>(i + top, j + left) = input.at<cv::Vec3b>(i, j);
        }
    }

    // Replicate top border (clamp to valid rows)
    for (int i = 0; i < top; i++) {
        for (int j = left; j < paddedCols - right; j++) {
            int srcRow = std::min(top, input.rows - 1);  // Clamp to valid row
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(srcRow, j);
        }
    }

    // Replicate bottom border (clamp to valid rows)
    for (int i = paddedRows - bottom; i < paddedRows; i++) {
        for (int j = left; j < paddedCols - right; j++) {
            int srcRow = std::max(paddedRows - bottom - 1, top);  // Clamp to valid row
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(srcRow, j);
        }
    }

    // Replicate left border (clamp to valid columns)
    for (int i = 0; i < paddedRows; i++) {
        for (int j = 0; j < left; j++) {
            int srcCol = std::min(left, input.cols - 1);  // Clamp to valid column
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, srcCol);
        }
    }

    // Replicate right border (clamp to valid columns)
    for (int i = 0; i < paddedRows; i++) {
        for (int j = paddedCols - right; j < paddedCols; j++) {
            int srcCol = std::max(paddedCols - right - 1, left);  // Clamp to valid column
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, srcCol);
        }
    }

    return paddedImage;
}