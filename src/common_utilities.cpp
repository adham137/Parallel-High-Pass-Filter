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

// /**
//  * Applies manual border replication padding to an image
//  * 
//  * @param inputImage The original input image
//  * @param padding Padding size to add on each side (must be >= 0)
//  * @param padTop Whether to pad the top edge
//  * @param padBottom Whether to pad the bottom edge
//  * @param padLeft Whether to pad the left edge
//  * @param padRight Whether to pad the right edge
//  * @return Padded image with replicated borders as specified
//  */
// cv::Mat replicatePadding(
//     const cv::Mat& inputImage, 
//     int padding, 
//     bool padTop = true, 
//     bool padBottom = true, 
//     bool padLeft = true, 
//     bool padRight = true
// ) {
//     if (inputImage.empty()) {
//         std::cerr << "Error: Empty input image for padding." << std::endl;
//         return cv::Mat();
//     }
    
//     if (padding < 0) {
//         std::cerr << "Error: Padding must be non-negative." << std::endl;
//         return inputImage.clone();
//     }
    
//     // Calculate dimensions
//     int rows = inputImage.rows;
//     int cols = inputImage.cols;
    
//     // Calculate new dimensions based on which sides are being padded
//     int newRows = rows + (padTop ? padding : 0) + (padBottom ? padding : 0);
//     int newCols = cols + (padLeft ? padding : 0) + (padRight ? padding : 0);
    
//     // If no padding requested, return a copy of the original
//     if (newRows == rows && newCols == cols) {
//         return inputImage.clone();
//     }
    
//     // Create the padded image
//     cv::Mat paddedImage = cv::Mat::zeros(newRows, newCols, inputImage.type());
    
//     // Determine start positions for copying the original image
//     int startRow = padTop ? padding : 0;
//     int startCol = padLeft ? padding : 0;
    
//     // Copy the original image to the appropriate position in the padded image
//     inputImage.copyTo(paddedImage(cv::Rect(startCol, startRow, cols, rows)));
    
//     // Replicate top edge if requested
//     if (padTop) {
//         for (int i = 0; i < padding; i++) {
//             for (int j = 0; j < newCols; j++) {
//                 // For the corners, we need to be careful about the source pixel
//                 int sourceCol = j;
                
//                 // Adjust sourceCol for corners
//                 if (padLeft && j < padding) {
//                     sourceCol = padding;  // Use the first column of original image
//                 } else if (padRight && j >= cols + startCol) {
//                     sourceCol = cols + startCol - 1;  // Use the last column of original image
//                 }
                
//                 paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(startRow, sourceCol);
//             }
//         }
//     }
    
//     // Replicate bottom edge if requested
//     if (padBottom) {
//         for (int i = rows + startRow; i < newRows; i++) {
//             for (int j = 0; j < newCols; j++) {
//                 // Adjust sourceCol for corners
//                 int sourceCol = j;
                
//                 if (padLeft && j < padding) {
//                     sourceCol = padding;  // Use the first column of original image
//                 } else if (padRight && j >= cols + startCol) {
//                     sourceCol = cols + startCol - 1;  // Use the last column of original image
//                 }
                
//                 paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(rows + startRow - 1, sourceCol);
//             }
//         }
//     }
    
//     // Replicate left edge if requested (excluding corners that were already handled)
//     if (padLeft) {
//         int topSkip = padTop ? padding : 0;
//         int bottomLimit = newRows - (padBottom ? padding : 0);
        
//         for (int i = topSkip; i < bottomLimit; i++) {
//             for (int j = 0; j < padding; j++) {
//                 paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, padding);
//             }
//         }
//     }
    
//     // Replicate right edge if requested (excluding corners that were already handled)
//     if (padRight) {
//         int topSkip = padTop ? padding : 0;
//         int bottomLimit = newRows - (padBottom ? padding : 0);
//         int rightStart = cols + startCol;
        
//         for (int i = topSkip; i < bottomLimit; i++) {
//             for (int j = rightStart; j < newCols; j++) {
//                 paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, rightStart - 1);
//             }
//         }
//     }
    
//     return paddedImage;
// }

/**
 * Replicates OpenCV's copyMakeBorder with BORDER_REPLICATE behavior
 * 
 * @param src The original input image
 * @param top Number of pixels to add to the top
 * @param bottom Number of pixels to add to the bottom
 * @param left Number of pixels to add to the left
 * @param right Number of pixels to add to the right
 * @return Padded image with replicated borders
 */
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