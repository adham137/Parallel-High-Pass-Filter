// run command: build\Debug\high_pass_filter_app.exe
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <omp.h>
// #include <chrono>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/highgui.hpp>

// using namespace cv;
// using namespace std;
// cv::Mat replicatePadding(
//     const cv::Mat& src, 
//     int top, 
//     int bottom,
//     int left,
//     int right
// ) {
//     if (src.empty()) {
//         std::cerr << "Error: Empty input image for padding." << std::endl;
//         return cv::Mat();
//     }
    
//     if (top < 0 || bottom < 0 || left < 0 || right < 0) {
//         std::cerr << "Error: Padding values must be non-negative." << std::endl;
//         return src.clone();
//     }
    
//     // Calculate dimensions
//     int srcRows = src.rows;
//     int srcCols = src.cols;
    
//     // Calculate new dimensions
//     int dstRows = srcRows + top + bottom;
//     int dstCols = srcCols + left + right;
    
//     // If no padding requested, return a copy of the original
//     if (dstRows == srcRows && dstCols == srcCols) {
//         return src.clone();
//     }
    
//     // Create the padded image
//     cv::Mat dst = cv::Mat::zeros(dstRows, dstCols, src.type());
    
//     // Copy the original image to the center of dst
//     src.copyTo(dst(cv::Rect(left, top, srcCols, srcRows)));
    
//     // Replicate top edge
//     for (int i = 0; i < top; i++) {
//         for (int j = 0; j < dstCols; j++) {
//             int srcCol = j - left;
            
//             // Handle corners by clamping the source column
//             srcCol = std::max(0, std::min(srcCols - 1, srcCol));
            
//             dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(0, srcCol);
//         }
//     }
    
//     // Replicate bottom edge
//     for (int i = srcRows + top; i < dstRows; i++) {
//         for (int j = 0; j < dstCols; j++) {
//             int srcCol = j - left;
            
//             // Handle corners by clamping the source column
//             srcCol = std::max(0, std::min(srcCols - 1, srcCol));
            
//             dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(srcRows - 1, srcCol);
//         }
//     }
    
//     // Replicate left edge (excluding corners that were already handled)
//     for (int i = top; i < srcRows + top; i++) {
//         for (int j = 0; j < left; j++) {
//             dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i - top, 0);
//         }
//     }
    
//     // Replicate right edge (excluding corners that were already handled)
//     for (int i = top; i < srcRows + top; i++) {
//         for (int j = srcCols + left; j < dstCols; j++) {
//             dst.at<cv::Vec3b>(i, j) = src.at<cv::Vec3b>(i - top, srcCols - 1);
//         }
//     }
    
//     return dst;
// }
// // Function to create high pass filter kernels of different sizes
// cv::Mat createHighPassKernel(int size) {
//     if (size == 3) {
//         return (cv::Mat_<float>(3, 3) << 
//             0.0f, -1.0f, 0.0f,
//             -1.0f, 4.0f, -1.0f,
//             0.0f, -1.0f, 0.0f
//         );
//     }

//     // For other sizes (5x5, 7x7, 9x9), create a kernel with -1 and a central positive value
//     cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) * -1;
//     int center = size / 2;
//     kernel.at<float>(center, center) = size * size - 1;
//     return kernel;
// }

// // Function to apply replicate padding to the input image
// Mat applyReplicatePadding(const Mat& input, int kernel_size) {
//     int pad = kernel_size / 2; // Calculate padding size
//     Mat padded;
//     copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_REPLICATE); // Replicate padding
//     return padded;
// }

// int main() {
//     string imagePath = "C:/Users/omar_/Desktop/HPC/Images/lena_input.png";
    
//     // Read the image in grayscale
//     Mat input = imread(imagePath, IMREAD_GRAYSCALE);
//     if (input.empty()) {
//         cout << "Could not open or find the image!" << endl;
//         return -1;
//     }

//     // Dynamically choose the kernel size
//     int kernel_size = 13; // Can be set to 3, 5, 7, or 9
//     Mat kernel = createHighPassKernel(kernel_size);

//     // Show the original image (before padding)
//     imshow("Original Image", input);

//     // Apply replicate padding to the image
//     Mat padded_input = replicatePadding(
//         input, 
//         kernel_size/2, 
//         kernel_size/2,
//         kernel_size/2,
//         kernel_size/2
//     );

//     // Show the padded image (after padding)
//     imshow("Padded Image", padded_input);

//     // Output image (same size as the input image)
//     Mat output = Mat::zeros(input.size(), CV_8U);

//     int rows = padded_input.rows;
//     int cols = padded_input.cols;

//     // Start measuring time using chrono
//     auto start = chrono::high_resolution_clock::now();

//     // Parallel processing with OpenMP
//     #pragma omp parallel for collapse(2)
//     for (int i = kernel_size / 2; i < rows - kernel_size / 2; ++i) {
//         for (int j = kernel_size / 2; j < cols - kernel_size / 2; ++j) {
//             int sum = 0;
//             for (int ki = -kernel_size / 2; ki <= kernel_size / 2; ++ki) {
//                 for (int kj = -kernel_size / 2; kj <= kernel_size / 2; ++kj) {
//                     sum += kernel.at<float>(ki + kernel_size / 2, kj + kernel_size / 2) *
//                            padded_input.at<uchar>(i + ki, j + kj);
//                 }
//             }
//             output.at<uchar>(i - kernel_size / 2, j - kernel_size / 2) = saturate_cast<uchar>(sum);
//         }
//     }

//     auto end = chrono::high_resolution_clock::now();
//     chrono::duration<double> duration = end - start;
//     cout << "Time taken for filtering: " << duration.count() << " seconds" << endl;

//     // Show the filtered image (after high-pass filter)
//     imshow("High Pass Filtered Image", output);

//     // Wait for the user to press a key
//     waitKey(0);
//     return 0;
// }
// // #include <opencv2/opencv.hpp>
// // #include <iostream>
// // #include <omp.h>
// // #include <opencv2/opencv.hpp>
// // #include <iostream>
// // #include <chrono>
// // #include <vector>
// // #include <omp.h>

// // using namespace cv;
// // using namespace std;

// // // Create high-pass filter kernel
// // cv::Mat createHighPassKernel(int size) {
// //     if (size == 3) {
// //         return (cv::Mat_<float>(3, 3) <<
// //             0.0f, -1.0f, 0.0f,
// //            -1.0f,  4.0f, -1.0f,
// //             0.0f, -1.0f, 0.0f
// //         );
// //     }
// //     cv::Mat kernel = Mat::ones(size, size, CV_32F) * -1;
// //     int center = size / 2;
// //     kernel.at<float>(center, center) = size * size - 1;
// //     return kernel;
// // }

// // int main() {
// //     // Load image in grayscale
// //     Mat image = imread("C:/Users/omar_/Desktop/HPC/Images/lena_input.png", IMREAD_GRAYSCALE);
// //     if (image.empty()) {
// //         cerr << "Failed to load image./n";
// //         return -1;
// //     }

// //     int kernel_size = 3;
// //     int pad = kernel_size / 2;
// //     int padded_rows = image.rows + 2 * pad;
// //     int padded_cols = image.cols + 2 * pad;
// //     Mat kernel = createHighPassKernel(kernel_size);

// //     vector<double> times;

// //     for (int run = 0; run < 10; ++run) {
// //         // Manual replicate padding
// //         Mat padded(padded_rows, padded_cols, CV_8U);
// //         // #pragma omp parallel for collapse(2)
// //         // for (int i = 0; i < padded_rows; ++i) {
// //         //     for (int j = 0; j < padded_cols; ++j) {
// //         //         int src_i = min(max(i - pad, 0), image.rows - 1);
// //         //         int src_j = min(max(j - pad, 0), image.cols - 1);
// //         //         padded.at<uchar>(i, j) = image.at<uchar>(src_i, src_j);
// //         //     }
// //         // }

// //         Mat output = Mat::zeros(image.size(), CV_8U);

// //         auto start = chrono::high_resolution_clock::now();

// //         // Apply high-pass filter
// //         // #pragma omp parallel for collapse(2)
// //         // for (int i = pad; i < padded_rows - pad; ++i) {
// //         //     for (int j = pad; j < padded_cols - pad; ++j) {
// //         //         float sum = 0.0f;
// //         //         for (int ki = -pad; ki <= pad; ++ki) {
// //         //             for (int kj = -pad; kj <= pad; ++kj) {
// //         //                 sum += kernel.at<float>(ki + pad, kj + pad) *
// //         //                        padded.at<uchar>(i + ki, j + kj);
// //         //             }
// //         //         }
// //         //         output.at<uchar>(i - pad, j - pad) = saturate_cast<uchar>(sum);
// //         //     }
// //         // }
// // // #pragma omp parallel for schedule(dynamic)
// // // for (int i = pad; i < padded_rows - pad; ++i) {
// // //     for (int j = pad; j < padded_cols - pad; ++j) {
// // //         int thread_id = omp_get_thread_num();
// // //         printf("Thread %d is processing pixel (%d, %d)/n", thread_id, i - pad, j - pad);

// // //         float sum = 0.0f;
// // //         for (int ki = -pad; ki <= pad; ++ki) {
// // //             for (int kj = -pad; kj <= pad; ++kj) {
// // //                 sum += kernel.at<float>(ki + pad, kj + pad) *
// // //                        padded.at<uchar>(i + ki, j + kj);
// // //             }
// // //         }
// // //         output.at<uchar>(i - pad, j - pad) = saturate_cast<uchar>(sum);
// // //     }
// // // }

// //     //     auto end = chrono::high_resolution_clock::now();
// //     //     chrono::duration<double> duration = end - start;
// //     //     times.push_back(duration.count());
// //     // }

// //     // Compute average, min, and max times
// //     double sum = 0, min_time = times[0], max_time = times[0];
// //     for (double t : times) {
// //         sum += t;
// //         if (t < min_time) min_time = t;
// //         if (t > max_time) max_time = t;
// //     }

// //     double avg_time = sum / times.size();
// //     cout << "Average Time: " << avg_time << " seconds" << endl;
// //     cout << "Min Time: " << min_time << " seconds" << endl;
// //     cout << "Max Time: " << max_time << " seconds" << endl;

// //     return 0;
// // }
// // #include <iostream>
// // #include <opencv2/opencv.hpp>
// // #include <omp.h>
// // #include <chrono>
// // #include <opencv2/imgcodecs.hpp>
// // #include <opencv2/highgui.hpp>
// // using namespace cv;
// // using namespace std;

// // // Function to create the high-pass filter kernel dynamically based on the size
// // cv::Mat createHighPassKernel(int size) {
// //     // If size is 3, return the standard 3x3 high-pass kernel
// //     if (size == 3) {
// //         return (cv::Mat_<float>(3, 3) << 
// //             0.0f, -1.0f,  0.0f,
// //             -1.0f,  4.0f, -1.0f,
// //             0.0f, -1.0f,  0.0f
// //         );
// //     }
    
// //     // For any other odd size, create a custom kernel
// //     cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) * -1;
// //     int center = size / 2;
// //     kernel.at<float>(center, center) = size * size - 1;
// //     return kernel;
// // }

// // int main() {
// //     string imagePath = "C:/Users/omar_/Desktop/HPC/Images/lena_input.png";

// //     Mat input = imread(imagePath, IMREAD_GRAYSCALE);
// //     if (input.empty()) {
// //         cout << "Could not open or find the image!" << endl;
// //         return -1;
// //     }

// //     Mat output = Mat::zeros(input.size(), CV_8U);

// //     // Ask user for the kernel size
// //     int kernelSize;
// //     cout << "Enter the kernel size (odd number): ";
// //     cin >> kernelSize;

// //     // Ensure the kernel size is odd
// //     if (kernelSize % 2 == 0) {
// //         cout << "The kernel size must be an odd number. Using 3x3 instead." << endl;
// //         kernelSize = 3;  // Default to 3x3 if even size is entered
// //     }

// //     // Create high pass kernel based on user input
// //     Mat kernel = createHighPassKernel(kernelSize);

// //     int rows = input.rows;
// //     int cols = input.cols;

// //     // Start measuring time using chrono
// //     auto start = chrono::high_resolution_clock::now();

// //     // Parallel processing with OpenMP
// //     #pragma omp parallel for collapse(2)
// //     for (int i = 1; i < rows - 1; ++i) {
// //         for (int j = 1; j < cols - 1; ++j) {
// //             int sum = 0;
// //             for (int ki = -kernelSize / 2; ki <= kernelSize / 2; ++ki) {
// //                 for (int kj = -kernelSize / 2; kj <= kernelSize / 2; ++kj) {
// //                     sum += kernel.at<float>(ki + kernelSize / 2, kj + kernelSize / 2) * input.at<uchar>(i + ki, j + kj);
// //                 }
// //             }
// //             output.at<uchar>(i, j) = saturate_cast<uchar>(sum);

// //             // Print which thread handled which pixel
// //             int thread_id = omp_get_thread_num();
// //             // #pragma omp critical
// //             cout << "Thread " << thread_id << " processed pixel (" << i << ", " << j << ")/n";
// //         }
// //     }

// //     auto end = chrono::high_resolution_clock::now();
// //     chrono::duration<double> duration = end - start;
// //     cout << "Time taken for filtering: " << duration.count() << " seconds" << endl;

// //     imshow("Original Image", input);
// //     imshow("High Pass Filtered Image", output);
// //     waitKey(0);
// //     return 0;
// // }#include <iostream>#include <iostream>
// // #include <opencv2/opencv.hpp>
// // #include <omp.h>
// // #include <chrono>

// // using namespace cv;
// // using namespace std;

// // cv::Mat createHighPassKernel(int size) {
// //     if (size == 3) {
// //         return (cv::Mat_<float>(3, 3) << 
// //             0.0f, -1.0f, 0.0f,
// //             -1.0f, 4.0f, -1.0f,
// //             0.0f, -1.0f, 0.0f
// //         );
// //     }

// //     cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) * -1;
// //     int center = size / 2;
// //     kernel.at<float>(center, center) = size * size - 1;
// //     return kernel;
// // }

// // Mat applyReplicatePadding(const Mat& input, int kernel_size) {
// //     int pad = kernel_size / 2;
// //     Mat padded;
// //     copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_REPLICATE);
// //     return padded;
// // }

// // void runFilterMultipleTimes(const Mat& input, int kernel_size, int iterations) {
// //     Mat kernel = createHighPassKernel(kernel_size);
// //     Mat padded_input = applyReplicatePadding(input, kernel_size);
// //     int rows = padded_input.rows;
// //     int cols = padded_input.cols;

// //     for (int run = 1; run <= iterations; ++run) {
// //         Mat output = Mat::zeros(input.size(), CV_8U);
// //         auto start = chrono::high_resolution_clock::now();

// //         #pragma omp parallel for collapse(2) schedule(guided)
// //         for (int i = kernel_size / 2; i < rows - kernel_size / 2; ++i) {
// //             for (int j = kernel_size / 2; j < cols - kernel_size / 2; ++j) {
// //                 int sum = 0;
// //                 for (int ki = -kernel_size / 2; ki <= kernel_size / 2; ++ki) {
// //                     for (int kj = -kernel_size / 2; kj <= kernel_size / 2; ++kj) {
// //                         sum += kernel.at<float>(ki + kernel_size / 2, kj + kernel_size / 2) *
// //                                padded_input.at<uchar>(i + ki, j + kj);
// //                     }
// //                 }
// //                 output.at<uchar>(i - kernel_size / 2, j - kernel_size / 2) = saturate_cast<uchar>(sum);
// //             }
// //         }

// //         auto end = chrono::high_resolution_clock::now();
// //         chrono::duration<double> duration = end - start;

// //         // Output iteration and time as CSV
// //         cout << run << "," << duration.count() << endl;
// //     }
// // }

// // int main() {
// //     string imagePath = "C:/Users/omar_/Desktop/HPC/Images/lena_input.png";
// //     Mat input = imread(imagePath, IMREAD_GRAYSCALE);
// //     if (input.empty()) {
// //         cout << "Could not open or find the image!" << endl;
// //         return -1;
// //     }

// //     int kernel_size = 3;

// //     // Run for different iteration counts
// //     runFilterMultipleTimes(input, kernel_size, 1);
// //     runFilterMultipleTimes(input, kernel_size, 10);
// //     runFilterMultipleTimes(input, kernel_size, 100);
// //     runFilterMultipleTimes(input, kernel_size, 1000);

// //     return 0;
// // }
#include <iostream>
#include <opencv2/opencv.hpp>
#include <omp.h>
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

// Function to create high pass filter kernels of different sizes
cv::Mat createHighPassKernel(int size) {
    if (size == 3) {
        return (cv::Mat_<float>(3, 3) << 
            0.0f, -1.0f, 0.0f,
            -1.0f, 4.0f, -1.0f,
            0.0f, -1.0f, 0.0f
        );
    }

    // For other sizes (5x5, 7x7, 9x9), create a kernel with -1 and a central positive value
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) * -1;
    int center = size / 2;
    kernel.at<float>(center, center) = size * size - 1;
    return kernel;
}

// Function to apply replicate padding to the input image
Mat applyReplicatePadding(const Mat& input, int kernel_size) {
    int pad = kernel_size / 2; // Calculate padding size
    Mat padded;
    copyMakeBorder(input, padded, pad, pad, pad, pad, BORDER_REPLICATE); // Replicate padding
    return padded;
}

int main() {
    string imagePath = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/input/lena.png";
    
    // Read the image in grayscale
    Mat input = imread(imagePath, IMREAD_GRAYSCALE);
    if (input.empty()) {
        cout << "Could not open or find the image!" << endl;
        return -1;
    }

    // Dynamically choose the kernel size
    int kernel_size = 3; // Can be set to 3, 5, 7, or 9
    Mat kernel = createHighPassKernel(kernel_size);

    // Apply replicate padding to the image
    Mat padded_input = applyReplicatePadding(input, kernel_size);

    // Output image (same size as the input image)
    Mat output = Mat::zeros(input.size(), CV_8U);

    int rows = padded_input.rows;
    int cols = padded_input.cols;

    // Start measuring time using chrono
    auto start = chrono::high_resolution_clock::now();

    // Parallel processing with OpenMP
    #pragma omp parallel for collapse(2)
    for (int i = kernel_size / 2; i < rows - kernel_size / 2; ++i) {
        for (int j = kernel_size / 2; j < cols - kernel_size / 2; ++j) {
            int sum = 0;
            for (int ki = -kernel_size / 2; ki <= kernel_size / 2; ++ki) {
                for (int kj = -kernel_size / 2; kj <= kernel_size / 2; ++kj) {
                    sum += kernel.at<float>(ki + kernel_size / 2, kj + kernel_size / 2) *
                           padded_input.at<uchar>(i + ki, j + kj);
                }
            }
            output.at<uchar>(i - kernel_size / 2, j - kernel_size / 2) = saturate_cast<uchar>(sum);
        }
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = end - start;
    cout << "Time taken for filtering: " << duration.count() << " seconds" << endl;

    // Show original and filtered images
    imshow("Original Image", input);
    imshow("High Pass Filtered Image", output);
    waitKey(0);
    return 0;
}
