// Static kernel size implementation

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
#include <chrono>
using namespace std;

namespace fs = std::filesystem;

int main() {
    const string baseDir = "E:/Downloads/Semester 10/HPC/Parallel-High-Pass-Filter/";
    const string inDir   = baseDir + "images/input";
    const string outDir  = baseDir + "images/output";

    cout << "CWD:       " << fs::current_path()   << "\n";
    cout << "Input dir: " << fs::absolute(inDir) << "\n";

    try {
        //  Check existence
        if (!fs::exists(inDir) || !fs::is_directory(inDir))
            throw std::runtime_error("Input directory not found: " + inDir);
        if (!fs::exists(outDir))
            fs::create_directories(outDir);

        // Static 3×3 high-pass kernel
        const int kernelSize = 3;
        const int padding    = kernelSize / 2;
        cv::Mat kernel = (cv::Mat_<float>(3,3) <<
              0, -1,  0,
             -1,  4, -1,
              0, -1,  0
        );

        // Process each image
        for (auto& entry : fs::directory_iterator(inDir)) {
            if (!entry.is_regular_file()) continue;

            string inputPath  = entry.path().string();
            string filename   = entry.path().filename().string();
            string outputPath = outDir + "/" + filename;

            cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
            if (src.empty()) {
                cerr << "Error reading: " << inputPath << "\n";
                continue;
            }

            int rows     = src.rows;
            int cols     = src.cols;
            int channels = src.channels();
            cv::Mat dst  = cv::Mat::zeros(src.size(), src.type());

            // Prepare padded image with manual border-replicate
            int paddedRows = rows + 2 * padding;
            int paddedCols = cols + 2 * padding;
            cv::Mat padded = cv::Mat::zeros(paddedRows, paddedCols, src.type());

            //  Copy the original image to the center of paddedImage
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    padded.at<cv::Vec3b>(i + padding, j + padding) = src.at<cv::Vec3b>(i, j);
                }
            }

            //  Replicate top & bottom edges
            for (int j = padding; j < paddedCols - padding; ++j) {
                // top rows
                for (int i = 0; i < padding; ++i) {
                    padded.at<cv::Vec3b>(i, j) = padded.at<cv::Vec3b>(padding, j);
                }
                // bottom rows
                for (int i = paddedRows - padding; i < paddedRows; ++i) {
                    padded.at<cv::Vec3b>(i, j) = padded.at<cv::Vec3b>(paddedRows - padding - 1, j);
                }
            }

            //  Replicate left & right edges (including corners)
            for (int i = 0; i < paddedRows; ++i) {
                for (int j = 0; j < padding; ++j) {
                    padded.at<cv::Vec3b>(i, j) = padded.at<cv::Vec3b>(i, padding);
                }
                for (int j = paddedCols - padding; j < paddedCols; ++j) {
                    padded.at<cv::Vec3b>(i, j) = padded.at<cv::Vec3b>(i, paddedCols - padding - 1);
                }
            }

            // Convolution & timing
            auto start = std::chrono::high_resolution_clock::now();

            for (int i = padding; i < paddedRows - padding; ++i) {
                for (int j = padding; j < paddedCols - padding; ++j) {
                    float sumB = 0.0f, sumG = 0.0f, sumR = 0.0f;
                    // Apply 3×3 kernel
                    for (int ki = -padding; ki <= padding; ++ki) {
                        for (int kj = -padding; kj <= padding; ++kj) {
                            cv::Vec3b pix = padded.at<cv::Vec3b>(i + ki, j + kj);
                            float k    = kernel.at<float>(ki + padding, kj + padding);
                            sumB += pix[0] * k;
                            sumG += pix[1] * k;
                            sumR += pix[2] * k;
                        }
                    }
                    // Write to dst (with saturation)
                    dst.at<cv::Vec3b>(i - padding, j - padding) = cv::Vec3b(
                        cv::saturate_cast<uchar>(sumB),
                        cv::saturate_cast<uchar>(sumG),
                        cv::saturate_cast<uchar>(sumR)
                    );
                }
            }

            auto end     = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            cout << "Processed " << filename
                 << " in " << elapsed.count() << " sec\n";

            if (!cv::imwrite(outputPath, dst)) {
                cerr << "Failed to save: " << outputPath << "\n";
            }
        }

        cout << "All done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
}


//-----------------------------------------------------------------------------------------------------------------------------------
// Dynamic kernel size implementation

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <vector>
// #include <chrono>
// #include <filesystem>
// #include <stdexcept>

// namespace fs = std::filesystem;

// cv::Mat applyHighPassFilter(const cv::Mat& inputImage, int kernelSize = 3) {
//     if (inputImage.empty()) {
//         std::cerr << "Error: Empty input image." << std::endl;
//         return cv::Mat();
//     }

//     // Force kernel size to be odd and >= 3
//     if (kernelSize < 3 || kernelSize % 2 == 0) {
//         kernelSize = 3;
//         std::cerr << "Invalid kernel size. Using default 3x3." << std::endl;
//     }

//     cv::Mat kernel;
//     if (kernelSize == 3) {
//         kernel = (cv::Mat_<float>(3, 3) << 
//             0, -1,  0,
//            -1,  4, -1,  
//             0, -1,  0);
//     } else {
//         // Dynamic kernel for larger sizes (N²-1 center)
//         kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) * -1;
//         int center = kernelSize / 2;
//         kernel.at<float>(center, center) = kernelSize * kernelSize - 1;
//     }
    
//     std::cout << "Using " << kernelSize << "x" << kernelSize << " kernel:\n";
//     for (int i = 0; i < kernel.rows; i++) {
//         for (int j = 0; j < kernel.cols; j++) {
//             std::cout << kernel.at<float>(i, j) << "\t";
//         }
//         std::cout << "\n";
//     }

//     // Manual convolution
//     int padding = kernelSize / 2;
//     int paddedRows = inputImage.rows + 2 * padding;
//     int paddedCols = inputImage.cols + 2 * padding;
    
//     // Create output image
//     cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
    
//     auto start = std::chrono::high_resolution_clock::now();

//     // Manual border replication - Create a padded image without using copyMakeBorder
//     cv::Mat paddedImage = cv::Mat::zeros(paddedRows, paddedCols, inputImage.type());
    
//     // Copy the original image to the center of paddedImage
//     for (int i = 0; i < inputImage.rows; i++) {
//         for (int j = 0; j < inputImage.cols; j++) {
//             paddedImage.at<cv::Vec3b>(i + padding, j + padding) = inputImage.at<cv::Vec3b>(i, j);
//         }
//     }
    
//     // Replicate borders - top and bottom edges
//     for (int j = padding; j < paddedCols - padding; j++) {
//         // Top padding - replicate from first row
//         for (int i = 0; i < padding; i++) {
//             paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(padding, j);
//         }
//         // Bottom padding - replicate from last row
//         for (int i = paddedRows - padding; i < paddedRows; i++) {
//             paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(paddedRows - padding - 1, j);
//         }
//     }
    
//     // Replicate borders - left and right edges (including corners)
//     for (int i = 0; i < paddedRows; i++) {
//         // Left padding - replicate from first column
//         for (int j = 0; j < padding; j++) {
//             paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, padding);
//         }
//         // Right padding - replicate from last column
//         for (int j = paddedCols - padding; j < paddedCols; j++) {
//             paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, paddedCols - padding - 1);
//         }
//     }

//     // Apply convolution
//     for (int i = padding; i < paddedRows - padding; i++) {
//         for (int j = padding; j < paddedCols - padding; j++) {
//             float sum[3] = {0}; // For BGR channels
            
//             // Apply kernel
//             for (int ki = -padding; ki <= padding; ki++) {
//                 for (int kj = -padding; kj <= padding; kj++) {
//                     cv::Vec3b pixel = paddedImage.at<cv::Vec3b>(i + ki, j + kj);
//                     float k = kernel.at<float>(ki + padding, kj + padding);
                    
//                     sum[0] += pixel[0] * k; // Blue
//                     sum[1] += pixel[1] * k; // Green
//                     sum[2] += pixel[2] * k; // Red
//                 }
//             }

//             // Clamp and store result
//             outputImage.at<cv::Vec3b>(i - padding, j - padding) = cv::Vec3b(
//                 cv::saturate_cast<uchar>(sum[0]),
//                 cv::saturate_cast<uchar>(sum[1]),
//                 cv::saturate_cast<uchar>(sum[2])
//             );
//         }
//     }

//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> elapsed = end - start;
//     std::cout << "Processing time: " << elapsed.count() << " seconds\n";

//     return outputImage;
// }

// int main(int argc, char** argv) {
//     const std::string baseDir = "E:/Downloads/Semester 10/HPC/Parallel-High-Pass-Filter/";
//     const std::string inDir = baseDir + "images/input";
//     const std::string outDir = baseDir + "images/output";
    
//     // Default to 3x3 kernel
//     int kernelSize = 3;
//     if (argc >= 2) {
//         try {
//             kernelSize = std::stoi(argv[1]);
//             if (kernelSize % 2 == 0 || kernelSize < 3) {
//                 kernelSize = 3;
//                 std::cerr << "Invalid size. Using default 3x3 kernel.\n";
//             }
//         } catch (...) {
//             std::cerr << "Invalid argument. Using default 3x3 kernel.\n";
//         }
//     }

//     try {
//         // Verify directories
//         if (!fs::exists(inDir)) {
//             throw std::runtime_error("Input directory not found: " + inDir);
//         }

//         if (!fs::exists(outDir)) {
//             fs::create_directories(outDir);
//             std::cout << "Created output directory: " << outDir << "\n";
//         }

//         // Process all images in input directory
//         for (const auto& entry : fs::directory_iterator(inDir)) {
//             if (!entry.is_regular_file()) continue;
            
//             std::string inputPath = entry.path().string();
//             std::string filename = entry.path().filename().string();
//             std::string outputPath = outDir + "/" + filename;

//             // Read image
//             cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
//             if (inputImage.empty()) {
//                 std::cerr << "Warning: Could not read image: " << inputPath << "\n";
//                 continue;
//             }

//             std::cout << "\nProcessing: " << filename 
//                       << " (" << inputImage.cols << "x" << inputImage.rows << ")"
//                       << " with " << kernelSize << "x" << kernelSize << " kernel\n";

//             // Apply filter
//             cv::Mat result = applyHighPassFilter(inputImage, kernelSize);

//             // Save result
//             if (!cv::imwrite(outputPath, result)) {
//                 std::cerr << "Warning: Failed to save image: " << outputPath << "\n";
//             } else {
//                 std::cout << "Saved processed image to: " << outputPath << "\n";
//             }
//         }

//         std::cout << "\nProcessing complete. Results saved to: " << outDir << "\n";
//         return 0;

//     } catch (const std::exception& e) {
//         std::cerr << "\nFatal Error: " << e.what() << "\n";
//         return 1;
//     }
// }