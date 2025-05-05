#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

// Manual convolution function
// cv::Mat applyHighPassFilter(const cv::Mat& inputImage, int kernelSize = 3) {
//     // Validate input
//     if (inputImage.empty()) {
//         std::cerr << "Error: Empty input image." << std::endl;
//         return cv::Mat();
//     }

//     // Force kernel size to be odd and >= 3
//     if (kernelSize < 3 || kernelSize % 2 == 0) {
//         kernelSize = 3;
//         std::cerr << "Invalid kernel size. Using default 3x3." << std::endl;
//     }

//     // Create kernel
//     cv::Mat kernel;
//     if (kernelSize == 3) {
//         // Use standard high-pass kernel
//         kernel = (cv::Mat_<float>(3,3) << 
//             0, -1,  0,
//            -1,  4, -1,
//             0, -1,  0);
//     } else {
//         // Create dynamic kernel
//         kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) * -1;
//         int center = kernelSize / 2;
//         kernel.at<float>(center, center) = kernelSize * kernelSize - 1;
//     }

//     // Display kernel
//     std::cout << "Using " << kernelSize << "x" << kernelSize << " kernel:\n";
//     for (int i = 0; i < kernel.rows; i++) {
//         for (int j = 0; j < kernel.cols; j++) {
//             std::cout << kernel.at<float>(i, j) << "\t";
//         }
//         std::cout << "\n";
//     }
cv::Mat applyHighPassFilter(const cv::Mat& inputImage, int kernelSize = 3) {
    if (inputImage.empty()) {
        std::cerr << "Error: Empty input image." << std::endl;
        return cv::Mat();
    }

    // Force kernel size to be odd and >= 3
    if (kernelSize < 3 || kernelSize % 2 == 0) {
        kernelSize = 3;
        std::cerr << "Invalid kernel size. Using default 3x3." << std::endl;
    }

    // Create kernel using your original logic for ALL sizes
    cv::Mat kernel;
    if (kernelSize == 3) {
        // Your original 3x3 kernel (normalized Laplacian)
        kernel = (cv::Mat_<float>(3, 3) << 
            0, -1,  0,
           -1,  4, -1,  // Note: 4 instead of 8 (normalized)
            0, -1,  0);
    } else {
        // Dynamic kernel for larger sizes (N²-1 center)
        kernel = cv::Mat::ones(kernelSize, kernelSize, CV_32F) * -1;
        int center = kernelSize / 2;
        kernel.at<float>(center, center) = kernelSize * kernelSize - 1;
    }
    // Display kernel
    std::cout << "Using " << kernelSize << "x" << kernelSize << " kernel:\n";
    for (int i = 0; i < kernel.rows; i++) {
        for (int j = 0; j < kernel.cols; j++) {
            std::cout << kernel.at<float>(i, j) << "\t";
        }
        std::cout << "\n";
    }

    // Manual convolution
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
    int padding = kernelSize / 2;
    
    auto start = std::chrono::high_resolution_clock::now();

    // Handle borders by replication
    cv::Mat paddedImage;
    cv::copyMakeBorder(inputImage, paddedImage, padding, padding, padding, padding, cv::BORDER_REPLICATE);

    for (int i = padding; i < paddedImage.rows - padding; i++) {
        for (int j = padding; j < paddedImage.cols - padding; j++) {
            float sum[3] = {0}; // For BGR channels
            
            // Apply kernel
            for (int ki = -padding; ki <= padding; ki++) {
                for (int kj = -padding; kj <= padding; kj++) {
                    cv::Vec3b pixel = paddedImage.at<cv::Vec3b>(i + ki, j + kj);
                    float k = kernel.at<float>(ki + padding, kj + padding);
                    
                    sum[0] += pixel[0] * k; // Blue
                    sum[1] += pixel[1] * k; // Green
                    sum[2] += pixel[2] * k; // Red
                }
            }

            // Clamp and store result
            outputImage.at<cv::Vec3b>(i - padding, j - padding) = cv::Vec3b(
                cv::saturate_cast<uchar>(sum[0]),
                cv::saturate_cast<uchar>(sum[1]),
                cv::saturate_cast<uchar>(sum[2])
            );
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Processing time: " << elapsed.count() << " seconds\n";

    return outputImage;
}

int main(int argc, char** argv) {
    const std::string baseDir = "D:/Semester 10/HPC/Parallel-High-Pass-Filter/";
    const std::string inDir = baseDir + "images/input";
    const std::string outDir = baseDir + "images/output";
    
    // Default to 3x3 kernel
    int kernelSize = 3;
    if (argc >= 2) {
        try {
            kernelSize = std::stoi(argv[1]);
            if (kernelSize % 2 == 0 || kernelSize < 3) {
                kernelSize = 3;
                std::cerr << "Invalid size. Using default 3x3 kernel.\n";
            }
        } catch (...) {
            std::cerr << "Invalid argument. Using default 3x3 kernel.\n";
        }
    }

    try {
        // Verify directories
        if (!fs::exists(inDir)) {
            throw std::runtime_error("Input directory not found: " + inDir);
        }

        if (!fs::exists(outDir)) {
            fs::create_directories(outDir);
            std::cout << "Created output directory: " << outDir << "\n";
        }

        // Process all images in input directory
        for (const auto& entry : fs::directory_iterator(inDir)) {
            if (!entry.is_regular_file()) continue;
            
            std::string inputPath = entry.path().string();
            std::string filename = entry.path().filename().string();
            std::string outputPath = outDir + "/" + filename;

            // Read image
            cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
            if (inputImage.empty()) {
                std::cerr << "Warning: Could not read image: " << inputPath << "\n";
                continue;
            }

            std::cout << "\nProcessing: " << filename 
                      << " (" << inputImage.cols << "x" << inputImage.rows << ")"
                      << " with " << kernelSize << "x" << kernelSize << " kernel\n";

            // Apply filter
            cv::Mat result = applyHighPassFilter(inputImage, kernelSize);

            // Save result
            if (!cv::imwrite(outputPath, result)) {
                std::cerr << "Warning: Failed to save image: " << outputPath << "\n";
            } else {
                std::cout << "Saved processed image to: " << outputPath << "\n";
            }
        }

        std::cout << "\nProcessing complete. Results saved to: " << outDir << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nFatal Error: " << e.what() << "\n";
        return 1;
    }
}