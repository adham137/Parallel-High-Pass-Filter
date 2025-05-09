#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include "../common_utilities.cpp"
namespace fs = std::filesystem;

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

    // Use the createHighPassKernel function from common_utilities.cpp
    cv::Mat kernel = createHighPassKernel(kernelSize);
    
    // Display kernel (unchanged)
    std::cout << "Using " << kernelSize << "x" << kernelSize << " kernel:\n";
    for (int i = 0; i < kernel.rows; i++) {
        for (int j = 0; j < kernel.cols; j++) {
            std::cout << kernel.at<float>(i, j) << "\t";
        }
        std::cout << "\n";
    }

    // Rest of the function remains exactly the same
    int padding = kernelSize / 2;
    int paddedRows = inputImage.rows + 2 * padding;
    int paddedCols = inputImage.cols + 2 * padding;
    
    cv::Mat outputImage = cv::Mat::zeros(inputImage.size(), inputImage.type());
    
    auto start = std::chrono::high_resolution_clock::now();

    // Manual border replication 
    cv::Mat paddedImage = cv::Mat::zeros(paddedRows, paddedCols, inputImage.type());
    
    // Copy original image to center
    for (int i = 0; i < inputImage.rows; i++) {
        for (int j = 0; j < inputImage.cols; j++) {
            paddedImage.at<cv::Vec3b>(i + padding, j + padding) = inputImage.at<cv::Vec3b>(i, j);
        }
    }
    
    // Replicate borders - top and bottom
    for (int j = padding; j < paddedCols - padding; j++) {
        for (int i = 0; i < padding; i++) {
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(padding, j);
        }
        for (int i = paddedRows - padding; i < paddedRows; i++) {
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(paddedRows - padding - 1, j);
        }
    }
    
    // Replicate borders - left and right
    for (int i = 0; i < paddedRows; i++) {
        for (int j = 0; j < padding; j++) {
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, padding);
        }
        for (int j = paddedCols - padding; j < paddedCols; j++) {
            paddedImage.at<cv::Vec3b>(i, j) = paddedImage.at<cv::Vec3b>(i, paddedCols - padding - 1);
        }
    }

    // Apply convolution
    for (int i = padding; i < paddedRows - padding; i++) {
        for (int j = padding; j < paddedCols - padding; j++) {
            float sum[3] = {0};
            
            for (int ki = -padding; ki <= padding; ki++) {
                for (int kj = -padding; kj <= padding; kj++) {
                    cv::Vec3b pixel = paddedImage.at<cv::Vec3b>(i + ki, j + kj);
                    float k = kernel.at<float>(ki + padding, kj + padding);
                    
                    sum[0] += pixel[0] * k;
                    sum[1] += pixel[1] * k;
                    sum[2] += pixel[2] * k;
                }
            }

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




