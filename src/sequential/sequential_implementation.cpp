// Static kernel size implementation

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <filesystem>
// #include <stdexcept>
// using namespace std;

// namespace fs = filesystem;

// int main() {
//     const string baseDir = "E:/Downloads/Semester 10/HPC/Parallel-High-Pass-Filter/";
//     const string inDir   = baseDir + "images/input";
//     const string outDir  = baseDir + "images/output";

//     cout << "CWD:       " << fs::current_path()     << "\n";
//     cout << "Input dir: " << fs::absolute(inDir) << "\n";

//     try {
//         // Sanity‐check existence
//         if (!fs::exists(inDir) || !fs::is_directory(inDir))
//             throw runtime_error("Input directory not found: " + inDir);

//         // Ensure output folder exists
//         if (!fs::exists(outDir)) {
//             fs::create_directories(outDir);
//         }

//         // high‐pass kernel
//         cv::Mat kernel = (cv::Mat_<float>(3,3) <<
//               0, -1,  0,
//              -1,  4, -1,
//               0, -1,  0
//         );

//         // Process each image file
//         for (auto& entry : fs::directory_iterator(inDir)) {
//             if (!entry.is_regular_file()) continue;

//             const string  inputPath  = entry.path().string();
//             const string  filename   = entry.path().filename().string();
//             const string  outputPath = outDir + "/" + filename;

//             cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
//             if (src.empty()) {
//                 cerr << "Error reading: " << inputPath << "\n";
//                 continue;
//             }

//             // Prepare destination image
//             cv::Mat dst = cv::Mat::zeros(src.size(), src.type());

//             const int rows     = src.rows;
//             const int cols     = src.cols;
//             const int channels = src.channels();

//             // Manual convolution with border replicate
//             for (int y = 0; y < rows; ++y) {
//                 for (int x = 0; x < cols; ++x) {
//                     // Accumulators for each channel
//                     float acc[3] = {0.0f, 0.0f, 0.0f};

//                     // Slide the 3×3 kernel
//                     for (int ky = -1; ky <= 1; ++ky) {
//                         for (int kx = -1; kx <= 1; ++kx) {
//                             // Compute source coordinates with manual border replication:
//                             int yy = y + ky;
//                             if (yy < 0)        yy = 0;
//                             else if (yy >= rows) yy = rows - 1;

//                             int xx = x + kx;
//                             if (xx < 0)        xx = 0;
//                             else if (xx >= cols) xx = cols - 1;

//                             float kval = kernel.at<float>(ky + 1, kx + 1);
//                             cv::Vec3b pix = src.at<cv::Vec3b>(yy, xx);

//                             // Accumulate per‐channel
//                             for (int c = 0; c < channels; ++c) {
//                                 acc[c] += kval * static_cast<float>(pix[c]);
//                             }
//                         }
//                     }

//                     // Write out
//                     cv::Vec3b& outPix = dst.at<cv::Vec3b>(y, x);
//                     for (int c = 0; c < channels; ++c) {
//                         // saturate_cast keeps values in [0-255]
//                         outPix[c] = cv::saturate_cast<uchar>(acc[c]);
//                     }
//                 }
//             }

//             if (!cv::imwrite(outputPath, dst)) {
//                 cerr << "Failed to save: " << outputPath << "\n";
//             } else {
//                 cout << "Processed: " << filename << "\n";
//             }
//         }

//         cout << "All done.\n";
//         return 0;
//     }
//     catch (const exception& e) {
//         cerr << "Fatal Error: " << e.what() << "\n";
//         return 1;
//     }
// }

//-----------------------------------------------------------------------------------------------------------------------------------
// Dynamic kernel size implementation
#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;
using std::cout;
using std::cerr;
using std::endl;
using std::string;

#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <stdexcept>
using namespace std;

namespace fs = filesystem;

// create a high-pass kernel of any size
cv::Mat createHighPassKernel(int size) {
    cv::Mat kernel = cv::Mat::ones(size, size, CV_32F) * -1;
    int center = size / 2;
    kernel.at<float>(center, center) = size * size - 1;
    return kernel;
}

int main() {
    const string baseDir = "E:/Downloads/Semester 10/HPC/Parallel-High-Pass-Filter/";
    const string inDir   = baseDir + "images/input";
    const string outDir  = baseDir + "images/output";

    // Set kernel size 
    const int kernel_size = 9; 

    cout << "CWD:       " << fs::current_path() << "\n";
    cout << "Input dir: " << fs::absolute(inDir) << "\n";

    try {
        if (!fs::exists(inDir) || !fs::is_directory(inDir))
            throw runtime_error("Input directory not found: " + inDir);

        if (!fs::exists(outDir)) {
            fs::create_directories(outDir);
        }

        // Create dynamic kernel
        cv::Mat kernel = createHighPassKernel(kernel_size);
        int kernel_radius = kernel_size / 2;

        for (auto& entry : fs::directory_iterator(inDir)) {
            if (!entry.is_regular_file()) continue;

            const string  inputPath  = entry.path().string();
            const string  filename   = entry.path().filename().string();
            const string  outputPath = outDir + "/" + filename;

            cv::Mat src = cv::imread(inputPath, cv::IMREAD_COLOR);
            if (src.empty()) {
                cerr << "Error reading: " << inputPath << "\n";
                continue;
            }

            cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
            const int rows = src.rows;
            const int cols = src.cols;
            const int channels = src.channels();

            for (int y = 0; y < rows; ++y) {
                for (int x = 0; x < cols; ++x) {
                    float acc[3] = {0.0f, 0.0f, 0.0f};

                    // Dynamic kernel iteration
                    for (int ky = -kernel_radius; ky <= kernel_radius; ++ky) {
                        for (int kx = -kernel_radius; kx <= kernel_radius; ++kx) {
                            int yy = y + ky;
                            yy = max(0, min(yy, rows - 1));
                            
                            int xx = x + kx;
                            xx = max(0, min(xx, cols - 1));

                            // Calculate kernel matrix indices
                            int kernel_y = ky + kernel_radius;
                            int kernel_x = kx + kernel_radius;
                            
                            float kval = kernel.at<float>(kernel_y, kernel_x);
                            cv::Vec3b pix = src.at<cv::Vec3b>(yy, xx);

                            for (int c = 0; c < channels; ++c) {
                                acc[c] += kval * static_cast<float>(pix[c]);
                            }
                        }
                    }

                    cv::Vec3b& outPix = dst.at<cv::Vec3b>(y, x);
                    for (int c = 0; c < channels; ++c) {
                        outPix[c] = cv::saturate_cast<uchar>(acc[c]);
                    }
                }
            }

            if (!cv::imwrite(outputPath, dst)) {
                cerr << "Failed to save: " << outputPath << "\n";
            } else {
                cout << "Processed: " << filename << "\n";
            }
        }

        cout << "All done.\n";
        return 0;
    }
    catch (const exception& e) {
        cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
}