#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
#include <iostream>
#include <algorithm>
#include <cassert>
using namespace std;


cv::Mat initializeKernelMat(int kernel_size);                               // initializes a kernel of dynamic size
cv::Mat convolve(const cv::Mat& paddedSrc, const cv::Mat& kernel);          // performs convolution using custom functions
cv::Mat validConvolution(const cv::Mat& src, const cv::Mat& kernel);        // performs convolution using opencv functions

int main(int argc, char* argv[])
{

    int kernel_size = 3;
    string abs_input_img_path = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/input/lena.png";
    string abs_output_img_path = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/output/lena_MPI_3x3.png";

    MPI_Init(&argc, &argv);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // root processes reads the image
    cv::Mat image;
    int rows, cols;
    if(rank==0){
        image = cv::imread(abs_input_img_path, cv::IMREAD_GRAYSCALE);
        cout << endl << "Number of Rows ("<<image.rows<<") , Number of Columns ("<<image.cols<<")" << endl;
        rows = image.rows;
        cols = image.cols;
        if (image.empty()) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if(!image.isContinuous()) image = image.clone();    // make sure the image is stored continously
    }

    // root broadcasts the image dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // all processes initialize the kernel
    cv::Mat kernel = initializeKernelMat(kernel_size);
    int kernel_radius = (kernel_size - 1) / 2;

    // root process prepares for distribtuting image rows
    vector<int> send_counts;
    vector<int> displacements;
    if(rank == 0){
        const int rows_per_process = rows/size;
        int remaining_rows = rows%size;
        int offset = 0;
        for(int i=0; i<size; i++){
            send_counts.push_back(rows_per_process * cols);
            if(remaining_rows > 0){
                send_counts[i] += cols;
                remaining_rows--;
            }
            displacements.push_back(offset);
            offset += send_counts[i];
        }
    }

    // root process scatters send_counts (so that each process can resize its local_img)
    int n_local_rows;   // number of rows assigned to the process
    MPI_Scatter(send_counts.data(), 1, MPI_INT,
                &n_local_rows, 1, MPI_INT,
                0, MPI_COMM_WORLD);
    n_local_rows /= rows;
    cout<< "Rank (" << rank << "), recieves ("<< n_local_rows <<") rows." << endl ;

    // processes intialize local_img
    cv::Mat local_img = cv::Mat::zeros( n_local_rows + 2*kernel_radius,     // 1 buffer row + inner rows + 1 buffer row
                                        cols,                               
                                        CV_8UC1);

    // root process scatters  the image rows,
    // NB: the local_image receives the rows at it's center, leaving the buffer rows blank                                    
    MPI_Scatterv(   image.data, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                    local_img.ptr(kernel_radius), n_local_rows*rows, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
    if(rank == 1){
        cv::imshow("local of rank 1 before communication", local_img);
        cv::waitKey(0);

    }

    
    // get buffer rows from rank above and below you (respectin the image boundries) 
    int prev_rank = rank -1 , next_rank = rank + 1;
    int top_row_index = kernel_radius;
    int bottom_row_index = n_local_rows;
    int elements_to_send_count = kernel_radius*cols;
    if(prev_rank >= 0){
        MPI_Sendrecv( local_img.ptr(top_row_index), elements_to_send_count, MPI_UNSIGNED_CHAR, prev_rank, 0,
                      local_img.ptr(0), elements_to_send_count, MPI_UNSIGNED_CHAR, prev_rank, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
    }
    if(next_rank < size){
        MPI_Sendrecv( local_img.ptr(bottom_row_index), elements_to_send_count, MPI_UNSIGNED_CHAR, next_rank, 0,
                      local_img.ptr(top_row_index+n_local_rows), elements_to_send_count, MPI_UNSIGNED_CHAR, next_rank, 0,
                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // to make sure all processes communicated the buffer rows before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

    if(rank == 1){
        cv::imshow("local of rank 1 after communication", local_img);
        cv::waitKey(0);
    }
    if(rank == 1) cout << "Rank (" << rank << "), output dimensions before padding: (" << local_img.rows << ", " << local_img.cols << ")" << endl;

    // apply padding to the left and right of the image
    cv::copyMakeBorder(local_img, local_img, 0, 0, kernel_radius, kernel_radius, cv::BORDER_CONSTANT);
    if(rank == 1){
        cout << "Rank (" << rank << "), output dimensions after padding: (" << local_img.rows << ", " << local_img.cols << ")" << endl;
        cv::imshow("local of rank 1 after padding", local_img);
        cv::waitKey(0);
    }
    
    // convolove the local image with the kernel
    local_img = convolve(local_img, kernel);
    if(rank == 1){
        cout << "Rank (" << rank << "), output dimensions after convolution: (" << local_img.rows << ", " << local_img.cols << ")" << endl;
        cv::imshow("local of rank 1 after convolution", local_img);
        cv::waitKey(0);
    }
    // assert(local_img.rows == n_local_rows && local_img.cols == cols);

    // gather the local image back to the root process
    MPI_Gatherv(local_img.data, n_local_rows*cols, MPI_UNSIGNED_CHAR,
                image.data, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);
    
    if(rank == 0){
        cout << "Rank (" << rank << "), output dimensions of the final image: (" << image.rows << ", " << image.cols << ")" << endl;
        cv::imshow("Final gathered image", image);
        cv::waitKey(0);
        cv::imwrite(abs_output_img_path, image);
    }
    


    MPI_Finalize();
    return 0;
    
}                 





cv::Mat initializeKernelMat(int kernel_size) {
    // assert(kernel_size == 3 && "This function only supports a 3×3 kernel");
    return (cv::Mat_<float>(3,3) << 
    0.0f, -1.0f,  0.0f,
   -1.0f,  4.0f, -1.0f,
    0.0f, -1.0f,  0.0f
    );
}


cv::Mat convolve(const cv::Mat& src, const cv::Mat& kernel) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(kernel.rows == kernel.cols && kernel.rows % 2 == 1);

    int ksize  = kernel.rows;
    int kr     = ksize / 2;
    int H      = src.rows, W = src.cols;
    int outH   = H - 2*kr, outW = W - 2*kr;

    cv::Mat dst(outH, outW, CV_8UC1, cv::Scalar(0));

    for (int y = 0; y < outH; ++y) {
        for (int x = 0; x < outW; ++x) {
            float acc = 0.0f;
            int k_center_y = y + kr;
            int k_center_x = x + kr;

            // convolution window centered at (y+kr, x+kr) in src
            for (int ky = -kr; ky <= kr; ++ky) {
                for (int kx = -kr; kx <= kr; ++kx) {
                    int yy = k_center_y + ky;
                    int xx = k_center_x + kx;
                    // yy = std::clamp(yy, 0, H - 1);
                    // xx = std::clamp(xx, 0, W - 1);

                    float kval = kernel.at<float>(ky + kr, kx + kr);
                    uchar pix  = src.at<uchar>(yy, xx);
                    acc += kval * pix;
                }
            }
            dst.at<uchar>(y, x) = cv::saturate_cast<uchar>(acc);
        }
    }
    return dst;
}

cv::Mat validConvolution(const cv::Mat& src, const cv::Mat& kernel) {
    // perform full convolution
    cv::Mat full;
    cv::filter2D(src, full, -1, kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);  // same size as src :contentReference[oaicite:3]{index=3}

    
    int kr = kernel.rows / 2;  
    cv::Rect roi(kr, kr, src.cols - 2*kr, src.rows - 2*kr);

    // crop the valid region
    cv::Mat valid = full(roi).clone();  // .clone() if you need own buffer :contentReference[oaicite:4]{index=4}
    return valid;
}

