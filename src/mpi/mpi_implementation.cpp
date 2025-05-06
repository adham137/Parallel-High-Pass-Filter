#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
#include <iostream>
#include <cassert>
using namespace std;


template<typename T>
constexpr const T& my_clamp(const T& v, const T& lo, const T& hi) {
    return (v < lo) ? lo : (hi < v) ? hi : v;
}

cv::Mat initializeKernelMat(int kernel_size);                         // initializes a kernel of dynamic size
cv::Mat applyConvolution(const cv::Mat& src, const cv::Mat& kernel);       // convolve the src mtx with the filter

int main(int argc, char* argv[])
{
    string abs_input_img_path = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/input/lena.png";
    string abs_output_img_path = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/output";
    int kernel_size = 31;

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
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 1){
        cv::imshow("local of rank 1 after communication", local_img);
        cv::waitKey(0);
    }

    // apply convulution
    cv::Mat local_res = applyConvolution(local_img, kernel);
    if(rank == 1){
        cv::imshow("local of rank 1 after convultion", local_res);
        cv::waitKey(0);
    }


    MPI_Finalize();
    return 0;
    
}                                         

cv::Mat initializeKernelMat(int kernel_size) {
    // assert(kernel_size == 3 && "This function only supports a 3×3 kernel");

    
    cv::Mat kernel = (cv::Mat_<int>(3, 3) <<
         0, -1,  0,
        -1,  4, -1,
         0, -1,  0);

    
    return kernel;
}

cv::Mat applyConvolution(const cv::Mat& src, const cv::Mat& kernel) {
    CV_Assert(src.type() == CV_8UC1);
    CV_Assert(kernel.rows == kernel.cols && kernel.rows % 2 == 1);

    int ksize = kernel.rows;
    int kr    = ksize / 2;

    // Original dimensions
    int origRows = src.rows;
    int origCols = src.cols;

    // Output is smaller by 2*kr in each dim
    int outRows = origRows - 2*kr;
    int outCols = origCols - 2*kr;
    cv::Mat dst(outRows, outCols, CV_8UC1);

    for (int y = 0; y < outRows; ++y) {
        for (int x = 0; x < outCols; ++x) {
            float acc = 0.0f;

            // Center of kernel at (y+kr, x+kr) in original image
            for (int ky = -kr; ky <= kr; ++ky) {
                for (int kx = -kr; kx <= kr; ++kx) {
                    int yy = y + ky + kr;              // map to original coords
                    int xx = x + kx + kr;

                    // my_clamp against ORIGINAL image
                    yy = my_clamp(yy, 0, origRows - 1);
                    xx = my_clamp(xx, 0, origCols - 1);

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
