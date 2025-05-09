#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <mpi.h>
#include <iostream>
using namespace std;


vector<int> initialize_kernel(int kernel_size);     // initializes a kernel of dynamic size
vector<uchar> matrix_to_bytes(cv::Mat matrix);      // convert matrix type to vector of bytes to be able to send it
void bytes_to_matrix(uchar*, vector<uchar> bytes, int kernel_radius, int cols); // copy the bytes array into the center of the matrix

int main(int argc, char* argv[])
{
    string abs_input_img_path = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/input/lena.png";
    string abs_output_img_path = "D:/ASU/sem 10/HPC/Parallel_High_Pass_Filter/images/output";
    int kernel_size = 3;

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
    vector<int> kernel = initialize_kernel(kernel_size);
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

    // root process scatters the image row wise

    // vector<uchar> global_data;
    // vector<uchar> local_data (n_local_rows*rows);
    // if(rank==0){
    //     global_data = matrix_to_bytes(image); 
    // }
    // MPI_Scatterv(   global_data.data(), send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
    //                 local_data.data(), n_local_rows*rows, MPI_UNSIGNED_CHAR,
    //                 0, MPI_COMM_WORLD);
    //cout <<"Rank (" << rank << "), local data vector is of size: ("<< (local_data.size()) <<"). "<< "First 5 elements casted as int: " << (int)local_data[0] << ", " << (int)local_data[1] << ", " << (int)local_data[2] << ", " << (int)local_data[3] << ", " << (int)local_data[4] <<endl;

    MPI_Scatterv(   image.data, send_counts.data(), displacements.data(), MPI_UNSIGNED_CHAR,
                    local_img.ptr(kernel_radius), n_local_rows*rows, MPI_UNSIGNED_CHAR,
                    0, MPI_COMM_WORLD);
    // if(rank == 0){
    //     cv::imshow("local image of rank 0", local_img);
    //     cv::waitKey(0);

    // }




    // // fillout the inner part of the local_img
    // bytes_to_matrix(local_img.data, local_data, kernel_radius, cols);
    
    // get buffer rows from rank above and below you (respectin the image boundries) 
    int prev_rank = rank -1 , next_rank = rank + 1;



    MPI_Finalize();
    return 0;
    
}                                         
vector<int> initialize_kernel(int kernel_size) {
    vector<int> kernel = { 0, -1, 0,
                          -1, 4, -1,
                           0, -1, 0  };

    return kernel;
}

vector<uchar> matrix_to_bytes(cv::Mat matrix){
    return vector<uchar>(matrix.datastart, matrix.dataend);
}