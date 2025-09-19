#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernEffScan(int* g_odata, int* g_idata, int n, int* blockSum) {
            extern __shared__ int temp[];
            int thid = threadIdx.x;
            int offset = 1;
            temp[2 * thid] = g_idata[2 * thid];
            temp[2 * thid + 1] = g_idata[2 * thid + 1];

            for (int d = n >> 1; d > 0; d = d >> 1) {
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    temp[bi] += temp[ai];
                }
                offset *= 2;
            }

            if (thid == 0) {
                if (blockSum) {
                    blockSum[0] = temp[n - 1];  // Save total sum of this block
                }
                temp[n - 1] = 0;
            }

            for (int d = 1; d < n; d *= 2) {
                offset = offset >> 1;
                __syncthreads();
                if (thid < d) {
                    int ai = offset * (2 * thid + 1) - 1;
                    int bi = offset * (2 * thid + 2) - 1;

                    int t = temp[ai];
                    temp[ai] = temp[bi];
                    temp[bi] += t;
                }
            }

            __syncthreads();

            if (!blockSum) { 
                // exclusive scan
                g_odata[2 * thid] = temp[2 * thid];
                g_odata[2 * thid + 1] = temp[2 * thid + 1];
            }
            else {
				// inclusive scan
                g_odata[2 * thid] = temp[2 * thid + 1];
                g_odata[2 * thid + 1] = temp[2 * thid + 2];
                if (thid == 0) {
                    g_odata[n - 1] = blockSum[0];
				}
            }
            
        }


        __global__ void kernAddBlockSums(int n, int* odata, int blockSum) {
            int index = threadIdx.x;

            odata[2 * index] += blockSum;
            odata[2 * index + 1] += blockSum;
        }

        __global__ void kernShiftRight(int n, int* odata, const int* idata) {
            int index = (blockIdx.x * blockDim.x) + threadIdx.x;
            if (index >= n) return;
            odata[index] = (index == 0) ? 0 : idata[index - 1];
        }

        int nextPowerOf2(int n) {
            n--;
            n |= n >> 1;
            n |= n >> 2;
            n |= n >> 4;
            n |= n >> 8;
            n |= n >> 16;
            n++;
            return n;
        }

        __global__ void kernUpSweep(int n, int* data, int level) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int stride = 1 << (level + 1);

            //  Only process if this thread has work to do
            int numActiveThreads = n / stride;
            if (index >= numActiveThreads) return;

			int ai = stride * (index + 1) - 1;
            if (ai < n) {
                int bi = ai - (1 << level);
                data[ai] += data[bi];
			}
        }

		__global__ void kernDownSweep(int n, int* data, int level) {
			int index = (blockIdx.x * blockDim.x) + threadIdx.x;
			int stride = 1 << (level + 1);

            //  Only process if this thread has work to do
            int numActiveThreads = n / stride;
            if (index >= numActiveThreads) return;

			int ai = stride * (index + 1) - 1;

			if (ai < n) {
				int bi = ai - (1 << level);
				int t = data[bi];
				data[bi] = data[ai];
				data[ai] += t;
			}
		}



        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            int paddedSize = nextPowerOf2(n);
			const int blockSize = 256;

            if (paddedSize <= 2 * blockSize) {
                int* d_idata;
                int* d_odata;
                cudaMalloc((void**)&d_idata, paddedSize * sizeof(int));
                cudaMalloc((void**)&d_odata, paddedSize * sizeof(int));

                cudaMemset(d_idata, 0, paddedSize * sizeof(int));
                cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

                int numThreads = paddedSize / 2;

                kernEffScan<<<1, numThreads, n * sizeof(int)>>>(d_odata, d_idata, paddedSize, nullptr);
                cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

                cudaFree(d_idata);
                cudaFree(d_odata);

            } else {
                int* d_data;

				size_t bytes = paddedSize * sizeof(int);

                cudaMalloc((void**)&d_data, bytes);

                // Check for allocation failure
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
                    timer().endGpuTimer();
                    return;
                }

                cudaMemset(d_data, 0, bytes);
                cudaMemcpy(d_data, idata, bytes, cudaMemcpyHostToDevice);

                // For large arrays, use multiple kernel launches
                int depth = ilog2ceil(paddedSize);


                // Up-sweep phase
                for (int level = 0; level < depth; level++) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    int launchedBlockSize = numActiveThreads < blockSize ? numActiveThreads : blockSize;
                    kernUpSweep << <numBlocks, launchedBlockSize >> > (paddedSize, d_data, level);
                }


                // Set last element to 0 for exclusive scan
                cudaMemset(d_data + paddedSize - 1, 0, sizeof(int));

                // Down-sweep phase
                for (int level = depth - 1; level >= 0; level--) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    int launchedBlockSize = numActiveThreads < blockSize ? numActiveThreads : blockSize;
                    kernDownSweep << <numBlocks, launchedBlockSize >> > (paddedSize, d_data, level);
                }


                // Copy result from device to host
                cudaMemcpy(odata, d_data, bytes, cudaMemcpyDeviceToHost);
                cudaFree(d_data);
            }

            

            timer().endGpuTimer();
        }

        // Wrapper for compact, removes timer and takes in device pointers
        void scanCompact(int n, int* odata, const int* idata) {
            // TODO
            int paddedSize = nextPowerOf2(n);
			const int blockSize = 256;  

            if (paddedSize <= 2 * blockSize) {
                int* d_idata;
                int* d_odata;
                cudaMalloc((void**)&d_idata, paddedSize * sizeof(int));
                cudaMalloc((void**)&d_odata, paddedSize * sizeof(int));

                cudaMemset(d_idata, 0, paddedSize * sizeof(int));
                cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

                int numThreads = paddedSize / 2;

                kernEffScan << <1, numThreads, n * sizeof(int) >> > (d_odata, d_idata, paddedSize, nullptr);
                cudaMemcpy(odata, d_odata, n * sizeof(int), cudaMemcpyDeviceToDevice);

                cudaFree(d_idata);
                cudaFree(d_odata);
            }
            else {
                int* d_data;
                cudaMalloc((void**)&d_data, paddedSize * sizeof(int));
                cudaMemset(d_data, 0, paddedSize * sizeof(int));
                cudaMemcpy(d_data, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);

                // For large arrays, use multiple kernel launches
                int depth = ilog2ceil(paddedSize);

                // Up-sweep phase
                for (int level = 0; level < depth; level++) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    int launchedBlockSize = numActiveThreads < blockSize ? numActiveThreads : blockSize;
                    kernUpSweep << <numBlocks, launchedBlockSize >> > (paddedSize, d_data, level);
                }

                // Set last element to 0
                cudaMemset(d_data + paddedSize - 1, 0, sizeof(int));

                // Down-sweep phase
                for (int level = depth - 1; level >= 0; level--) {
                    int numActiveThreads = paddedSize / (1 << (level + 1));
                    dim3 numBlocks((numActiveThreads + blockSize - 1) / blockSize);
                    int launchedBlockSize = numActiveThreads < blockSize ? numActiveThreads : blockSize;
                    kernDownSweep << <numBlocks, launchedBlockSize >> > (paddedSize, d_data, level);
                }

                // Copy to output
                cudaMemcpy(odata, d_data, n * sizeof(int), cudaMemcpyDeviceToDevice);
                cudaFree(d_data);
            }

        }



        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
			int* d_idata;
			int* d_odata;

			int* d_bools;
			int* d_indices;

			cudaMalloc((void**)&d_idata, n * sizeof(int));
			cudaMalloc((void**)&d_odata, n * sizeof(int));
			cudaMalloc((void**)&d_bools, n * sizeof(int));
			cudaMalloc((void**)&d_indices, n * sizeof(int));

			cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			cudaDeviceSynchronize();

			int blockSize = 1024;
			dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);

			StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> > (n, d_bools, d_idata);
			cudaDeviceSynchronize();

			scanCompact(n, d_indices, d_bools);
			cudaDeviceSynchronize();

			StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> > (n, d_odata, d_idata, d_bools, d_indices);
			cudaDeviceSynchronize();

			int lastBool, lastIndex;
			cudaMemcpy(&lastBool, d_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
			cudaMemcpy(&lastIndex, d_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

			int count = lastBool + lastIndex;
			cudaMemcpy(odata, d_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();

			cudaFree(d_idata);
			cudaFree(d_odata);
			cudaFree(d_bools);
			cudaFree(d_indices);

            timer().endGpuTimer();
            return count;
        }
    }
}
